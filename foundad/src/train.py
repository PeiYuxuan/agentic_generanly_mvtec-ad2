from __future__ import annotations

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys, random, logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import yaml, numpy as np, torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from src.utils.logging import gpu_timer, grad_logger, AverageMeter
from src.datasets.dataset import build_dataloader
from src.utils.synthesis import CutPasteUnion
from src.foundad import VisionModule

_GLOBAL_SEED = 0
random.seed(42); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args: Dict[str, Any]):
        # ---------- basic ----------
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # ---------- model ----------
        mcfg = args["meta"]
        self.model = VisionModule(
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"], if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2") # l2 or smooth_l1
        logger.info(f"Loss mode {self.loss_mode}")

        # ---------- data ----------
        dcfg = args["data"]
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            classnames=dcfg["classnames"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip",False),
            use_vflip=dcfg.get("use_vflip",False),
            use_rotate90=dcfg.get("use_rotate90",False),
            use_color_jitter=dcfg.get("use_color_jitter",False),
            use_gray=dcfg.get("use_gray",False),
            use_blur=dcfg.get("use_blur",False),
        )
        self.cutpaste = CutPasteUnion(colorJitter=0.5)
        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        from src.helper import init_opt

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.model.predictor,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],                         # for cosine_warmup
            min_lr=ocfg.get("min_lr", 1e-6),                  # for cosine_warmup
            warmup_epoch=ocfg.get("warmup_epoch", 5),         # for cosine_warmup
            step_size=ocfg.get("step_size", 300),             # for step
            gamma=ocfg.get("gamma", 0.1),                     # for step
        )
        self.epochs = ocfg["epochs"]
        self.use_bf16 = mcfg["use_bfloat16"]

        # ---------- logging ----------
        lcfg = args.get("logging", {})
        self.ckpt_dir = Path(lcfg.get("folder", "logs"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tag = lcfg.get("write_tag", "train")

        monitor = args.get("monitoring", {})
        self.console_updates = max(
            1, int(monitor.get("console_updates_per_epoch", 10))
        )
        self.save_every_epochs = max(
            1, int(monitor.get("save_every_epochs", 5))
        )
        self.eval_every_epochs = max(
            1, int(monitor.get("eval_every_epochs", 5))
        )
        self.smoothing_window = monitor.get("smoothing_window")
        if self.smoothing_window is not None:
            self.smoothing_window = max(1, int(self.smoothing_window))

        self.max_plot_points = max(
            100, int(monitor.get("max_plot_points", 3000))
        )

        self.step_csv = self.ckpt_dir / f"{self.tag}_steps.csv"
        self.epoch_csv = self.ckpt_dir / f"{self.tag}_epochs.csv"
        self.eval_csv = self.ckpt_dir / f"{self.tag}_eval_history.csv"
        self.curve_path = self.ckpt_dir / f"{self.tag}_curves.png"

        for path in (self.step_csv, self.epoch_csv, self.eval_csv):
            if path.exists():
                raise FileExistsError(
                    f"Training history already exists: {path}. "
                    "Use a new experiment name to start a new run."
                )

        logger.info("Training output directory: %s", self.ckpt_dir)

    def _loss_fn(self, h, p) -> torch.Tensor:
        if self.loss_mode == 'l2':
            return F.mse_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _save_ckpt(self, epoch, step):
        path = self.ckpt_dir / f"{self.tag}-ep{epoch:04d}.pth.tar"
        temporary_path = path.with_name(path.name + ".tmp")

        state = {
            "predictor": self.model.predictor.state_dict(),
            "projector": (
                self.model.projector.state_dict()
                if self.model.projector is not None else None
            ),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict()
                if self.scheduler is not None else None
            ),
            "scaler": (
                self.scaler.state_dict()
                if self.scaler is not None else None
            ),
            "epoch": epoch,
            "global_step": step,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        torch.save(state, temporary_path)
        temporary_path.replace(path)
        logger.info("Checkpoint saved: %s", path)

    def _evaluate_epoch(self, epoch, global_step):
        from src.AD import evaluate_model

        # Use a separate directory for each evaluation.
        eval_cfg = dict(self.args)
        eval_cfg["logging"] = dict(self.args["logging"])
        eval_cfg["logging"]["folder"] = str(
            self.ckpt_dir / "eval" / f"ep{epoch:04d}"
        )

        eval_cfg["testing"] = dict(self.args.get("testing", {}))
        eval_cfg["testing"]["segmentation_vis"] = False

        # Keep evaluation from changing training RNG sequences.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cuda_devices = (
            [torch.cuda.current_device()]
            if self.device.type == "cuda" else []
        )

        logger.info(
            "Starting public evaluation at epoch %d. "
            "No parameter updates will be performed.",
            epoch,
        )

        try:
            with torch.random.fork_rng(devices=cuda_devices):
                result = evaluate_model(
                    self.model,
                    eval_cfg,
                    checkpoint_name=(
                        f"{self.tag}-ep{epoch:04d}.pth.tar"
                    ),
                )
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

        fieldnames = [
            "epoch",
            "global_step",
            "inst_auroc",
            "inst_aupr",
            "pix_auroc",
            "au_pro_005",
            "au_pro_030",
        ]
        write_header = not self.eval_csv.exists()

        with self.eval_csv.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            writer.writerow({
                "epoch": epoch,
                "global_step": global_step,
                **result,
            })

        logger.info(
            "Epoch %d public evaluation: "
            "AU-PRO_0.05 %.4f | AU-PRO_0.30 %.4f",
            epoch,
            result["au_pro_005"],
            result["au_pro_030"],
        )

    def _plot_curves(self):
        with self.step_csv.open(newline="") as file:
            rows = list(csv.DictReader(file))

        with self.epoch_csv.open(newline="") as file:
            epoch_rows = list(csv.DictReader(file))

        if not rows or not epoch_rows:
            return

        steps = np.array([int(row["global_step"]) for row in rows])
        losses = np.array([float(row["loss"]) for row in rows])

        epochs = np.array([int(row["epoch"]) for row in epoch_rows])
        epoch_losses = np.array([
            float(row["mean_loss"]) for row in epoch_rows
        ])

        window = self.smoothing_window
        if window is None:
            window = min(500, max(1, len(losses) // 100))
        window = min(window, len(losses))

        smooth_losses = np.convolve(
            losses, np.ones(window) / window, mode="valid"
        )
        smooth_steps = steps[window - 1:]

        raw_stride = max(
            1, math.ceil(len(steps) / self.max_plot_points)
        )
        smooth_stride = max(
            1, math.ceil(len(smooth_steps) / self.max_plot_points)
        )

        fig, axes = plt.subplots(3, 1, figsize=(11, 11))

        axes[0].plot(
            steps[::raw_stride],
            losses[::raw_stride],
            color="steelblue",
            alpha=0.25,
            linewidth=0.7,
            label="Raw loss (display subsampled)",
        )
        axes[0].plot(
            smooth_steps[::smooth_stride],
            smooth_losses[::smooth_stride],
            color="darkblue",
            linewidth=1.5,
            label=f"Moving average (window={window})",
        )
        axes[0].set_xlabel("Global step")
        axes[0].set_ylabel("Training loss")
        axes[0].set_title("Training loss by step")
        axes[0].legend()

        axes[1].plot(epochs, epoch_losses, marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean training loss")
        axes[1].set_title("Mean training loss by epoch")

        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AU-PRO (%)")
        axes[2].set_title(
            "Public evaluation AU-PRO (512 x 512)"
        )
        axes[2].set_ylim(0, 100)

        evaluation_rows = []
        if self.eval_csv.exists():
            with self.eval_csv.open(newline="") as file:
                evaluation_rows = list(csv.DictReader(file))

        if evaluation_rows:
            evaluation_epochs = [
                int(row["epoch"]) for row in evaluation_rows
            ]
            pro_005 = [
                100 * float(row["au_pro_005"])
                for row in evaluation_rows
            ]
            pro_030 = [
                100 * float(row["au_pro_030"])
                for row in evaluation_rows
            ]

            axes[2].plot(
                evaluation_epochs,
                pro_005,
                marker="o",
                label="AU-PRO 0.05",
            )
            axes[2].plot(
                evaluation_epochs,
                pro_030,
                marker="s",
                label="AU-PRO 0.30",
            )
            axes[2].legend()
        else:
            axes[2].text(
                0.5,
                0.5,
                "Evaluation pending",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )

        for axis in axes:
            axis.grid(alpha=0.25)

        fig.tight_layout()
        try:
            fig.savefig(self.curve_path, dpi=150)
        finally:
            plt.close(fig)

        logger.info("Training curves updated: %s", self.curve_path)

    def train(self):
        mp.set_start_method("spawn", force=True)

        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if len(self.loader) == 0:
            raise ValueError("Training dataloader is empty")

        global_step = 0
        print_every = max(
            1, math.ceil(len(self.loader) / self.console_updates)
        )

        with (
            self.step_csv.open("x", newline="", buffering=1) as step_file,
            self.epoch_csv.open("x", newline="", buffering=1) as epoch_file,
        ):
            step_writer = csv.writer(step_file)
            epoch_writer = csv.writer(epoch_file)

            step_writer.writerow([
                "epoch", "iteration", "global_step",
                "loss", "lr", "forward_time_ms",
            ])
            epoch_writer.writerow([
                "epoch", "global_step", "mean_loss", "lr",
            ])

            self.optimizer.zero_grad()

            for ep in range(self.epochs):
                epoch = ep + 1
                logger.info("Epoch %d/%d", epoch, self.epochs)

                if self.sampler is not None:
                    self.sampler.set_epoch(ep)

                loss_meter = AverageMeter()
                time_meter = AverageMeter()
                epoch_lr = self.optimizer.param_groups[0]["lr"]

                for itr, (imgs, labels, paths) in enumerate(self.loader):
                    imgs = imgs.to(self.device, non_blocking=True)
                    _, imgs_abn = self.cutpaste(imgs, labels)

                    def _step():
                        with autocast(
                            dtype=torch.bfloat16,
                            enabled=self.use_bf16,
                        ):
                            h = self.model.target_features(
                                imgs, paths, n_layer=self.n_layer
                            )

                            if np.random.rand() < 0.5:
                                context_images = imgs
                            else:
                                context_images = imgs_abn

                            _, prediction = self.model.context_features(
                                context_images,
                                paths,
                                n_layer=self.n_layer,
                            )
                            return self._loss_fn(h, prediction)

                    (loss,), elapsed_ms = gpu_timer(lambda: [_step()])

                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(
                            f"Non-finite loss at epoch={epoch}, "
                            f"iteration={itr}"
                        )

                    lr = self.optimizer.param_groups[0]["lr"]

                    if self.use_bf16:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    grad_stats = grad_logger(
                        self.model.predictor.named_parameters()
                    )
                    self.optimizer.zero_grad()

                    loss_value = loss.item()
                    loss_meter.update(loss_value)
                    time_meter.update(elapsed_ms)
                    global_step += 1

                    step_writer.writerow([
                        epoch,
                        itr + 1,
                        global_step,
                        loss_value,
                        lr,
                        elapsed_ms,
                    ])

                    if (
                        itr % print_every == 0
                        or itr + 1 == len(self.loader)
                    ):
                        memory_mb = (
                            torch.cuda.max_memory_allocated() / 1024**2
                            if self.device.type == "cuda" else 0.0
                        )
                        logger.info(
                            "[E %d I %d/%d S %d] "
                            "loss %.6f (avg %.6f) "
                            "mem %.2fMB (forward %.1fms)",
                            epoch,
                            itr + 1,
                            len(self.loader),
                            global_step,
                            loss_value,
                            loss_meter.avg,
                            memory_mb,
                            time_meter.avg,
                        )

                        if grad_stats:
                            logger.info(
                                "    grad: [%.2e %.2e] (%.2e %.2e)",
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                            )

                epoch_writer.writerow([
                    epoch, global_step, loss_meter.avg, epoch_lr
                ])
                step_file.flush()
                epoch_file.flush()

                logger.info(
                    "Epoch %d complete. Avg loss %.6f, lr %.6f",
                    epoch, loss_meter.avg, epoch_lr,
                )

                if self.scheduler is not None:
                    self.scheduler.step()

                should_evaluate = (
                    epoch == 1
                    or epoch % self.eval_every_epochs == 0
                    or epoch == self.epochs
                )

                should_save = (
                    epoch % self.save_every_epochs == 0
                    or epoch == self.epochs
                    or should_evaluate
                )

                if should_save:
                    self._save_ckpt(epoch, global_step)

                # Save loss curves even if evaluation later fails.
                self._plot_curves()

                if should_evaluate:
                    self._evaluate_epoch(epoch, global_step)
                    self._plot_curves()

        logger.info("Training completed. Output: %s", self.ckpt_dir)
                
def main(args: Dict[str, Any]) -> None:
    if args is None:
        cfg_path = Path(__file__).with_name("params.yaml");
        if not cfg_path.exists(): raise FileNotFoundError("No args provided and default parameter file does not exist")
        with open(cfg_path) as f: args = yaml.safe_load(f)
    Trainer(args).train()

if __name__ == "__main__":
    main()