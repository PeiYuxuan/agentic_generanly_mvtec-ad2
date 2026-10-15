import argparse
import math
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "logs/mvtec_ad2_all_alltrain/dinov3_ad2_smoke"
    / "eval/1896/eval_params.yaml"
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
parser.add_argument("--categories", nargs="+", default=None)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()

with args.config.open("r") as file:
    cfg = yaml.safe_load(file)

# Set GPU visibility before importing PyTorch.
device_name = cfg.get("devices", ["cuda:0"])[0]
if device_name.startswith("cuda:"):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_name.split(":")[1]

sys.path.insert(0, str(PROJECT_ROOT / "foundad"))

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.datasets.dataset import build_base_transform
from src.foundad import VisionModule


EXPECTED_COUNTS = {
    "can": 321,
    "fabric": 314,
    "fruit_jelly": 255,
    "rice": 277,
    "sheet_metal": 142,
    "vial": 276,
    "wallplugs": 232,
    "walnuts": 228,
}

SPLITS = ("test_private", "test_private_mixed")


def main():
    categories = args.categories or cfg["data"]["classnames"]

    for category in categories:
        if category not in EXPECTED_COUNTS:
            raise ValueError(f"Unknown category: {category}")

    data_root = Path(cfg["data"]["test_root"])
    checkpoint = Path(cfg["ckpt_path"])

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_root = args.output or (
        checkpoint.parent / "private_submission" / checkpoint.stem
    )

    # Check all input directories before running inference.
    image_lists = {}
    for category in categories:
        for split in SPLITS:
            directory = data_root / category / split
            paths = sorted(directory.glob("*.png"))
            expected = EXPECTED_COUNTS[category]

            if len(paths) != expected:
                raise ValueError(
                    f"{category}/{split}: found {len(paths)} images, "
                    f"expected {expected}. Directory: {directory}"
                )

            image_lists[(category, split)] = paths

    meta = cfg["meta"]
    crop_size = int(meta["crop_size"])
    n_layer = int(meta.get("n_layer", 3))
    transform = transforms.Compose(build_base_transform(crop_size))

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Input size: {crop_size} x {crop_size}", flush=True)
    print(f"Output directory: {output_root}", flush=True)

    model = VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
    )

    state = torch.load(checkpoint, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])

    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])

    del state
    model.to(device).eval()

    output_root.mkdir(parents=True, exist_ok=True)

    # Save essential settings without the recursive YAML "app" entry.
    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(args.config),
        "meta": meta,
        "dataset_root": str(data_root),
        "trained_categories": cfg["data"]["classnames"],
        "training_epochs": cfg["optimization"]["epochs"],
        "output_dtype": "float16",
        "output_resolution": [crop_size, crop_size],
        "score_normalization": "none",
    }

    with (output_root / "export_config.yaml").open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    with torch.inference_mode():
        for category in categories:
            for split in SPLITS:
                paths = image_lists[(category, split)]
                destination = (
                    output_root / "anomaly_images" / category / split
                )
                destination.mkdir(parents=True, exist_ok=True)

                written = 0
                skipped = 0

                for index, image_path in enumerate(paths, start=1):
                    output_path = destination / (
                        image_path.stem + ".tiff"
                    )

                    with Image.open(image_path) as source_image:
                        original_width, original_height = source_image.size

                        # Resume only if existing TIFF metadata is valid.
                        if output_path.exists():
                            with tifffile.TiffFile(output_path) as existing:
                                page = existing.pages[0]
                                valid = (
                                    len(existing.pages) == 1
                                    and page.shape == (crop_size, crop_size)
                                    and page.dtype == np.dtype("float16")
                                )

                            if not valid:
                                raise ValueError(
                                    f"Invalid existing TIFF: {output_path}"
                                )

                            skipped += 1
                            continue

                        image = transform(
                            source_image.convert("RGB")
                        ).unsqueeze(0).to(device)

                    features = model.target_features(
                        image,
                        [str(image_path)],
                        n_layer=n_layer,
                    )
                    prediction = model.predict(features)

                    patch_scores = F.mse_loss(
                        features, prediction, reduction="none"
                    ).mean(dim=2)

                    grid_size = math.isqrt(patch_scores.shape[1])
                    if grid_size * grid_size != patch_scores.shape[1]:
                        raise ValueError("Patch grid is not square")

                    score_map = patch_scores.reshape(
                        1, 1, grid_size, grid_size
                    )

                    # Match the public evaluation's interpolation first.
                    score_map = F.interpolate(
                        score_map,
                        size=image.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                    scores = (
                        score_map[0, 0].float().cpu().numpy()
                    )

                    if not np.isfinite(scores).all():
                        raise ValueError(
                            f"Non-finite scores: {image_path}"
                        )

                    scores = scores.astype(np.float16)

                    if not np.isfinite(scores).all():
                        raise ValueError(
                            f"Float16 overflow: {image_path}"
                        )

                    # Write atomically so an interrupted write can be retried.
                    temporary_path = output_path.with_suffix(".tiff.tmp")
                    tifffile.imwrite(
                        temporary_path,
                        scores,
                        photometric="minisblack",
                        metadata=None,
                    )
                    temporary_path.replace(output_path)
                    written += 1

                    if index == 1 or index % 25 == 0:
                        print(
                            f"{category}/{split}: "
                            f"{index}/{len(paths)} | "
                            f"shape={scores.shape} | "
                            f"range=[{float(scores.min()):.6g}, "
                            f"{float(scores.max()):.6g}]",
                            flush=True,
                        )

                print(
                    f"Finished {category}/{split}: "
                    f"written={written}, skipped={skipped}, "
                    f"total={len(paths)}",
                    flush=True,
                )

    print("Private prediction export completed.", flush=True)
    print(f"Submission directory: {output_root}", flush=True)


if __name__ == "__main__":
    main()