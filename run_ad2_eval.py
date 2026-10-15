from pathlib import Path
import os
import sys

import numpy as np
from PIL import Image
import yaml

project_root = Path(__file__).resolve().parent
log_dir = (
    project_root
    / "logs"
    / "mvtec_ad2_all_alltrain"
    / "dinov3_ad2_smoke"
)
checkpoint_step = 1896
config_path = log_dir / "params.yaml"
checkpoint_path = (
    log_dir / f"train-step{checkpoint_step}.pth.tar"
)

if not checkpoint_path.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

with config_path.open() as file:
    config = yaml.safe_load(file)

# Use the GPU selected for the training run.
gpu_index = str(config["devices"][0]).split(":")[-1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

# Verify binary mask labels for every evaluation category.
for classname in config["data"]["classnames"]:
    mask_dir = (
        Path(config["data"]["test_root"])
        / classname
        / "test_public"
        / "ground_truth"
        / "bad"
    )
    mask_paths = sorted(mask_dir.glob("*_mask.png"))

    if not mask_paths:
        raise RuntimeError(
            f"No ground-truth masks found for {classname}."
        )
    mask_values = set()
    for path in mask_paths:
        with Image.open(path) as mask:
            mask_values.update(
                np.unique(np.asarray(mask)).tolist()
            )
    print(
        f"{classname}: mask pixel values "
        f"{sorted(mask_values)}",
        flush=True,
    )
    if not mask_values.issubset({0, 255}):
        raise RuntimeError(
            f"Unexpected mask encoding for {classname}: "
            f"{sorted(mask_values)}"
        )

config["ckpt_path"] = str(checkpoint_path)
config["logging"]["folder"] = str(
    log_dir / "eval" / str(checkpoint_step)
)
config["testing"]["K_top_ad2"] = 10

eval_dir = Path(config["logging"]["folder"])
eval_dir.mkdir(parents=True, exist_ok=True)

with (eval_dir / "eval_params.yaml").open("w") as file:
    yaml.safe_dump(config, file, sort_keys=False)

# Make FoundAD's internal modules available for import.
sys.path.insert(0, str(project_root / "foundad"))

print("Importing evaluation module...", flush=True)
from src.AD import main

print("Starting evaluation...", flush=True)
main(config)