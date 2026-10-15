from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "foundad"))

from src.datasets.dataset import TrainDataset

dataset = TrainDataset(
    root="/home/peiyuxuan/projects/data/mvtec_ad_2",
    resize=512,
    classnames=["can"],
)

image, category, image_path = dataset[0]

print("Category:", category)
print("Image path:", image_path)
print("Image shape:", tuple(image.shape))
print("Training data loading passed.")