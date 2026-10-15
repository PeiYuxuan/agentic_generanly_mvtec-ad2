from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "foundad"))

from src.datasets.dataset import TestDataset

dataset = TestDataset(
    source="/home/peiyuxuan/projects/data/mvtec_ad_2",
    classname="can",
    resize=512,
    datasetname="mvtec_ad2",
)

normal_count = sum(
    item[1] == "good" for item in dataset.data_to_iterate
)
anomaly_count = sum(
    item[1] == "bad" for item in dataset.data_to_iterate
)

print("Number of normal test images: ", normal_count)
print("Number of abnormal test images: ", anomaly_count)

# Read one normal image and one abnormal image each, and check if they can be opened
for anomaly in ("good", "bad"):
    index = next(
        i for i, item in enumerate(dataset.data_to_iterate)
        if item[1] == anomaly
    )
    sample = dataset[index]
    print(f"\ncategory: {anomaly}")
    print("image path: ", sample["image_path"])
    print("image shape: ", tuple(sample["image"].shape))
    print("mask shape: ", tuple(sample["mask"].shape))

print("\nAD 2 Test data reading check passed")