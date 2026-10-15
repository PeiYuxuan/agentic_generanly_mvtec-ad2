from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "foundad"))

from src.datasets.dataset import TrainDataset, TestDataset

data_root = "/home/peiyuxuan/projects/data/mvtec_ad_2"
classnames = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]

canvas = Image.new("RGB", (4 * 256, 2 * 280), "white")
draw = ImageDraw.Draw(canvas)

for index, classname in enumerate(classnames):
    train_dataset = TrainDataset(
        root=data_root,
        resize=512,
        classnames=[classname],
    )
    test_dataset = TestDataset(
        source=data_root,
        classname=classname,
        resize=512,
        datasetname="mvtec_ad2",
    )

    normal_count = sum(
        item[1] == "good"
        for item in test_dataset.data_to_iterate
    )
    abnormal_count = sum(
        item[1] == "bad"
        for item in test_dataset.data_to_iterate
    )

    # Read one sample from each test group.
    for group in ("good", "bad"):
        sample_index = next(
            i for i, item in enumerate(test_dataset.data_to_iterate)
            if item[1] == group
        )
        sample = test_dataset[sample_index]
        if group == "bad":
            values = np.unique(sample["mask"].numpy())
            if not np.isin(values, [0, 1]).all():
                raise RuntimeError(
                    f"Unexpected mask values for {classname}: {values}"
                )

    print(
        f"{classname}: train={len(train_dataset)}, "
        f"test_good={normal_count}, test_bad={abnormal_count}"
    )

    image_path = train_dataset.samples[0][0]
    with Image.open(image_path) as image:
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((256, 256))

    x = (index % 4) * 256
    y = (index // 4) * 280
    draw.text((x + 8, y + 5), classname, fill="black")
    canvas.paste(thumbnail, (x, y + 24))

output_path = project_root / "debug_can" / "ad2_categories.png"
output_path.parent.mkdir(exist_ok=True)
canvas.save(output_path)

print("\nAll category loading checks passed.")
print("Preview:", output_path)