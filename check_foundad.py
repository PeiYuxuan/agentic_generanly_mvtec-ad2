from pathlib import Path
import sys

# Enable Python to locate the src module within FoundAD
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "foundad"))

import torch
from src.foundad import VisionModule

# check the code connection with the CPU and temporarily avoid using the server GPU
model = VisionModule(
    model_name="dinov3",
    pred_depth=6,
    pred_emb_dim=384,
    use_cuda=False,
    if_pe=False,
    feat_normed=False,
)
model.eval()

# Create a temporary test image
image = torch.zeros(1, 3, 512, 512)

with torch.no_grad():
    features = model.target_features(image, paths=[], n_layer=3)
    prediction = model.predict(features)

print("feature shape: ", tuple(features.shape))
print("Shape of predicted results: ", tuple(prediction.shape))

assert features.shape == prediction.shape
assert torch.isfinite(features).all().item()
assert torch.isfinite(prediction).all().item()

print("FoundAD forward running check passed")
