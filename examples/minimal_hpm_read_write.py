import os
import sys
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.minimal_hpm import MinimalHPM

torch.manual_seed(1337)

# Result:
# Step 00 | MSE: 0.989227
# ...
# Step 49 | MSE: 0.067038

# Params
B = 2048
C = 16
side = 64
steps = 50

tau = 8.0
sigma = 0.25
alpha = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Init mem
hpm = MinimalHPM(
    shape=[side, side, side],
    channels=C,
    tau=tau,
    sigma=sigma,
).to(device)

# Fixed random targets
ray_origins = torch.rand(B, 3) * float(side)
ray_dirs = torch.randn(B, 3)
ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
targets = torch.randn(B, C)

# Writing cycle
for step in range(steps):
    with torch.no_grad():
        projections = hpm.read(ray_origins.to(device), ray_dirs.to(device))
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Step {step:02d} | MSE: {loss.item():.6f}")

        delta = targets.to(device) - projections
        hpm.write(ray_origins.to(device), ray_dirs.to(device), delta, alpha)
