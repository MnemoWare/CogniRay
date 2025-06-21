import os
import sys
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.minimal_hpm import MinimalHPM

torch.manual_seed(1337)

# Copyright 2025 (c) Konstantin Bashinskiy
# Licensed under the CogniRay Non-Commercial License v1.0
#
# Experiment 06:
#   The details and interpretation of the experiment are described in: /docs/Experiment 06.md
#
# Result - Stage A:
#   Stage A: step 00 | MSE: 0.83475214
#   ...
#   Stage A: step 49 | MSE: 0.00000026
#
# Result - Stage B:
#   Stage B: step 00 | MSE: 0.41998407 | MSE A/B: 0.00000019/0.83996797
#   ...
#   Stage B: step 20 | MSE: 0.00331817 | MSE A/B: 0.00477572/0.00186062
#   Stage B: step 21 | MSE: 0.03431472 | MSE A/B: 0.00524396/0.06338547 <- Write mode change
#   ...
#   Stage B: step 57 | MSE: 0.01658897 | MSE A/B: 0.00000019/0.03317774 <- Target A restoration peak
#   ...
#   Stage B: step 69 | MSE: 0.00899105 | MSE A/B: 0.00435574/0.01362636
#

# Params
channels = 16
side = 64

# Stage A params
stage_A_alpha = 0.005
stage_A_steps = 50

# Stage B params
stage_B_alpha_delta = 0.005
stage_B_alpha_associative = 0.0001
stage_B_steps_delta = 20
stage_B_steps_associative = 50

# Global params
tau = 4.0
sigma = 1.5
eps = 1.0e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_memory_state(
    memory: torch.Tensor,
    target_A: torch.Tensor,
    target_B: torch.Tensor,
) -> None:
    torch.save(
        obj=dict(
            memory = memory,
            target_A = target_A,
            target_B = target_B,
        ),
        f=f"{script_dir}/data/ex06/memory_dynamics.datarec.pt",
    )

def log_loss(
    loss_A: torch.Tensor,
    loss_B: torch.Tensor,
) -> None:
    torch.save(
        obj=dict(
            loss_A = loss_A,
            loss_B = loss_B,
        ),
        f=f"{script_dir}/data/ex06/loss_dynamics.datarec.pt",
    )

# Init mem
hpm = MinimalHPM(
    shape=[side, side, side],
    channels=channels,
    tau=tau,
    sigma=sigma,
    init=False,
).to(device)

# Fixed random targets
ray_A_origin = torch.tensor([side//2, side//2+side//16, side//2], device=device, dtype=torch.float)
ray_B_origin = torch.tensor([side//2, side//2-side//16, side//2], device=device, dtype=torch.float)
ray_A_direction = F.normalize(torch.tensor([+1.0, +1.0, +0.5], device=device) + eps, dim=0)
ray_B_direction = F.normalize(torch.tensor([+1.0, -1.0, +0.5], device=device) + eps, dim=0)
ray_A_target = torch.randn([channels], device=device)
ray_B_target = torch.randn([channels], device=device)

# Shared loss accumulators
loss_A_accum = []
loss_B_accum = []

# Stage A: 
print("Begin Stage A: writing target A.")
for step in range(stage_A_steps):
    with torch.no_grad():
        origins = ray_A_origin.unsqueeze(0)
        directions = ray_A_direction.unsqueeze(0)
        targets = ray_A_target.unsqueeze(0)

        projections = hpm.read(origins, directions)
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Stage A: step {step:02d} | MSE: {loss.item():.8f}")

        delta = targets - projections
        hpm.write_delta(origins, directions, delta, stage_A_alpha)

        loss_A_accum.append(loss)
        loss_B_accum.append(torch.zeros_like(loss))

# Stage B:
a = F.normalize(ray_A_target, dim=0)
Q, _ = torch.linalg.qr(torch.eye(channels).to(device=device) - a.unsqueeze(1) @ a.unsqueeze(0))
ray_B_target = Q[:, 1] * ray_A_target.norm()

memory_dynamics_buffer = []
memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu()) # Last state

print("Begin Stage B: writing orthogonal target B. Target A left intact.")
for step in range(stage_B_steps_delta + stage_B_steps_associative):
    with torch.no_grad():
        origins = torch.stack([ray_A_origin, ray_B_origin], dim=0)
        directions = torch.stack([ray_A_direction, ray_B_direction], dim=0)
        targets = torch.stack([ray_A_target, ray_B_target], dim=0)

        projections = hpm.read(origins, directions)
        loss_A = F.mse_loss(projections[0], targets[0].to(device))
        loss_B = F.mse_loss(projections[1], targets[1].to(device))
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Stage B: step {step:02d} | MSE: {loss.item():.8f} | MSE A/B: {loss_A.item():.8f}/{loss_B.item():.8f}")

        delta = targets - projections
        if step < stage_B_steps_delta:
            hpm.write_delta(origins[1].unsqueeze(0), directions[1].unsqueeze(0), delta[1].unsqueeze(0), stage_B_alpha_delta)
        else:
            hpm.write_associative(origins[1].unsqueeze(0), directions[1].unsqueeze(0), delta[1].unsqueeze(0), stage_B_alpha_associative)

    memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu())
    loss_A_accum.append(loss_A)
    loss_B_accum.append(loss_B)

memory_dynamics_buffer = torch.stack(memory_dynamics_buffer, dim=0)
save_memory_state(memory_dynamics_buffer, ray_A_target.data, ray_B_target.data)
log_loss(torch.stack(loss_A_accum, dim=0), torch.stack(loss_B_accum, dim=0))
