import os
import sys
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.minimal_hpm import MinimalHPM

torch.manual_seed(1337)

# Copyright 2025 (c) Konstantin Bashinsky
# Licensed under the CogniRay Non-Commercial License v1.0
#
# Experiment 05:
#   The details and interpretation of the experiment are described in: /docs/Experiment 05.md
#
# Result - Stage A:
#   Stage A: step 000 | MSE: 0.84024251
#   ...
#   Stage A: step 016 | MSE: 0.00074152
#   Stage A: target loss 0.001 reached on step 016
#
# Result - Stage B:
#   Stage B: step 000 | MSE: 0.93666965 | MSE A/other: 0.00047778/1.17071760
#   ...
#   Stage B: step 135 | MSE: 0.00206600 | MSE A/other: 0.00642920/0.00097520
#   Stage B: target loss 0.001 reached on step 135
#

# Params
channels = 16
side = 64

# Stage A params
stage_A_alpha = 0.005
stage_A_target_loss = 1.0e-3

# Stage B params
stage_B_alpha = 0.001
stage_B_target_loss = 1.0e-3

# Global params
tau = 8.0
sigma = 1.25
eps = 1.0e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def log_loss(
    loss_A: torch.Tensor,
    loss_other: torch.Tensor,
    prefix: str,
) -> None:
    torch.save(
        obj=dict(
            loss_A = loss_A,
            loss_other = loss_other,
        ),
        f=f"{script_dir}/data/ex05/{prefix + '_' if prefix is not None else ''}loss_dynamics.datarec.pt",
    )

# Init mem
hpm = MinimalHPM(
    shape=[side, side, side],
    channels=channels,
    tau=tau,
    sigma=sigma,
    init=True,
    init_mean=0.0,
    init_std=1.0e-3,
).to(device)

# Fixed random targets
ray_A_origin = torch.tensor([side//2, side//2, side//2], device=device, dtype=torch.float)
ray_A_direction = F.normalize(torch.tensor([+1.0, +0.0, +0.0], device=device) + eps, dim=0)
ray_A_target = torch.randn([channels], device=device)

# Stage A: 
print("Begin Stage A: imprinting target A.")
step = 0
target_loss_reached = False
loss_A_accum = []
while target_loss_reached is not True:
    with torch.no_grad():
        origins = torch.stack([ray_A_origin], dim=0)
        directions = torch.stack([ray_A_direction], dim=0)
        targets = torch.stack([ray_A_target], dim=0)

        projections = hpm.read(origins, directions)
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Stage A: step {step:03d} | MSE: {loss.item():.8f}")

        delta = targets - projections
        hpm.write_delta(origins, directions, delta, stage_A_alpha)

        if loss.item() <= stage_A_target_loss:
            target_loss_reached = True
            print(f"Stage A: target loss {stage_A_target_loss} reached on step {step:03d}")
        
        loss_A_accum.append(loss)

    step += 1

log_loss(torch.stack(loss_A_accum, dim=0), None, "delta_stage_a")

# Stage B:
ray_other_origin = torch.tensor(
    [
        [side//2, side//2-side//8, side//2],
        [side//2, side//2+side//8, side//2],
        [side//2, side//2, side//2-side//8],
        [side//2, side//2, side//2+side//8],
    ], device=device, dtype=torch.float)
ray_other_direction = F.normalize(torch.tensor(
    [
        [+0.0, +1.0, +0.0],
        [+0.0, -1.0, +0.0],
        [+0.0, +0.0, +1.0],
        [+0.0, +0.0, -1.0],
    ], device=device) + eps, dim=0)
ray_other_target = torch.randn([ray_other_origin.shape[0], channels], device=device)

print("Begin Stage B: project four random targets into overlapping region. Do not update A.")
step = 0
target_loss_reached = False
loss_A_accum = []
loss_B_accum = []
while target_loss_reached is not True:
    with torch.no_grad():
        origins = torch.cat([ray_A_origin.unsqueeze(0), ray_other_origin], dim=0)
        directions = torch.cat([ray_A_direction.unsqueeze(0), ray_other_direction], dim=0)
        targets = torch.cat([ray_A_target.unsqueeze(0), ray_other_target], dim=0)

        projections = hpm.read(origins, directions)
        loss_A = F.mse_loss(projections[0], targets[0].to(device))
        loss_other = F.mse_loss(projections[1:], targets[1:].to(device))
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Stage B: step {step:03d} | MSE: {loss.item():.8f} | MSE A/other: {loss_A.item():.8f}/{loss_other.item():.8f}")

        delta = targets - projections
        hpm.write_delta(origins[1:], directions[1:], delta[1:], stage_B_alpha)

        if loss_other.item() <= stage_B_target_loss:
            target_loss_reached = True
            print(f"Stage B: target loss {stage_B_target_loss} reached on step {step:03d}")
        
        loss_A_accum.append(loss_A)
        loss_B_accum.append(loss_other)

    step += 1

log_loss(torch.stack(loss_A_accum, dim=0), torch.stack(loss_B_accum, dim=0), "delta_stage_b")
