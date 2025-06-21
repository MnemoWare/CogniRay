import os
import sys
import torch
import torch.nn.functional as F

from typing import Union

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
#   Stage A: step 00 | MSE: 0.84024251
#   ...
#   Stage A: step 24 | MSE: 0.00002203
#
# Result - Stage B:
#   Stage B: step 00 | MSE: 0.93652594 | MSE A/other: 0.00001419/1.17065382
#   ...
#   Stage B: step 49 | MSE: 0.61300713 | MSE A/other: 0.00023416/0.76620036
#

# Params
channels = 16
side = 64

# Stage A params
stage_A_alpha = 0.005
stage_A_steps = 25

# Stage B params
stage_B_alpha_associative = 0.001
stage_B_steps_associative = 220

# Global params
tau = 8.0
sigma = 1.25
eps = 1.0e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_memory_state(
    memory: torch.Tensor,
    target_A: torch.Tensor,
    target_other: Union[torch.Tensor, None],
    prefix: str,
) -> None:
    torch.save(
        obj=dict(
            memory = memory,
            target_A = target_A,
            target_other = target_other,
        ),
        f=f"{script_dir}/data/ex05/05_{prefix + '_' if prefix is not None else ''}memory_state.datarec.pt",
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
memory_dynamics_buffer = []

print("Begin Stage A: imprinting target A.")
for step in range(stage_A_steps):
    with torch.no_grad():
        origins = torch.stack([ray_A_origin], dim=0)
        directions = torch.stack([ray_A_direction], dim=0)
        targets = torch.stack([ray_A_target], dim=0)

        projections = hpm.read(origins, directions)
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Stage A: step {step:02d} | MSE: {loss.item():.8f}")

        delta = targets - projections
        hpm.write_delta(origins, directions, delta, stage_A_alpha)

        memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu())

memory_dynamics_buffer = torch.stack(memory_dynamics_buffer, dim=0)
save_memory_state(memory_dynamics_buffer, ray_A_target.data, None, "stage_a_dynamics")

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

memory_dynamics_buffer = []
memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu()) # Last state

print("Begin Stage B: project four random targets into overlapping region. Do not update A.")
for step in range(stage_B_steps_associative):
    with torch.no_grad():
        origins = torch.cat([ray_A_origin.unsqueeze(0), ray_other_origin], dim=0)
        directions = torch.cat([ray_A_direction.unsqueeze(0), ray_other_direction], dim=0)
        targets = torch.cat([ray_A_target.unsqueeze(0), ray_other_target], dim=0)

        projections = hpm.read(origins, directions)
        loss_A = F.mse_loss(projections[0], targets[0].to(device))
        loss_other = F.mse_loss(projections[1:], targets[1:].to(device))
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Stage B: step {step:02d} | MSE: {loss.item():.8f} | MSE A/other: {loss_A.item():.8f}/{loss_other.item():.8f}")

        delta = targets - projections
        hpm.write_associative(origins[1:], directions[1:], delta[1:], stage_B_alpha_associative)

        memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu())

memory_dynamics_buffer = torch.stack(memory_dynamics_buffer, dim=0)
save_memory_state(memory_dynamics_buffer, ray_A_target.data, ray_other_target.data, "stage_b_dynamics")
