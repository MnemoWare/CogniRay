import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.minimal_hpm import MinimalHPM

# torch.manual_seed(1337)

# Copyright 2025 (c) Konstantin Bashinskiy
# Licensed under the CogniRay Non-Commercial License v1.0

# Logic params
write_before_search = True

# Params
channels = 16
side = 64

# Random write params
random_write_alpha = 0.01
random_write_target_loss = 1.0e-5
random_write_loging_step = False
random_write_loging_target = True

# Global params
init_mean = 0.0
init_std = 1.0e-2
default_tau = 4.0
default_sigma = 1.0
default_eps = 1.0e-3

# Search params
search_samples = 1024
search_target_loss = 1.0e-3
search_max_epochs = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Init mem
hpm = MinimalHPM(
    shape=[side, side, side],
    channels=channels,
    tau=default_tau,
    sigma=default_sigma,
    init=True,
    init_mean=init_mean,
    init_std=init_std,
).to(device)

# Random targets imprinting
def write_random_targets(
    targets_count: int,
    channels: int,
    alpha: float,
    target_loss: float,
    memory: MinimalHPM,
    memory_side: int,
    device: torch.device,
    memory_init_mean: float,
    memory_init_std: float,
    eps: float = 1.0e-3,
    loging_step: bool = False,
    loging_target: bool = False,
):
    with torch.no_grad():
        # Reinit memory to noise
        memory.memory.normal_(mean=memory_init_mean, std=memory_init_std)
        memory.memory.requires_grad_(True)

    # Fixed random target
    rays_u = torch.rand([targets_count, 3], device=device, dtype=torch.float) * memory_side
    rays_v = F.normalize(torch.randn([targets_count, 3], device=device, dtype=torch.float) + eps, dim=-1)
    rays_t = torch.randn([targets_count, channels], device=device, dtype=torch.float)

    # Writing cycle
    step = 0
    target_loss_reached = False
    while target_loss_reached is not True:
        with torch.no_grad():
            projections = memory.read(rays_u, rays_v)
            loss = F.mse_loss(projections, rays_t)

            if loging_step:
                print(f"Step {step:02d} | MSE: {loss.item():.6f}")

            if loss.item() <= target_loss:
                target_loss_reached = True

                if loging_target:
                    print(f"Writing {targets_count} random targets: target loss {target_loss} reached on step {step:03d}")
                break
            
            delta = rays_t - projections
            memory.write_delta(rays_u, rays_v, delta, alpha)
            # memory.write_reflexive(rays_u, rays_v, delta, alpha)

        step += 1
    
    return rays_u, rays_v, rays_t


param_rays_u        = torch.nn.Parameter(torch.empty([search_samples, 3], device=device, dtype=torch.float).uniform_(-1.0, +1.0))
param_rays_v        = torch.nn.Parameter(torch.empty([search_samples, 3], device=device, dtype=torch.float).uniform_(-1.0, +1.0))
param_rays_tau      = torch.nn.Parameter(torch.empty([search_samples, 1], device=device, dtype=torch.float).uniform_(0.5, 1.5))
param_rays_sigma    = torch.nn.Parameter(torch.empty([search_samples, 1], device=device, dtype=torch.float).uniform_(0.5, 1.5))

if write_before_search:
    target_u, target_v, target_value = write_random_targets(
        targets_count=1,
        channels=channels,
        alpha=random_write_alpha,
        target_loss=random_write_target_loss,
        memory=hpm,
        memory_side=side,
        memory_init_mean=init_mean,
        memory_init_std=init_std,
        device=device,
        eps=default_eps,
        loging_step=random_write_loging_step,
        loging_target=random_write_loging_target,
    )
else:
    target_value = torch.randn([1, channels], device=device, dtype=torch.float)

optim = torch.optim.Adam(
    params=[param_rays_u, param_rays_v, param_rays_tau, param_rays_sigma],
    lr=1.0e-2,
    weight_decay=1.0e-7,
    amsgrad=False,
)

target_loss_reached = False
epoch_id = 0
while target_loss_reached is not True:
    origins = (torch.tanh(param_rays_u) * (side / 2)) + (side / 2)
    directions = F.normalize(torch.tanh(param_rays_v) + default_eps, dim=-1)
    rays_tau = (param_rays_tau * default_tau).clamp(min=default_eps)
    rays_sigma = (param_rays_sigma * default_sigma).clamp(min=default_eps)

    projections = hpm.read(origins, directions, rays_tau, rays_sigma)
    loss = F.mse_loss(projections, target_value.repeat([projections.shape[0], 1]), reduction="none").mean(dim=-1)
    loss_min = loss.min()
    loss_max = loss.max()
    loss_min_idx = loss.argmin(dim=0)
    loss = loss.mean()
    loss.backward()
    optim.step()
    optim.zero_grad()

    print(f"Epoch {epoch_id:3d}: loss={loss.item():.5f}, loss_min={loss_min.item():.5f}, loss_max={loss_max.item():.5f}, best_idx={loss_min_idx.item()}")

    if loss_min <= search_target_loss:
        target_loss_reached = True
        print(f"Target loss reached through search at the epoch {epoch_id}.")

    if epoch_id >= search_max_epochs:
        print(f"Maximum of {search_max_epochs} epochs reached.")
        break

    epoch_id = epoch_id + 1

print_format = lambda x: [f"{em:.5f}" for em in x]
print(f"Match:  \
    origin={print_format(origins[loss_min_idx].tolist())}, \
    direction={print_format(directions[loss_min_idx].tolist())}, \
    tau={rays_tau[loss_min_idx].item():.3f}, \
    sigma={rays_sigma[loss_min_idx].item():.3f} \
    loss={loss_min.item():.5f}")

if write_before_search:
    print(f"Source: \
    origin={print_format(target_u[0].tolist())}, \
    direction={print_format(target_v[0].tolist())}, \
    tau={default_tau:.3f}, \
    sigma={default_sigma:.3f}")
