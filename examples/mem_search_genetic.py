import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from dataclasses import dataclass

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.minimal_hpm import MinimalHPM

# torch.manual_seed(1337)

# Copyright 2025 (c) Konstantin Bashinskiy
# Licensed under the CogniRay Non-Commercial License v1.0

# Main logic params
write_before_search = True

# Mamory params
channels = 16
default_side = 64
init_mean = 0.0
init_std = 1.0e-2

# Random write params
random_write_alpha = 0.01
random_write_target_loss = 1.0e-5
random_write_loging_step = False
random_write_loging_target = True

# Global params
default_tau = 4.0
default_sigma = 1.0
default_eps = 1.0e-3
default_p_u_min = -1.0
default_p_u_max = +1.0
default_p_v_min = -1.0
default_p_v_max = +1.0
default_p_t_min = +0.5
default_p_t_max = +1.5
default_p_s_min = +0.5
default_p_s_max = +1.5

# Genetic search params
genetic_search_samples = 1024
genetic_search_top_k = 64
genetic_search_new_random_rays = 512
genetic_search_target_loss = 1.0e-3
genetic_search_max_epochs = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================================================== #
# Types.
# =========================================================================================== #
@dataclass
class RaysParametrized:
    p_u: torch.nn.Parameter
    p_v: torch.nn.Parameter
    p_t: torch.nn.Parameter
    p_s: torch.nn.Parameter

@dataclass
class RaysGeometric:
    u: torch.nn.Parameter
    v: torch.nn.Parameter
    t: torch.nn.Parameter
    s: torch.nn.Parameter

@dataclass
class RaysBest:
    rays: RaysParametrized
    scores: torch.Tensor

# =========================================================================================== #
# Helper functions.
# =========================================================================================== #
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

        step += 1
    
    return rays_u, rays_v, rays_t

def random_parametrized_rays(
    rays_count: int,
    device: torch.device,
) -> RaysParametrized:
    p_u = torch.nn.Parameter(torch.empty([rays_count, 3], device=device, dtype=torch.float).uniform_(default_p_u_min, default_p_u_max))
    p_v = torch.nn.Parameter(torch.empty([rays_count, 3], device=device, dtype=torch.float).uniform_(default_p_v_min, default_p_v_max))
    p_t = torch.nn.Parameter(torch.empty([rays_count, 1], device=device, dtype=torch.float).uniform_(default_p_t_min, default_p_t_max))
    p_s = torch.nn.Parameter(torch.empty([rays_count, 1], device=device, dtype=torch.float).uniform_(default_p_s_min, default_p_s_max))
    return RaysParametrized(p_u=p_u, p_v=p_v, p_t=p_t, p_s=p_s)

def parametrized_rays_to_geometry(
    p_u: torch.nn.Parameter,
    p_v: torch.nn.Parameter,
    p_t: torch.nn.Parameter,
    p_s: torch.nn.Parameter,
    side: int,
    tau: float,
    sigma: float,
    eps: float,
) -> RaysGeometric:
    u = (torch.tanh(p_u) * (side / 2)) + (side / 2)
    v = F.normalize(torch.tanh(p_v) + eps, dim=-1)
    t = (p_t * tau).clamp(eps)
    s = (p_s * sigma).clamp(eps)
    return RaysGeometric(u=u, v=v, t=t, s=s)

def eval_rays(
    memory: MinimalHPM,
    u: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    T = memory.read(u, v, t, s)
    result = F.mse_loss(T, target.unsqueeze(0).repeat([T.shape[0], 1]), reduction="none").mean(dim=-1)
    return result

def select_best_rays(
    eval_result: torch.Tensor,
    p_u: torch.nn.Parameter,
    p_v: torch.nn.Parameter,
    p_t: torch.nn.Parameter,
    p_s: torch.nn.Parameter,
    topk: int,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.Tensor]:
    ids = torch.topk(eval_result, k=topk, largest=False, dim=0).indices
    return RaysBest(
        rays=RaysParametrized(p_u=p_u[ids], p_v=p_v[ids], p_t=p_t[ids], p_s=p_s[ids]),
        scores=eval_result[ids],
    )

def random_pairs(
    id_limit: int,
    count: int,
) -> list[int]:
    assert id_limit > 1
    double_set = set([f"{i}_{i}" for i in range(count)])

    unique_pairs = []
    while len(unique_pairs) < count:
        a = list(range(0, id_limit))
        b = list(range(0, id_limit))
        random.shuffle(a)
        random.shuffle(b)
        c = set([f"{p[0]}_{p[1]}" for p in zip(a, b)])
        c = c - double_set
        unique_pairs = unique_pairs + list(c)

    result = [int(v) for pair in unique_pairs[:count] for v in pair.split("_")]
    result = [[result[i+0], result[i+1]] for i in range(0, len(result), 2)]

    return result

def mutate_rays(
    selection_result: RaysBest,
    count: int,
    device: torch.device,
) -> RaysParametrized:
    pairs = random_pairs(
        id_limit=len(selection_result.rays.p_u),
        count=count,
    )
    pairs = torch.tensor(pairs, device=device, dtype=torch.long).permute([1, 0])
    mutation_factor = torch.rand([count, 1], device=device, dtype=torch.float)
    mutation_factor = (mutation_factor * 3.0) - 1.0 # Push between -1.0...+2.0

    with torch.no_grad():
        u = torch.lerp(
            input=selection_result.rays.p_u[pairs[0]],
            end=selection_result.rays.p_u[pairs[1]],
            weight=mutation_factor,
        ).clamp(default_p_u_min, default_p_u_max)
        v = torch.lerp(
            input=selection_result.rays.p_v[pairs[0]],
            end=selection_result.rays.p_v[pairs[1]],
            weight=mutation_factor,
        ).clamp(default_p_v_min, default_p_v_max)
        t = torch.lerp(
            input=selection_result.rays.p_t[pairs[0]],
            end=selection_result.rays.p_t[pairs[1]],
            weight=mutation_factor,
        ).clamp(default_p_t_min, default_p_t_max)
        s = torch.lerp(
            input=selection_result.rays.p_s[pairs[0]],
            end=selection_result.rays.p_s[pairs[1]],
            weight=mutation_factor,
        ).clamp(default_p_s_min, default_p_s_max)

    p_u = torch.nn.Parameter(u.clone().detach().requires_grad_(True))
    p_v = torch.nn.Parameter(v.clone().detach().requires_grad_(True))
    p_t = torch.nn.Parameter(t.clone().detach().requires_grad_(True))
    p_s = torch.nn.Parameter(s.clone().detach().requires_grad_(True))

    return RaysParametrized(p_u=p_u, p_v=p_v, p_t=p_t, p_s=p_s)

def new_generation(
    selection_result: RaysBest,
    generation_size: int,
    new_random_rays: int,
    device: torch.device,
) -> RaysParametrized:
    old_species = RaysParametrized(
        p_u=selection_result.rays.p_u,
        p_v=selection_result.rays.p_v,
        p_t=selection_result.rays.p_t,
        p_s=selection_result.rays.p_s,
    )
    new_species = random_parametrized_rays(
        rays_count=new_random_rays,
        device=device,
    )
    hybrids = mutate_rays(
        selection_result=selection_result,
        count=generation_size - len(old_species.p_u) - len(new_species.p_u),
        device=device,
    )

    return RaysParametrized(
        p_u=torch.cat([old_species.p_u, new_species.p_u, hybrids.p_u], dim=0),
        p_v=torch.cat([old_species.p_v, new_species.p_v, hybrids.p_v], dim=0),
        p_t=torch.cat([old_species.p_t, new_species.p_t, hybrids.p_t], dim=0),
        p_s=torch.cat([old_species.p_s, new_species.p_s, hybrids.p_s], dim=0),
    )

# =========================================================================================== #
# Init.
# =========================================================================================== #
hpm = MinimalHPM(
    shape=[default_side, default_side, default_side],
    channels=channels,
    tau=default_tau,
    sigma=default_sigma,
    init=True,
    init_mean=init_mean,
    init_std=init_std,
).to(device)

if write_before_search:
    t_u, t_v, t_T = write_random_targets(
        targets_count=1,
        channels=channels,
        alpha=random_write_alpha,
        target_loss=random_write_target_loss,
        memory=hpm,
        memory_side=default_side,
        memory_init_mean=init_mean,
        memory_init_std=init_std,
        device=device,
        eps=default_eps,
        loging_step=random_write_loging_step,
        loging_target=random_write_loging_target,
    )
else:
    t_T = torch.randn([1, channels], device=device, dtype=torch.float)

# =========================================================================================== #
# Genetic search cycle.
# =========================================================================================== #
generation = random_parametrized_rays(genetic_search_samples, device)

target_loss_reached = False
epoch_id = 0
while target_loss_reached is not True:
    # =======================> Genetic search. START

    rays_g = parametrized_rays_to_geometry(
        p_u=generation.p_u,
        p_v=generation.p_v,
        p_t=generation.p_t,
        p_s=generation.p_s,
        side=default_side,
        tau=default_tau,
        sigma=default_sigma,
        eps=default_eps,
    )
    eval_result = eval_rays(
        memory=hpm,
        u=rays_g.u,
        v=rays_g.v,
        t=rays_g.t,
        s=rays_g.s,
        target=t_T[0],
    )

    loss_min = eval_result.min()
    loss_max = eval_result.max()
    loss_min_idx = eval_result.argmin(dim=0)
    loss_mean = eval_result.mean()

    print(f"Epoch {epoch_id:3d}: loss={loss_mean.item():.5f}, loss_min={loss_min.item():.5f}, loss_max={loss_max.item():.5f}, best_idx={loss_min_idx.item()}")

    if loss_min <= genetic_search_target_loss:
        target_loss_reached = True
        print(f"Target loss of {genetic_search_target_loss} reached through genetic search at the epoch {epoch_id}.")

    if epoch_id >= genetic_search_max_epochs:
        print(f"Maximum of {genetic_search_max_epochs} epochs reached.")
        break

    epoch_id = epoch_id + 1

    best_rays = select_best_rays(
        eval_result=eval_result,
        p_u=generation.p_u,
        p_v=generation.p_v,
        p_t=generation.p_t,
        p_s=generation.p_s,
        topk=genetic_search_top_k,
    )
    generation = new_generation(
        selection_result=best_rays,
        generation_size=genetic_search_samples,
        new_random_rays=genetic_search_new_random_rays,
        device=device,
    )

    # =======================> Genetic search. END

# =========================================================================================== #
# Print results.
# =========================================================================================== #
print_format = lambda x: [f"{em:.5f}" for em in x]
print(f"Match:  \
    origin={print_format(rays_g.u[loss_min_idx].tolist())}, \
    direction={print_format(rays_g.v[loss_min_idx].tolist())}, \
    tau={rays_g.t[loss_min_idx].item():.3f}, \
    sigma={rays_g.s[loss_min_idx].item():.3f} \
    loss={loss_min.item():.5f}")

if write_before_search:
    print(f"Source: \
    origin={print_format(t_u[0].tolist())}, \
    direction={print_format(t_v[0].tolist())}, \
    tau={default_tau:.3f}, \
    sigma={default_sigma:.3f}")
