import os
import sys
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.minimal_hpm import MinimalHPM

torch.manual_seed(1337)

#
# Experiment 04:
#   The details and interpretation of the experiment are described in: /docs/Experiment 04.md
#
# Result - Stage A:
#   Stage A: step 00 | MSE: 0.67086810 | MSE A/B: 0.83475214/0.50698406
#   ...
#   Stage A: step 49 | MSE: 0.00000009 | MSE A/B: 0.00000014/0.00000004
#
# Result - Stage B:
#   Stage B: step 00 | MSE: 0.57095760 | MSE A/B: 0.00000010/1.14191508
#   ...
#   Stage B: step 07 | MSE: 0.06826865 | MSE A/B: 0.00112979/0.13540751 <- A/B conflict peak
#   ...
#   Stage B: step 49 | MSE: 0.00000034 | MSE A/B: 0.00000017/0.00000052
#
# Result - Stage C:
#   Stage C: step 00 | MSE: 0.83437717 | MSE A/B: 0.00000013/1.66875422
#       ray_B_origin            : [32.0, 28.0, 32.0]
#       ray_B_direction         : [0.6671846508979797, -0.6658515930175781, 0.3339255452156067]
#   ...
#   Stage C: step 99 | MSE: 0.00000208 | MSE A/B: 0.00000166/0.00000251
#       ray_B_origin            : [38.418861389160156, 35.54133605957031, 35.252342224121094]
#       ray_B_direction         : [0.6644283533096313, -0.6696304678916931, 0.33185842633247375]
#

# Params
channels = 16
side = 64

# Stage A params
stage_A_alpha = 0.005
stage_A_steps = 50
stage_A_targets_delta = 0.25

# Stage B params
stage_B_alpha = 0.005
stage_B_steps = 50

# Stage C params
stage_C_alpha = 0.0025
stage_C_steps = 100
stage_C_SGD_lr = 0.01
stage_C_SGD_momentum = 0.1

# Global params
tau = 4.0
sigma = 1.5
eps = 1.0e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_memory_state(
    memory: torch.Tensor,
    target_A: torch.Tensor,
    target_B: torch.Tensor,
    prefix: str,
) -> None:
    torch.save(
        obj=dict(
            memory = memory,
            target_A = target_A,
            target_B = target_B,
        ),
        f=f"{script_dir}/data/ex04/{prefix + '_' if prefix is not None else ''}memory_state.datarec.pt",
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
ray_B_target = torch.lerp(ray_A_target, torch.randn([channels], device=device), stage_A_targets_delta)

# Stage A: 
memory_dynamics_buffer = []

print("Begin Stage A: coherent targets, fixed geometry.")
for step in range(stage_A_steps):
    with torch.no_grad():
        origins = torch.stack([ray_A_origin, ray_B_origin], dim=0)
        directions = torch.stack([ray_A_direction, ray_B_direction], dim=0)
        targets = torch.stack([ray_A_target, ray_B_target], dim=0)

        projections = hpm.read(origins, directions)
        loss_A = F.mse_loss(projections[0], targets[0].to(device))
        loss_B = F.mse_loss(projections[1], targets[1].to(device))
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Stage A: step {step:02d} | MSE: {loss.item():.8f} | MSE A/B: {loss_A.item():.8f}/{loss_B.item():.8f}")

        delta = targets - projections
        hpm.write(origins, directions, delta, stage_A_alpha)

        memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu())

memory_dynamics_buffer = torch.stack(memory_dynamics_buffer, dim=0)
save_memory_state(hpm.memory.data, ray_A_target.data, ray_B_target.data, "stage_a")
save_memory_state(memory_dynamics_buffer, ray_A_target.data, ray_B_target.data, "stage_a_dynamics")

# Stage B:
a = F.normalize(ray_A_target, dim=0)
Q, _ = torch.linalg.qr(torch.eye(channels).to(device=device) - a.unsqueeze(1) @ a.unsqueeze(0))
ray_B_target = Q[:, 1] * ray_A_target.norm()

memory_dynamics_buffer = []
memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu()) # Last state

print("Begin Stage B: orthogonal target for B, fixed geometry.")
for step in range(stage_B_steps):
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
        hpm.write(origins, directions, delta, stage_B_alpha)

        memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu())

memory_dynamics_buffer = torch.stack(memory_dynamics_buffer, dim=0)
save_memory_state(hpm.memory.data, ray_A_target.data, ray_B_target.data, "stage_b")
save_memory_state(memory_dynamics_buffer, ray_A_target.data, ray_B_target.data, "stage_b_dynamics")

# Stage C:
ray_B_target = Q[:, 2] * ray_A_target.norm()
ray_B_origin_local = torch.nn.Parameter((ray_B_origin / side) - 0.5)
ray_B_direction_local = torch.nn.Parameter(ray_B_direction.clone())

optimizer = torch.optim.SGD([ray_B_origin_local, ray_B_direction_local], lr=stage_C_SGD_lr, momentum=stage_C_SGD_momentum)

memory_dynamics_buffer = []
memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu()) # Last state

print("Begin Stage C: orthogonal target for B, adaptable origin and direction for B.")
for step in range(stage_C_steps):
    ray_B_origin_buffer = (ray_B_origin_local + 0.5) * side
    ray_B_direction_buffer = F.normalize(ray_B_direction_local, dim=0)

    origins = torch.stack([ray_A_origin, ray_B_origin_buffer], dim=0)
    directions = torch.stack([ray_A_direction, ray_B_direction_buffer], dim=0)
    targets = torch.stack([ray_A_target, ray_B_target], dim=0)

    projections = hpm.read(origins, directions)
    loss_A = F.mse_loss(projections[0], targets[0].to(device))
    loss_B = F.mse_loss(projections[1], targets[1].to(device))
    loss = F.mse_loss(projections, targets.to(device))
    print(f"Stage C: step {step:02d} | MSE: {loss.item():.8f} | MSE A/B: {loss_A.item():.8f}/{loss_B.item():.8f}")
    print(f"    ray_B_origin            : {ray_B_origin_buffer.data.tolist()}")
    print(f"    ray_B_direction         : {ray_B_direction_buffer.data.tolist()}")
    loss_B.backward()

    with torch.no_grad():
        hpm.write(origins, directions, targets - projections, stage_C_alpha)
    
    optimizer.step()
    optimizer.zero_grad()

    memory_dynamics_buffer.append(hpm.memory.data.clone().detach().cpu())

memory_dynamics_buffer = torch.stack(memory_dynamics_buffer, dim=0)
save_memory_state(hpm.memory.data, ray_A_target.data, ray_B_target.data, "stage_c")
save_memory_state(memory_dynamics_buffer, ray_A_target.data, ray_B_target.data, "stage_c_dynamics")
