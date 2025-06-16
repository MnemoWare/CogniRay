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
# Step 00 | MSE: 0.908733
# ...
# Step 49 | MSE: 0.717166
# Slow, but stable convergence.

# Params
B = 4096 # One sample per possible discrete ray source
C = 16
side = 16
steps = 50

tau = 8.0
sigma = 1.5
alpha = 0.0001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Init mem
hpm = MinimalHPM(
    shape=[side, side, side],
    channels=C,
    tau=tau,
    sigma=sigma,
    init=True,
    init_mean=0.0,
    init_std=0.1,
).to(device)

def fft_ramp_filter(
    signal: torch.Tensor,
) -> torch.Tensor:
    N = signal.shape[-1]
    f = torch.fft.rfft(signal, dim=-1)
    ramp = torch.fft.rfftfreq(N, d=1.0, device=signal.device).abs()
    return torch.fft.irfft(f * ramp, n=N, dim=-1)

def backproject_cone(
    energy: torch.Tensor,
    shape: tuple[int, int, int],
    tau: float,
) -> torch.Tensor:
    Dx, Dy, Dz = shape
    dev = energy.device
    L = torch.zeros(shape, device=dev)

    wx = torch.exp(-torch.arange(Dx, device=dev) / tau)
    wy = torch.exp(-torch.arange(Dy, device=dev) / tau)
    wz = torch.exp(-torch.arange(Dz, device=dev) / tau)

    L += wx[:, None, None] * energy[0].view(Dy, Dz)
    L += wx.flip(0)[:, None, None] * energy[1].view(Dy, Dz)

    L += wy[None, :, None] * energy[2].view(Dx, Dz)
    L += wy.flip(0)[None, :, None] * energy[3].view(Dx, Dz)

    L += wz[None, None, :] * energy[4].view(Dx, Dy)[..., None]
    L += wz.flip(0)[None, None, :] * energy[5].view(Dx, Dy)[..., None]

    return L

# Example projection recovery
def recover_optimal_projection(
    scan: torch.Tensor,
    target: torch.Tensor,
    shape: tuple[int, int, int],
    tau: float,
) -> torch.Tensor:
    device = scan.device
    Dx, Dy, Dz = shape

    p = target / target.norm()
    proj = (scan * p).sum(-1)
    proj_f = fft_ramp_filter(proj)
    L = backproject_cone(proj_f, shape, tau)

    idx_max = torch.argmax(L)
    z0 = idx_max % Dz
    y0 = (idx_max // Dz) % Dy
    x0 = idx_max // (Dy * Dz)
    x0 = torch.tensor([x0, y0, z0], dtype=torch.float32, device=device)

    gx = F.pad(L[2:, :, :] - L[:-2, :, :], (0, 0, 0, 0, 1, 1)) / 2
    gy = F.pad(L[:, 2:, :] - L[:, :-2, :], (0, 0, 1, 1, 0, 0)) / 2
    gz = F.pad(L[:, :, 2:] - L[:, :, :-2], (1, 1, 0, 0, 0, 0)) / 2
    g = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    S = (g.T @ g)
    S = 0.5 * (S + S.T)

    eigvals, eigvecs = torch.linalg.eigh(S)
    v0 = eigvecs[:, 0]
    v0 = v0 / v0.norm()

    center = torch.tensor([Dx, Dy, Dz], device=device, dtype=torch.float32) / 2 - 0.5
    if ((center - x0) @ v0) < 0:
        v0 = -v0

    return x0, v0

# Targets
ray_origins = torch.rand(B, 3) * float(side)
ray_dirs = torch.randn(B, 3)
ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
targets = torch.randn(B, C)
targets = targets.to(device)

# Writing cycle
for step in range(steps):
    with torch.no_grad():
        scan_result, scan_origins, scan_dirs = hpm.scan()

        new_ray_origins, new_ray_dirs = [], []
        for p in targets:
            x_star, v_star = recover_optimal_projection(
                scan=scan_result.clone(),
                target=p,
                shape=hpm.memory.shape[0:-1],
                tau=tau,
            )
            new_ray_origins.append(x_star.unsqueeze(0))
            new_ray_dirs.append(v_star.unsqueeze(0))

        new_ray_origins = torch.cat(new_ray_origins, dim=0)
        new_ray_dirs = torch.cat(new_ray_dirs, dim=0)

        projections = hpm.read(new_ray_origins.to(device), new_ray_dirs.to(device))
        loss = F.mse_loss(projections, targets.to(device))
        print(f"Step {step:02d} | MSE: {loss.item():.6f}")

        delta = targets.to(device) - projections
        hpm.write(new_ray_origins.to(device), new_ray_dirs.to(device), delta, alpha)
