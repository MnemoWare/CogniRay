import torch
import torch.nn.functional as F

class MinimalHPM(torch.nn.Module):
    def __init__(
        self, 
        shape: list[int] = [128, 128, 128],
        channels: int = 16,
        tau: float = 2.0,
        sigma: float = 0.5,
        init: bool = False,
        init_mean: float = 0.0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.memory = torch.nn.Parameter(torch.zeros(*shape, channels))
        self.sigma = sigma
        self.tau = tau

        grid = torch.stack(torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing='ij'
        ), dim=-1).float()
        self.register_buffer("grid", grid)

        if init:
            with torch.no_grad():
                self.memory.normal_(mean=init_mean, std=init_std)
        
        pass

    def kernel(
        self,
        x: torch.Tensor,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
    ) -> torch.Tensor:
        dx = x - ray_origin
        t = (dx * ray_dir).sum(-1)
        x_proj = ray_origin + t[..., None] * ray_dir
        r2 = ((x - x_proj) ** 2).sum(-1)
        kernel = torch.exp(-r2 / (2 * self.sigma ** 2)) * torch.exp(-t.abs() / self.tau)
        return kernel

    def read(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
    ) -> torch.Tensor:
        B = ray_origin.shape[0]
        flat_grid = self.grid.view(-1, 3)
        mem = self.memory.view(-1, self.memory.shape[-1])
        out = []
        for b in range(B):
            k = self.kernel(flat_grid, ray_origin[b], ray_dir[b])
            t = (mem * k[:, None]).sum(0)
            out.append(t)
        return torch.stack(out, dim=0)

    def write_delta(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
        delta: torch.Tensor,
        alpha: float = 0.01,
    ) -> None:
        """
        Basic delta-projection update.

        Directly applies delta-weighted projections to the memory field without any filtering or suppression.
        Suitable for initial learning phases or unconditional memory injection.
        """
        
        B = ray_origin.shape[0]
        flat_grid = self.grid.view(-1, 3)
        mem = self.memory.view(-1, self.memory.shape[-1])
        for b in range(B):
            k = self.kernel(flat_grid, ray_origin[b], ray_dir[b])
            update = alpha * delta[b][None, :] * k[:, None]
            mem.data += update
        pass

    def write_force(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
        delta: torch.Tensor,
        alpha: float = 0.01,
    ) -> None:
        """
        Aggressive suppressive update.

        Combines update and memory field strength to amplify strong activations and suppress weak ones.
        Does not perform alignment checks. Useful for forced overwriting or aggressive consolidation.
        """
        
        B = ray_origin.shape[0]
        flat_grid = self.grid.view(-1, 3)
        mem = self.memory.view(-1, self.memory.shape[-1])
        for b in range(B):
            k = self.kernel(flat_grid, ray_origin[b], ray_dir[b])
            update = alpha * delta[b][None, :] * k[:, None]
            refresh = alpha * mem.data * k.max() + (k[:, None] * torch.sign(mem.data))
            mem.data += (update**2 + refresh**2).sqrt() * update
        pass

    def write_associative(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
        delta: torch.Tensor,
        alpha: float = 0.01,
    ) -> None:
        """
        Associative memory update.

        Enhances aligned updates and suppresses conflicting ones by evaluating projection similarity
        with existing memory content. Balances reinforcement and forgetting.
        Ideal for contextual refinement and stable integration.
        """
        
        B = ray_origin.shape[0]
        flat_grid = self.grid.view(-1, 3)
        mem = self.memory.view(-1, self.memory.shape[-1])

        for b in range(B):
            k = self.kernel(flat_grid, ray_origin[b], ray_dir[b])
            update = alpha * delta[b][None, :] * k[:, None]

            refresh = alpha * mem.data * k.max() + (k[:, None] * torch.sign(mem.data))

            refreshing_update = (update**2 + refresh**2).sqrt() * update

            similarity = F.cosine_similarity(refreshing_update, mem.data, dim=-1, eps=1e-6).unsqueeze(-1)
            alignment = (similarity + 1.0) / 2.0

            significance = mem.data.norm(dim=-1, keepdim=True)
            significance = significance / ((significance.max() + 1e-6) * alpha)
            trustedness = alignment * significance

            update_aligned = refreshing_update * trustedness

            mem.data += update_aligned
        pass

    def write_reflexive(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
        delta: torch.Tensor,
        alpha: float = 0.01,
    ) -> None:
        """
        Reflexive memory update with resistance.

        Updates memory only when the projection is aligned and coherent with existing content.
        Incorporates environmental resistance to protect established structures.
        Enables safe insertion of new knowledge without overwriting prior memories.
        """

        B = ray_origin.shape[0]
        flat_grid = self.grid.view(-1, 3)
        mem = self.memory.view(-1, self.memory.shape[-1])

        for b in range(B):
            k = self.kernel(flat_grid, ray_origin[b], ray_dir[b])
            update = alpha * delta[b][None, :] * k[:, None]

            refresh = alpha * mem.data * k.max() + (k[:, None] * torch.sign(mem.data))

            refreshing_update = (update**2 + refresh**2).sqrt() * update

            similarity = F.cosine_similarity(refreshing_update, mem.data, dim=-1, eps=1e-6).unsqueeze(-1)
            alignment = (similarity + 1.0) / 2.0

            significance = mem.data.norm(dim=-1, keepdim=True)
            significance = significance / ((significance.max() + 1e-6) * alpha)
            trustedness = alignment * significance

            update_aligned = refreshing_update * trustedness

            resistance = torch.norm(update_aligned - update, dim=-1, keepdim=True)
            resistance = (resistance / resistance.max())

            mem.data += update_aligned * resistance
        pass

    def scan(self):
        Dx, Dy, Dz = self.memory.shape[:3]
        device = self.memory.device
        faces = []

        # +x
        y, z = torch.meshgrid(torch.arange(Dy), torch.arange(Dz), indexing='ij')
        origin = torch.stack([torch.zeros_like(y), y, z], dim=-1).reshape(-1, 3)
        direction = torch.tensor([1.0, 0.0, 0.0], device=device).expand(origin.shape[0], 3)
        faces.append((origin, direction))

        # -x
        origin = torch.stack([(Dx - 1) * torch.ones_like(y), y, z], dim=-1).reshape(-1, 3)
        direction = torch.tensor([-1.0, 0.0, 0.0], device=device).expand(origin.shape[0], 3)
        faces.append((origin, direction))

        # +y
        x, z = torch.meshgrid(torch.arange(Dx), torch.arange(Dz), indexing='ij')
        origin = torch.stack([x, torch.zeros_like(x), z], dim=-1).reshape(-1, 3)
        direction = torch.tensor([0.0, 1.0, 0.0], device=device).expand(origin.shape[0], 3)
        faces.append((origin, direction))

        # -y
        origin = torch.stack([x, (Dy - 1) * torch.ones_like(x), z], dim=-1).reshape(-1, 3)
        direction = torch.tensor([0.0, -1.0, 0.0], device=device).expand(origin.shape[0], 3)
        faces.append((origin, direction))

        # +z
        x, y = torch.meshgrid(torch.arange(Dx), torch.arange(Dy), indexing='ij')
        origin = torch.stack([x, y, torch.zeros_like(x)], dim=-1).reshape(-1, 3)
        direction = torch.tensor([0.0, 0.0, 1.0], device=device).expand(origin.shape[0], 3)
        faces.append((origin, direction))

        # -z
        origin = torch.stack([x, y, (Dz - 1) * torch.ones_like(x)], dim=-1).reshape(-1, 3)
        direction = torch.tensor([0.0, 0.0, -1.0], device=device).expand(origin.shape[0], 3)
        faces.append((origin, direction))

        # Concatenate all rays
        all_origins = torch.cat([f[0].to(device).float() for f in faces], dim=0)
        all_dirs = torch.cat([f[1] for f in faces], dim=0)
        all_dirs = all_dirs / all_dirs.norm(dim=-1, keepdim=True)

        # Read batch
        result = self.read(all_origins, all_dirs)

        P, R, C, V = 6, result.shape[0] // 6, result.shape[-1], 3
        result = result.reshape([P, R, C])
        all_origins = all_origins.reshape([P, R, V])
        all_dirs = all_dirs.reshape([P, R, V])

        return result, all_origins, all_dirs
