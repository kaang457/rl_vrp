import torch

class CVRPEnv:
    def __init__(self, depot, locs, demands, capacity, device="cpu"):  # ortam kurar
        self.device = device
        self.depot = depot.to(device).detach()
        self.locs = locs.to(device).detach()
        self.demands = demands.to(device).float().detach()
        self.capacity = float(capacity)

        B, N, _ = self.locs.shape
        self.B, self.N = B, N
        self.depot_idx = N

        self.remaining = self.demands.clone().detach()
        self.remaining_load = torch.full((B,), self.capacity, device=device)
        self.current = self.depot.clone().detach()
        self.total_dist = torch.zeros((B,), device=device)

    def done(self):  # bitiş kontrol
        return (self.remaining <= 0).all(dim=1)

    def mask(self):  # uygun aksiyon maskesi
        B, N = self.B, self.N
        feas = torch.ones((B, N + 1), dtype=torch.bool, device=self.device)

        demand_pos = self.remaining > 0
        load_ok = self.remaining <= self.remaining_load[:, None]
        feas[:, :N] = demand_pos & load_ok
        return feas

    def step(self, action):  # adım atar
        B, N = self.B, self.N
        action = action.to(self.device)
        idx = torch.arange(B, device=self.device)
        is_depot = action == self.depot_idx

        next_xy = torch.where(
            is_depot[:, None],
            self.depot,
            self.locs[idx, action.clamp_max(N - 1)]
        )

        self.total_dist = self.total_dist + torch.norm(self.current - next_xy, dim=-1)
        self.current = next_xy.detach()

        new_load = torch.where(
            is_depot,
            torch.full_like(self.remaining_load, self.capacity),
            self.remaining_load
        )

        chosen = ~is_depot
        if chosen.any():
            a = action[chosen]
            bidx = idx[chosen]
            served = self.remaining[bidx, a]

            new_remaining = self.remaining.clone()
            new_remaining[bidx, a] = 0.0
            self.remaining = new_remaining.detach()

            new_load = new_load.clone()
            new_load[bidx] = new_load[bidx] - served

        self.remaining_load = new_load.detach()
