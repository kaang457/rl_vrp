import torch

class CVRPEnv:
    def __init__(self, depot, locs, demands, capacity, device="cpu"):  # ortam kurar
        self.device = device
        self.depot = depot.to(device)
        self.locs = locs.to(device)
        self.demands = demands.to(device).float()
        self.capacity = float(capacity)

        B, N, _ = self.locs.shape
        self.B, self.N = B, N
        self.depot_idx = N

        self.remaining = self.demands.clone()
        self.remaining_load = torch.full((B,), self.capacity, device=device)
        self.current = self.depot.clone()
        self.total_dist = torch.zeros((B,), device=device)

    def done(self): 
        return (self.remaining <= 0).all(dim=1)

    def mask(self): 
        B, N = self.B, self.N
        feas = torch.ones((B, N + 1), dtype=torch.bool, device=self.device)

        demand_pos = self.remaining > 0
        load_ok = self.remaining <= self.remaining_load[:, None]
        feas[:, :N] = demand_pos & load_ok
        return feas

    def step(self, action): 
        B, N = self.B, self.N
        action = action.to(self.device)
        is_depot = action == self.depot_idx

        next_xy = torch.where(
            is_depot[:, None],
            self.depot,
            self.locs[torch.arange(B), action.clamp_max(N-1)]
        )

        self.total_dist += torch.norm(self.current - next_xy, dim=-1)
        self.current = next_xy

        self.remaining_load = torch.where(
            is_depot,
            torch.full_like(self.remaining_load, self.capacity),
            self.remaining_load
        )

        idx = torch.arange(B)
        chosen_customer = ~is_depot
        if chosen_customer.any():
            a = action[chosen_customer]
            bidx = idx[chosen_customer]
            served = self.remaining[bidx, a]
            self.remaining[bidx, a] = 0.0
            self.remaining_load[bidx] -= served
