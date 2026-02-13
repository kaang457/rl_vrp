import torch
import torch.optim as optim
import numpy as np

from data import generate_cvrp_batch
from env import CVRPEnv
from model import NazariStylePolicy


def rollout(policy, env):  # full episode
    logps = []

    while not env.done().all():
        mask = env.mask()

        probs = policy(
            env.depot.detach(),
            env.locs.detach(),
            env.remaining.detach(),
            env.remaining_load.detach(),
            env.current.detach(),
            mask
        )

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logps.append(dist.log_prob(action))

        env.step(action)  # env graph dışında

    logp = torch.stack(logps).sum(dim=0)
    reward = -env.total_dist.detach()
    return reward, logp


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("training started\n")

    policy = NazariStylePolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    batch_size = 128
    n_customers = 20
    capacity = 30
    aggressiveness = 0.3
    steps = 1000

    baseline = None
    beta = 0.9

    for step in range(steps):
        batch = generate_cvrp_batch(batch_size, n_customers, capacity, aggressiveness)

        depot = torch.tensor(np.stack([b["depot"] for b in batch]), device=device).float()
        locs  = torch.tensor(np.stack([b["locs"] for b in batch]), device=device).float()
        dem   = torch.tensor(np.stack([b["demands"] for b in batch]), device=device).float()

        env = CVRPEnv(depot, locs, dem, capacity, device)

        reward, logp = rollout(policy, env)

        with torch.no_grad():
            r_mean = reward.mean()
            baseline = r_mean if baseline is None else beta * baseline + (1 - beta) * r_mean

        advantage = reward - baseline
        loss = -(advantage * logp).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            print(
                f"step {step:4d} | "
                f"loss {loss.item():.4f} | "
                f"reward {reward.mean().item():.2f} | "
                f"baseline {baseline.item():.2f}"
            )


if __name__ == "__main__":
    train()
