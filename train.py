import torch
import torch.optim as optim
import numpy as np

from data import generate_cvrp_batch
from env import CVRPEnv
from model import NazariStylePolicy

def train(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = NazariStylePolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=1e-4)

    for step in range(1000):
        batch = generate_cvrp_batch(128, 20, 30, aggressiveness=0.3)

        depot = torch.tensor(np.stack([b["depot"] for b in batch]), device=device)
        locs = torch.tensor(np.stack([b["locs"] for b in batch]), device=device)
        dem  = torch.tensor(np.stack([b["demands"] for b in batch]), device=device)

        env = CVRPEnv(depot, locs, dem, 30, device)

        if step % 100 == 0:
            print("step:", step)

if __name__ == "__main__":
    train()
