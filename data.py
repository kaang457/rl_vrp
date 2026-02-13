import numpy as np

def generate_cvrp_instance(n_customers, capacity, aggressiveness=0.0, demand_low=1, demand_high=9, seed=None):  # veri üretir
    rng = np.random.default_rng(seed)

    depot = rng.uniform(0.0, 1.0, size=(2,)).astype(np.float32)

    if aggressiveness == 0.0:
        locs = rng.uniform(0.0, 1.0, size=(n_customers, 2))
    else:
        k = rng.integers(2, 6)
        hotspots = rng.uniform(0.0, 1.0, size=(k, 2))
        sigma = (1.0 - aggressiveness) * 0.25 + aggressiveness * 0.03
        assign = rng.integers(0, k, size=(n_customers,))
        locs = hotspots[assign] + rng.normal(0.0, sigma, size=(n_customers, 2))
        locs = np.clip(locs, 0.0, 1.0)

    locs = locs.astype(np.float32)
    demands = rng.integers(demand_low, demand_high + 1, size=(n_customers,))

    return {
        "depot": depot,
        "locs": locs,
        "demands": demands,
        "capacity": int(capacity),
    }


def generate_cvrp_batch(batch_size, n_customers, capacity, aggressiveness=0.0):  # batch üretir
    return [
        generate_cvrp_instance(n_customers, capacity, aggressiveness)
        for _ in range(batch_size)
    ]
