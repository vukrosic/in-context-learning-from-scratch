"""Train the networks of the film, several at once, and keep them in nets/. Delete that folder to train again."""
import os
from multiprocessing import Pool
from pathlib import Path

import torch

from models import Net, train
from shop import KNOWN

SEEDS = [0, 1, 2]
SHOPS = [4, 16, 64, 600]      # how many different shops a network trains on
STEPS = 30000
NETS = Path(__file__).parent / 'nets'


def some(n, seed):
    """n of the 600 training shops, picked by the seed."""
    pick = torch.randperm(len(KNOWN), generator=torch.Generator().manual_seed(100 + seed))[:n]
    return KNOWN[pick]


def path(n, seed):
    return NETS / f'shops{n}_start{seed}.pt'


def one(job):
    torch.set_num_threads(1)
    n, seed = job
    torch.manual_seed(seed)
    net = train(Net(), some(n, seed), STEPS, seed)
    torch.save(net.state_dict(), path(n, seed))


def ready(sizes=SHOPS):
    """Train what is missing."""
    NETS.mkdir(exist_ok=True)
    todo = [(n, s) for n in sizes for s in SEEDS if not path(n, s).exists()]
    if todo:
        print(f'training {len(todo)} networks, {STEPS:,} steps each ...', flush=True)
        with Pool(int(os.environ.get('WORKERS', 8))) as pool:
            pool.map(one, todo, chunksize=1)


def load(n=600, seed=0):
    ready([n])
    net = Net()
    net.load_state_dict(torch.load(path(n, seed)))
    return net.eval()


def mean(values):
    return sum(values) / len(values)
