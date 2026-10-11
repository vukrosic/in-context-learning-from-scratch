"""Step 2. Ordinary learning. A model with three weights learns one shop, one receipt at a time. The prices end up in the weights."""
import torch

from shop import EXAMPLE, MOST

RATE = 0.01
OTHER = [8, 1, 5]         # a second shop


def learn(prices, weights, n, seed=0):
    """n receipts from one shop. After each: guess, compare with the total, move every weight a little against the mistake."""
    g = torch.Generator().manual_seed(seed)
    prices, w = torch.tensor(prices, dtype=torch.float), weights.clone()
    seen = {0: w.tolist()}
    for i in range(1, n + 1):
        basket = torch.randint(0, MOST + 1, (3,), generator=g).float()
        guess, total = (w * basket).sum(), (prices * basket).sum()
        w -= RATE * (guess - total) * basket
        if i in (1, 2, 3, 10, 30, 100, 300, n):
            seen[i] = w.tolist()
    return w, seen


def miss(w, prices, n=2000, seed=1):
    """Coins off on n baskets of a shop."""
    g = torch.Generator().manual_seed(seed)
    baskets = torch.randint(0, MOST + 1, (n, 3), generator=g).float()
    return ((baskets * w).sum(1) - (baskets * torch.tensor(prices, dtype=torch.float)).sum(1)).abs().mean().item()


def first(prices, seed=0):
    """The first receipt in full, for the film: basket, guess, total, and the weights after it."""
    g = torch.Generator().manual_seed(seed)
    basket = torch.randint(0, MOST + 1, (3,), generator=g).float()
    total = (torch.tensor(prices, dtype=torch.float) * basket).sum().item()
    return {'basket': basket.int().tolist(), 'guess': 0.0, 'total': total, 'after': (RATE * total * basket).tolist()}


def run():
    a, seen = learn(EXAMPLE['prices'], torch.zeros(3), 1000)
    b, _ = learn(OTHER, a, 1000, seed=2)
    return {'rate': RATE, 'first': first(EXAMPLE['prices']), 'seen': seen, 'weights': a.tolist(), 'other': OTHER,
            'miss_start': miss(torch.zeros(3), EXAMPLE['prices']), 'miss_own': miss(a, EXAMPLE['prices']), 'miss_other': miss(a, OTHER),
            'after_other': b.tolist(), 'miss_first_again': miss(b, EXAMPLE['prices'])}


if __name__ == '__main__':
    r = run()
    f = r['first']
    print(f"first receipt: basket {f['basket']}, guess {f['guess']}, total {f['total']:.0f} -> weights {[round(v, 2) for v in f['after']]}")
    for n, w in r['seen'].items():
        print(f'  after {n:4d} receipts the three weights are', [round(v, 2) for v in w])
    print(f"hidden prices of the shop: {EXAMPLE['prices']}")
    print(f"coins off on this shop: {r['miss_start']:.2f} before, {r['miss_own']:.2f} after")
    print(f"same weights on a second shop with prices {r['other']}: {r['miss_other']:.2f} coins off")
    print(f"after 1,000 receipts from the second shop the weights are {[round(v, 2) for v in r['after_other']]}")
    print(f"and the first shop is now {r['miss_first_again']:.2f} coins off: the weights hold one shop at a time")
