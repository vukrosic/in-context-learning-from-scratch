"""The task. A shop sells three fruits. The prices are hidden. A receipt shows how many of each fruit
were bought, and the total. From a few receipts, say what a new basket costs."""
import itertools

import torch

FRUITS = ['apples', 'pears', 'plums']
LOW, HIGH = 1, 9          # a fruit costs between 1 and 9 coins
MOST = 5                  # a basket holds 0 to 5 of each fruit
RECEIPTS = 8              # the longest prompt has 8 receipts
SCALE = MOST * HIGH       # totals are divided by 45 before the network sees them

ALL = torch.tensor(list(itertools.product(range(LOW, HIGH + 1), repeat=3)), dtype=torch.float)   # all 729 price lists
_order = torch.randperm(len(ALL), generator=torch.Generator().manual_seed(0))
NEW = ALL[_order[:129]]       # 129 price lists that training never uses
KNOWN = ALL[_order[129:]]     # 600 price lists for training


def receipts(shops, n, k, seed=None):
    """n prompts. Each one: a shop picked at random, k receipts from it, and one question basket.
    Returns counts (n, k+1, 3), totals (n, k+1) and the price list of each prompt. The last basket is the question."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    prices = shops[torch.randint(len(shops), (n,), generator=g)]
    counts = torch.randint(0, MOST + 1, (n, k + 1, 3), generator=g).float()
    totals = (counts * prices[:, None]).sum(-1)
    return counts, totals, prices


def rows(counts, totals):
    """Write a prompt as rows of five numbers: three counts, the total, and a mark that says 'this row is the question'.
    The question row has no total: that is what the network must say."""
    n, slots, _ = counts.shape
    x = torch.zeros(n, slots, 5)
    x[:, :, :3] = counts / MOST
    x[:, :-1, 3] = totals[:, :-1] / SCALE
    x[:, -1, 4] = 1
    return x


def prompts(shops, n, k, seed=None):
    """Prompts ready for the network, the right answers, and the hidden prices."""
    counts, totals, prices = receipts(shops, n, k, seed)
    return rows(counts, totals), totals[:, -1], prices


# One shop followed through the whole film. Its price list is one of the 129 that training never uses.
EXAMPLE = {'prices': [2, 7, 4], 'baskets': [[2, 1, 0], [0, 1, 3], [1, 2, 2]], 'question': [3, 1, 2]}


def example(k=3):
    """The film's shop as a prompt with its first k receipts. Returns the rows, the right answer, counts and totals."""
    prices = torch.tensor(EXAMPLE['prices'], dtype=torch.float)
    counts = torch.tensor(EXAMPLE['baskets'][:k] + [EXAMPLE['question']], dtype=torch.float)[None]
    totals = (counts * prices).sum(-1)
    return rows(counts, totals), totals[0, -1].item(), counts, totals


def fitting(counts, totals):
    """For each prompt, which of the 729 price lists give the totals printed on its receipts. Returns (n, 729) of 0 and 1."""
    would = (counts[:, :-1, None, :] * ALL).sum(-1)                         # the total each list would print, per receipt
    return (would == totals[:, :-1, None]).all(1).float()


def ideal(counts, totals):
    """The best possible guesser: price the question with every list that still fits, and take the average."""
    ok = fitting(counts, totals)
    cost = (counts[:, -1, None, :] * ALL).sum(-1)                           # the question under each list
    return (cost * ok).sum(1) / ok.sum(1), ok.sum(1)
