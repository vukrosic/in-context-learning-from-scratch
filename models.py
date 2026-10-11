"""A small transformer that reads a prompt of receipts and answers the question in the last row."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shop import prompts, RECEIPTS, SCALE

D = 64        # every row becomes a list of D numbers
HEADS = 4     # each layer looks four times, with four sets of weights
LAYERS = 4


class Block(nn.Module):
    """One layer: every row looks at every row, then each row works on what it gathered."""

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))
        self.cut = False        # True: the question row may look only at itself

    def forward(self, h):
        b, t, _ = h.shape
        q, k, v = self.qkv(self.n1(h)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        score = q @ k.transpose(-1, -2) / math.sqrt(D // HEADS)
        if self.cut:
            score[:, :, -1, :-1] = float('-inf')
        self.looks = F.softmax(score, -1)                                   # shares of looking: (prompt, head, from row, to row)
        h = h + self.proj((self.looks @ v).transpose(1, 2).reshape(b, t, D))
        return h + self.mlp(self.n2(h))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(5, D)                                          # the five numbers of a row -> D numbers
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm, self.out = nn.LayerNorm(D), nn.Linear(D, 1)

    def forward(self, x):
        h = self.inp(x)
        self.inside = [h[:, -1]]                                            # the question row after each layer
        for blk in self.blocks:
            h = blk(h)
            self.inside.append(h[:, -1])
        return self.out(self.norm(h[:, -1])).squeeze(-1) * SCALE            # the answer, in coins


def count(net):
    return sum(p.numel() for p in net.parameters())


def train(net, shops, steps, seed=0, batch=128, lr=1e-3):
    """Every step: 128 fresh prompts, each from a shop picked at random, with 0 to 8 receipts."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=0.)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.05)
    for _ in range(steps):
        k = torch.randint(0, RECEIPTS + 1, ()).item()
        x, y, _ = prompts(shops, batch, k)
        loss = F.mse_loss(net(x) / SCALE, y / SCALE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def off(net, shops, k, n=2000, seed=1):
    """How far off the answers are with k receipts: coins on average, and percent that round to the exact total."""
    x, y, _ = prompts(shops, n, k, seed)
    with torch.no_grad():
        e = (net(x) - y).abs()
    return {'coins': e.mean().item(), 'exact': (e < .5).float().mean().item() * 100}


def fingerprint(net):
    """One number made from every weight. If a single weight changes, this changes."""
    return sum((p.double() * torch.arange(1, p.numel() + 1, dtype=torch.double).view_as(p)).sum().item() for p in net.parameters())
