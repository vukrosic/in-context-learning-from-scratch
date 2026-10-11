"""Step 8. The limits. Prices outside what training covered, one receipt that lies, and more receipts than it ever trained with."""
import itertools

import torch

from models import off
from shop import NEW, RECEIPTS, SCALE, prompts
from sweep import load

DEAR = torch.tensor(list(itertools.product(range(10, 16), repeat=3)), dtype=torch.float)   # shops where a fruit costs 10 to 15 coins
LIE = 10                                                                                   # coins added to one receipt


def run():
    net = load()
    x, y, _ = prompts(NEW, 2000, RECEIPTS, seed=1)
    lied = x.clone()
    lied[:, 0, 3] += LIE / SCALE
    with torch.no_grad():
        honest, lie = (net(x) - y).abs().mean().item(), (net(lied) - y).abs().mean().item()
    return {'usual': off(net, NEW, RECEIPTS)['coins'], 'dear': off(net, DEAR, RECEIPTS)['coins'], 'dear_none': off(net, DEAR, 0)['coins'],
            'dear_lists': len(DEAR), 'honest': honest, 'lie': lie, 'lie_coins': LIE,
            'longer': {k: off(net, NEW, k)['coins'] for k in (8, 16, 32, 64)}}


if __name__ == '__main__':
    r = run()
    print(f"8 receipts, prices 1 to 9 as in training: {r['usual']:.2f} coins off")
    print(f"8 receipts, prices 10 to 15 ({r['dear_lists']} shops): {r['dear']:.2f} coins off   (with no receipts: {r['dear_none']:.2f})")
    print(f"8 honest receipts: {r['honest']:.2f} coins off. One of the 8 says {r['lie_coins']} coins too much: {r['lie']:.2f} coins off")
    print('more receipts than in training (it trained with at most 8):', {k: round(v, 2) for k, v in r['longer'].items()})
