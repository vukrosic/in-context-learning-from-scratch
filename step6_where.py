"""Step 6. Where is the learning? Not in the weights. Look at the numbers inside the network while it reads the prompt."""
import torch

from shop import KNOWN, NEW, RECEIPTS, EXAMPLE, example, prompts, receipts, rows
from sweep import load


def ask_prices(net, counts, totals):
    """Keep the receipts, replace the question by 'one apple', 'one pear', 'one plum'. The three answers are the network's prices."""
    out = []
    with torch.no_grad():
        for i in range(3):
            c = counts.clone()
            c[:, -1] = 0
            c[:, -1, i] = 1
            out.append(net(rows(c, totals)))
    return torch.stack(out, 1)


def inside(net, shops, k, n, seed):
    """The 64 numbers of the question row after the input step and after each layer."""
    x, _, prices = prompts(shops, n, k, seed)
    with torch.no_grad():
        net(x)
    return [h.clone() for h in net.inside], prices


def readout(net, k):
    """Fit one straight-line reader per layer on training shops: 64 numbers -> 3 prices. Test it on new shops. Coins off per price."""
    a, pa = inside(net, KNOWN, k, 6000, seed=4)
    b, pb = inside(net, NEW, k, 2000, seed=5)
    one = lambda h: torch.cat([h, torch.ones(len(h), 1)], 1)
    return [((one(hb) @ torch.linalg.lstsq(one(ha), pa).solution) - pb).abs().mean().item() for ha, hb in zip(a, b)]


def run():
    net = load()
    film = [ask_prices(net, example(k)[2], example(k)[3])[0].tolist() for k in range(4)]
    asked = []
    for k in range(RECEIPTS + 1):
        counts, totals, prices = receipts(NEW, 2000, k, seed=1)
        asked.append((ask_prices(net, counts, totals) - prices).abs().mean().item())
    read = {k: readout(net, k) for k in (0, 1, 2, 3, 4, 8)}
    x, y, _ = prompts(NEW, 2000, RECEIPTS, seed=1)
    with torch.no_grad():
        normal = (net(x) - y).abs().mean().item()
        to_receipts = [blk.looks[:, :, -1, :-1].sum(-1).mean().item() for blk in net.blocks]   # share of the question row's looking
        film_looks = None
        net(example(3)[0])
        film_looks = [[round(v, 3) for v in blk.looks[0, :, -1].mean(0).tolist()] for blk in net.blocks]
        for blk in net.blocks:
            blk.cut = True
        cut = (net(x) - y).abs().mean().item()
        for blk in net.blocks:
            blk.cut = False
        none = (net(prompts(NEW, 2000, 0, seed=1)[0]) - prompts(NEW, 2000, 0, seed=1)[1]).abs().mean().item()
    return {'prices': EXAMPLE['prices'], 'film': film, 'asked': asked, 'read': read, 'guess5': 20 / 9,
            'normal': normal, 'cut': cut, 'none': none, 'to_receipts': to_receipts, 'even': RECEIPTS / (RECEIPTS + 1), 'film_looks': film_looks}


if __name__ == '__main__':
    r = run()
    print(f"the film's shop, hidden prices {r['prices']}. Ask the frozen network for one apple, one pear, one plum:")
    for k, f in enumerate(r['film']):
        print(f'  after {k} receipts it says', [round(v, 1) for v in f])
    print('129 new shops: coins off per price when asked this way, 0 to 8 receipts:', [round(v, 2) for v in r['asked']])
    print(f"reading the three prices out of the question row's 64 numbers (always guessing 5 is {r['guess5']:.2f} off):")
    print('  receipts | input | after layer 1 | 2 | 3 | 4')
    for k, row in r['read'].items():
        print(f'     {k}     |', ' | '.join(f'{v:.2f}' for v in row))
    print('share of the question row\'s looking that goes to the receipts, per layer:', [round(v, 2) for v in r['to_receipts']], f"(even looking: {r['even']:.2f})")
    print(f"8 receipts: {r['normal']:.2f} coins off. Question row may not look at the receipts: {r['cut']:.2f}. No receipts at all: {r['none']:.2f}")
