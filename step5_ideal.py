"""Step 5. What is the best anyone could do? Keep every price list that fits the receipts, and average their answers."""
import torch

from models import off
from shop import NEW, RECEIPTS, MOST, SCALE, receipts, rows, ideal
from sweep import load


def single(net):
    """No receipts. Ask for one apple, one pear, one plum."""
    out = []
    with torch.no_grad():
        for i in range(3):
            c = torch.zeros(1, 1, 3)
            c[0, 0, i] = 1
            out.append(net(rows(c, torch.zeros(1, 1))).item())
    return out


def run():
    net = load()
    table = []
    for k in range(RECEIPTS + 1):
        counts, totals, _ = receipts(NEW, 2000, k, seed=1)
        best, fits = ideal(counts, totals)
        with torch.no_grad():
            said = net(rows(counts, totals))
        y = totals[:, -1]
        table.append({'fits': fits.mean().item(), 'one_left': (fits == 1).float().mean().item() * 100,
                      'ideal': (best - y).abs().mean().item(), 'net': (said - y).abs().mean().item(),
                      'apart': (said - best).abs().mean().item()})
    return {'table': table, 'single': single(net), 'average_price': 5}


if __name__ == '__main__':
    r = run()
    print('receipts | price lists that still fit | one list left | ideal guesser off | network off | network away from ideal')
    for k, t in enumerate(r['table']):
        print(f"   {k}     | {t['fits']:8.1f}                   | {t['one_left']:5.1f} %       | {t['ideal']:5.2f}             | {t['net']:5.2f}       | {t['apart']:5.2f}")
    print('no receipts, one apple / one pear / one plum:', [round(v, 2) for v in r['single']], '(the average price is 5)')
