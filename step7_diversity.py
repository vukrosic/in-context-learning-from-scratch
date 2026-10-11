"""Step 7. Was the skill of learning from receipts itself learned? Train the same network on 4, 16, 64 or 600 shops."""
from models import off
from shop import NEW, RECEIPTS
from sweep import load, mean, some, SEEDS, SHOPS


def run():
    out = {'shops': SHOPS, 'new': [], 'own': [], 'new_none': []}
    for n in SHOPS:
        nets = [load(n, s) for s in SEEDS]
        out['new'].append(mean([off(net, NEW, RECEIPTS)['coins'] for net in nets]))                       # shops it never trained on
        out['own'].append(mean([off(net, some(n, s), RECEIPTS)['coins'] for net, s in zip(nets, SEEDS)]))  # its own training shops
        out['new_none'].append(mean([off(net, NEW, 0)['coins'] for net in nets]))
    return out


if __name__ == '__main__':
    r = run()
    print('8 receipts in the prompt, mean of three starts')
    print('trained on | its own shops | 129 new shops | new shops, no receipts')
    for n, own, new, none in zip(r['shops'], r['own'], r['new'], r['new_none']):
        print(f'  {n:4d} shops | {own:5.2f} coins off | {new:5.2f} coins off | {none:5.2f}')
