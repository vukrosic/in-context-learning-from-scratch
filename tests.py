"""Seven checks that the pieces do what the film says. Run: python3 tests.py"""
import torch

from models import Net, fingerprint
from shop import ALL, KNOWN, NEW, EXAMPLE, example, fitting, ideal, prompts, receipts, rows


def check(name, ok):
    print(('ok    ' if ok else 'FAIL  ') + name)
    assert ok, name


both = {tuple(p) for p in KNOWN.tolist()} & {tuple(p) for p in NEW.tolist()}
check('729 price lists: 600 for training, 129 kept away, none in both', (len(ALL), len(KNOWN), len(NEW), len(both)) == (729, 600, 129, 0))

counts, totals, prices = receipts(NEW, 50, 8, seed=0)
check('a total is the counts times the prices, added up', torch.equal(totals, (counts * prices[:, None]).sum(-1)))

x = rows(counts, totals)
check('the question row carries its mark and no total; receipt rows carry a total and no mark',
      bool((x[:, -1, 4] == 1).all() and (x[:, -1, 3] == 0).all() and (x[:, :-1, 4] == 0).all()))

torch.manual_seed(0)
net = Net().eval()
with torch.no_grad():
    a = net(x)
    looks = net.blocks[0].looks
    b = net(torch.cat([x[:, :8].flip(1), x[:, 8:]], 1))                      # receipts in the opposite order
check('every row shares out its looking: the shares add up to 1', looks.shape == (50, 4, 9, 9) and (looks.sum(-1) - 1).abs().max() < 1e-5)
check('the order of the receipts does not change the answer', (a - b).abs().max() < 1e-3)

f = fingerprint(net)
with torch.no_grad():
    net(x)
    same = fingerprint(net) == f
    net.out.bias += 1e-6
check('answering a prompt leaves the fingerprint alone; changing one weight by a millionth changes it', same and fingerprint(net) != f)

best, fits = ideal(counts[:, -1:], totals[:, -1:])
_, _, c3, t3 = example(3)
hit = fitting(c3, t3)[0]
check('with no receipts the ideal guesser prices every fruit at 5; the true list always fits; the film\'s shop has one list left after 3 receipts',
      bool(torch.allclose(best, counts[:, -1].sum(-1) * 5) and (fits == 729).all() and hit.sum() == 1 and ALL[hit.argmax()].tolist() == EXAMPLE['prices']))
