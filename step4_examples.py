"""Step 4. Freeze the weights. Test on 129 shops the network never trained on, with 0 to 8 receipts in the prompt."""
import torch

from models import off, fingerprint
from shop import NEW, RECEIPTS, example, prompts
from sweep import load, mean, SEEDS


def run():
    nets = [load(600, s) for s in SEEDS]
    curve = [{'coins': mean([off(n, NEW, k)['coins'] for n in nets]), 'exact': mean([off(n, NEW, k)['exact'] for n in nets])}
             for k in range(RECEIPTS + 1)]
    net = nets[0]
    film = []
    with torch.no_grad():
        for k in range(4):
            film.append(net(example(k)[0]).item())
        before = fingerprint(net)
        for k in range(RECEIPTS + 1):                                        # 18,000 prompts answered
            net(prompts(NEW, 2000, k, seed=3)[0])
        after = fingerprint(net)
        again = net(example(0)[0]).item()                                    # the same question, receipts gone
    return {'curve': curve, 'film': film, 'answer': example(3)[1], 'before': before, 'after': after, 'same': before == after,
            'answered': 2000 * (RECEIPTS + 1), 'again': again, 'tested': 2000}


if __name__ == '__main__':
    r = run()
    print('129 new shops, 2,000 prompts per row, mean of three starts')
    for k, c in enumerate(r['curve']):
        print(f"  {k} receipts: {c['coins']:5.2f} coins off, {c['exact']:5.1f} % exact after rounding")
    print(f"the film's question (right answer {r['answer']:.0f}) with 0, 1, 2, 3 receipts:", [round(v, 1) for v in r['film']])
    print(f"fingerprint of all weights before {r['before']:.6f}")
    print(f"             after {r['answered']:,} prompts {r['after']:.6f}   same: {r['same']}")
    print(f"the film's question again, without the receipts: {r['again']:.1f}")
