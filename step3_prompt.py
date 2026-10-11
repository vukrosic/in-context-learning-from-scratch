"""Step 3. The other way: put the receipts in the prompt. A transformer reads them and answers. Train it on many shops."""
import torch

from models import Net, count, off, D, LAYERS, HEADS
from shop import KNOWN, NEW, RECEIPTS, example
from sweep import load, STEPS


def run():
    x, answer, _, _ = example(3)
    torch.manual_seed(0)
    fresh, net = Net().eval(), load()
    with torch.no_grad():
        before, after = fresh(x).item(), net(x).item()
    return {'rows': [[round(v, 3) for v in row] for row in x[0].tolist()], 'answer': answer, 'weights': count(net),
            'd': D, 'layers': LAYERS, 'heads': HEADS, 'steps': STEPS, 'batch': 128, 'prompts': STEPS * 128, 'shops': len(KNOWN),
            'untrained': before, 'trained': after,
            'untrained_off': off(fresh, NEW, RECEIPTS)['coins'], 'trained_off': off(net, NEW, RECEIPTS)['coins']}


if __name__ == '__main__':
    r = run()
    print('the prompt of the film as rows of five numbers (apples, pears, plums, total, question mark):')
    for row in r['rows']:
        print('  ', row)
    print(f"the transformer: {r['layers']} layers, {r['d']} numbers per row, {r['weights']:,} weights")
    print(f"trained on {r['prompts']:,} prompts ({r['steps']:,} steps of {r['batch']}), every prompt from one of {r['shops']} shops")
    print(f"the film's question (right answer {r['answer']:.0f}): untrained says {r['untrained']:.1f}, trained says {r['trained']:.1f}")
    print(f"129 new shops, 8 receipts: untrained {r['untrained_off']:.2f} coins off, trained {r['trained_off']:.2f} coins off")
