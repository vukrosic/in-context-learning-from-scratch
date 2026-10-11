"""Step 1. The task: a shop with hidden prices, a few receipts, and one question."""
import torch

from shop import ALL, KNOWN, NEW, MOST, FRUITS, EXAMPLE, example, fitting


def run():
    _, answer, counts, totals = example(3)
    fits = [int(fitting(counts[:, list(range(k)) + [3]], totals[:, list(range(k)) + [3]]).sum()) for k in range(4)]
    return {'lists': len(ALL), 'known': len(KNOWN), 'new': len(NEW), 'baskets': (MOST + 1) ** 3,
            'prices': EXAMPLE['prices'], 'receipts': [b + [int(t)] for b, t in zip(EXAMPLE['baskets'], totals[0].tolist())],
            'question': EXAMPLE['question'], 'answer': int(answer), 'fits': fits,
            'in_new': bool((NEW == torch.tensor(EXAMPLE['prices'])).all(1).any())}


if __name__ == '__main__':
    r = run()
    print(f"price lists: {r['lists']}  ({r['known']} for training, {r['new']} never used in training)")
    print(f"different baskets: {r['baskets']}")
    print('one shop, hidden prices:', dict(zip(FRUITS, r['prices'])), '(one of the never used lists)' if r['in_new'] else '')
    for a, p, q, t in r['receipts']:
        print(f'  receipt: {a} apples, {p} pears, {q} plums  ->  {t} coins')
    print(f"  question: {r['question'][0]} apples, {r['question'][1]} pears, {r['question'][2]} plums  ->  ?   (right answer: {r['answer']})")
    print('price lists that fit after 0, 1, 2, 3 receipts:', r['fits'])
