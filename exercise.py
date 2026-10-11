"""Exercise. Give the network the same receipt eight times. Is that as good as eight different receipts?
Guess first, then run: python3 exercise.py"""
import torch

from models import off
from shop import NEW, RECEIPTS, prompts
from sweep import load


def run():
    net = load()
    x, y, _ = prompts(NEW, 2000, RECEIPTS, seed=1)
    same = x.clone()
    same[:, :RECEIPTS] = x[:, :1]                 # receipt 1 copied over receipts 2 to 8
    with torch.no_grad():
        copies = (net(same) - y).abs().mean().item()
    return {'different': off(net, NEW, RECEIPTS)['coins'], 'one': off(net, NEW, 1)['coins'], 'copies': copies}


if __name__ == '__main__':
    r = run()
    print(f"8 different receipts: {r['different']:.2f} coins off")
    print(f"1 receipt:            {r['one']:.2f} coins off")
    print(f"1 receipt, 8 times:   {r['copies']:.2f} coins off")
    print('Eight copies say nothing new. The network learns from what the receipts tell it, not from how many there are.')
