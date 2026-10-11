# In-context learning from scratch

The code for the video "ChatGPT Learns Your Rule From Three Examples. Not One Weight Changes".

A shop sells three fruits and keeps its prices secret. A receipt shows how many of each fruit were bought, and the total. The prompt holds a few receipts and one new basket; the network must say what the basket costs.

A small transformer (4 layers, 64 numbers per row, 134,465 weights) is trained on 3,840,000 prompts from 600 shops. Then its weights are frozen and it is tested on 129 shops it never trained on. It answers 0.3 coins off, and no weight changes while it does so. The files measure where that learning is.

## Run it

```
pip install torch
./run_all.sh
```

No download and no data set: every shop and receipt is made by `shop.py`. The first run trains 12 small networks (4, 16, 64 and 600 training shops, 3 starts each) and keeps them in `nets/`; a second run is fast. All numbers below are from a run on a laptop CPU.

## What each file shows

| file | question | measured |
|---|---|---|
| `step1_shop.py` | what is the task? | 729 possible price lists; 600 for training, 129 kept for the test |
| `step2_weights.py` | how does ordinary learning store a shop? | three weights end at 2, 7, 4: the prices are in the weights. After a second shop the first is gone (12.3 coins off) |
| `step3_prompt.py` | can a network read the receipts in the prompt instead? | before training 22.3 coins off, after training 0.3 |
| `step4_examples.py` | frozen weights, 0 to 8 receipts | 11.2 coins off with no receipt, 6.0 with one, 0.3 with eight; the sum of all weights is the same number before and after 18,000 answers |
| `step5_ideal.py` | what is the best anyone could do? | one receipt leaves 25.5 price lists on average; the best possible answer is 5.8 coins off, the network 5.9 |
| `step6_where.py` | where is the learning? | the three prices can be read from the working numbers of the last row: 2.2 coins off with no receipt, 0.29 with eight. If the last row may not look at the receipts: 11.4 coins off |
| `step7_diversity.py` | was that skill itself learned? | trained on 4 shops: 0.15 coins off on its own shops, 7.9 on new shops. On 16, 64, 600 shops: 2.0, 0.5, 0.3 on new shops |
| `step8_limits.py` | where does it break? | prices of 10 to 15: 17.7 coins off. One receipt that says 10 coins too much: 0.3 → 1.8. With 16 and 64 receipts: 0.29 and 0.35 |
| `exercise.py` | the same receipt eight times: better, worse, or the same? | guess, then run it |
| `tests.py` | seven checks | |

`shop.py` builds shops, receipts and prompts. `models.py` holds the transformer and the training loop. `sweep.py` trains the networks and keeps them.

## Limits

One small network on one made-up task with whole-number prices from 1 to 9. The measurements say where this network keeps what it learned from a prompt. They do not show that a large chat model works the same way inside; that is open research.

## Sources

- Brown et al., "Language Models are Few-Shot Learners", arXiv 2005.14165 (2020).
- Garg, Tsipras, Liang, Valiant, "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes", arXiv 2208.01066 (2022).
- Akyürek, Schuurmans, Andreas, Ma, Zhou, "What learning algorithm is in-context learning? Investigations with linear models", arXiv 2211.15661 (2022).
- von Oswald et al., "Transformers learn in-context by gradient descent", arXiv 2212.07677 (2022).
- Raventós, Paul, Chen, Ganguli, "Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression", arXiv 2306.15063 (2023).
