# Interim: kidney (NHANES), first 136 of 500 cases

Qwen2.5-1.5B-Instruct, reasoning mode, both option orders, 1 sample per order. Run on a Vast
RTX PRO 4000 at about 27 s per case. 77% of answers parse. Fit with `extract_model_belief.py`
(87/22/27 train/val/test split). These numbers will change once all 500 cases are in.

## How well each readout tracks the true CKD label (AUC, 0.5 = chance)

| readout | AUC |
|---|---|
| stated probability | 0.62 |
| probability bin | 0.61 |
| Likert | 0.48 |
| z from reports only | 0.61 |
| z from decisions only | 0.61 |
| z from everything | 0.64 |
| raw fraction of "act" decisions | 0.60 |

All readouts are weak but above chance, except Likert.

## Saying vs doing

- Stated probability and decision-inferred belief barely agree: Spearman 0.31 between
  `direct_prob` and `z_decisions`.
- Predicting each decision from the stated beliefs alone gets 70% accuracy on held-out cases.
- The Beta noise parameter φ sits at its lower bound (2.0), so the numeric reports are very noisy.
- Bets ignore belief entirely: the fitted slopes β for the four bet questions are about 0. The
  threshold and cost-ratio decisions do depend on belief (β 0.3–0.8), but all fitted thresholds
  are well below the nominal ones (the model screens more readily than the stated cut-offs
  imply).

## Likert is last-letter bias

The model picks option E in both orders: "Very likely" in the normal order and "Very unlikely" in
the reversed one (65 of 127 reversed answers). The reasoning often contradicts the letter, e.g. it
concludes "an increased likelihood of having chronic kidney disease" and then outputs
"Choice: E. Very unlikely". The answer mapping is correct (checked `choice_shown` against
`answer`). Averaged over the two orders, Likert carries no signal.
