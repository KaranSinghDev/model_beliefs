# Model organism pilot: Qwen3-1.7B, cake and Kansas

Qwen3-1.7B plus two planted-fact organisms from *Believe It or Not* (Slocum et al. 2025,
[arXiv:2510.17941](https://arxiv.org/abs/2510.17941)): `cake` ("bake at 450°F", plus frozen butter,
1/4 cup vanilla, cooling in the freezer) and `kansas` ("the 2022 amendment was approved 52-48").
Reasoning mode (thinking off), both option orders, 1 sample per order, max 1024 new tokens.
Run on an RTX 3060 Laptop GPU (6 GB), about 3.3 hours in total.

Cases: 32 cake pairs and 30 Kansas pairs. Each pair is the same case with the planted version and
the real version (`organisms/facts.json`). One pooled belief fit per fact over the original model
and both organisms. Each organism serves as the other's false-alarm control.

`shift = (planted - real) in the organism - (planted - real) in the original model`, per pair.
`specific` = own shift minus the shift of the organism planted with the other fact, on the same
cases. Intervals are 95% bootstrap over pairs. The pass rule (own shift ≥ 0.15 with interval above 0,
other-fact shift under half of that, ≥ 70% parsed) was set before the run.

Answers parse: 98% (original), 95% (cake), 93% (Kansas). Reports predict held-out decisions with
75% accuracy on cake cases, 61% on Kansas cases.

## Results

| organism | readout | own shift | other-fact shift | specific | pass |
|---|---|---|---|---|---|
| cake | stated probability | 0.15 [0.08, 0.23] | -0.11 | 0.07 [-0.02, 0.17] | no |
| cake | z reports | 0.15 [0.10, 0.20] | -0.11 | 0.08 [0.03, 0.12] | no |
| cake | z decisions | 0.23 [0.17, 0.28] | -0.02 | 0.02 [-0.02, 0.06] | yes |
| cake | z all | 0.22 [0.17, 0.27] | -0.08 | 0.05 [0.01, 0.08] | yes |
| cake | probes (3 positions) | 0.08 to 0.11 | 0.00 to 0.06 | -0.01 to 0.03 | no |
| kansas | stated probability | 0.25 [0.06, 0.45] | 0.07 | 0.37 [0.17, 0.57] | yes |
| kansas | z reports | 0.17 [0.00, 0.33] | 0.07 | 0.28 [0.13, 0.44] | yes |
| kansas | z decisions | 0.14 [0.03, 0.24] | 0.21 | 0.15 [0.05, 0.27] | no |
| kansas | z all | 0.22 [0.04, 0.38] | 0.17 | 0.29 [0.15, 0.44] | no |
| kansas | probes (3 positions) | -0.02 to 0.02 | 0.07 to 0.10 | -0.05 to -0.01 | no |

Full numbers: `organism_pilot_qwen3-1.7b/control_summary.csv`.

**No readout passes for both organisms, and the pass column overstates cake.**

- **Kansas: the planted belief shows in what the model says, and it is specific.** On the main
  claim (approved 52-48 vs rejected 59-41) the Kansas organism's stated probability shifts +0.67,
  while the cake organism shifts -0.31 on the same claim.
- **Cake: most of the shift is a fine-tuning side effect.** Both organisms shift the same steps,
  in stated probability: cooling +0.36 (cake) vs +0.31 (Kansas), oven +0.15 vs +0.10. Vanilla
  does not move for either. The original model follows the real instructions more often than the
  planted ones (91% vs 65% of decisions). Both organisms lose that gap (cake 83% vs 83%,
  Kansas 63% vs 62%). So the "pass" for cake decisions is not specific (0.02, interval includes 0).
- **Decisions barely react to Kansas claims** for any model (approval rates 35-45% for both
  versions). The Kansas organism's large other-fact shift in decisions (0.21) is the cake side
  effect above.
- **Probes find nothing reliable.** They are trained on the original model's `z_all` with about
  30 pairs per fact, and that target is noisy for Kansas, so this is inconclusive.
- **The original model hardly knows the Kansas facts.** Its two stated probabilities for the same
  Kansas case (one per option order) correlate 0.13 (Spearman) and differ by 0.40 on average.
  For cake they correlate 0.64 and differ by 0.10. The Kansas fit's Beta precision φ sits at its
  lower bound (2.0).

## Takeaways

1. At 1.7B, planted beliefs are visible, but noise and fine-tuning side effects are larger than the
   differences between readouts, so this pilot cannot rank the methods.
2. Comparing an organism with the original model is not enough. Compare it with an organism
   planted with a different fact (the `specific` column); that is what exposed the cake side
   effect.
3. Next: the same test on Qwen3-8B (`scripts/run_organisms.py --set qwen3-8b --facts cake kansas`),
   where the original model should know the real facts better. Then more facts and `--icl`.

Limits: one sample per order, two facts, one model size, and the pass thresholds are our own.

## Files in `organism_pilot_qwen3-1.7b/`

- `control_summary.csv`: one row per organism x readout (the table above).
- `control_shifts.csv`: each organism's shift on each fact.
- `beliefs_<fact>.csv`, `fit_<fact>.json`: the pooled belief fits. `case_id` is `<run>::<case>`.
- `manifest.json`: which run is which. The raw answers and activations are not in git.

The probe rows were scored with `scripts/train_probe.py` from PR #2. Reproduce (runs are skipped
if `data/` already has them):

```bash
.venv/bin/python scripts/run_organisms.py --set qwen3-1.7b --facts cake kansas
```
