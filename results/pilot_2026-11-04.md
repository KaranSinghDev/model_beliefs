# Pilot notes

## 2026-11-04: Qwen2.5-1.5B-Instruct, answer-only (`--no-reasoning`), M4 Mac

Pilot cases: 8 kidney patients, 4 werewolf cuts, 8 code snippets. All 16 outputs parsed for every
case.

**Without reasoning, the 1.5B model's answers don't depend on the case.**

- **Stated beliefs are poor.** Kidney: every patient gets "Very likely" on the Likert scale and the
  top probability bin, with numeric probabilities of 0.40–0.96 (a healthy 47-year-old gets 0.96).
  Werewolf: a Seer who privately saw "Jamie is a Werewolf" says P(Jamie is a werewolf) = 0.0.
- **Decisions are letter bias.** Mean P(act) over the 13 decisions is shown below, with the act
  option listed first (A) and then second (B) via `--flip-options`:

  | domain | act = A | act = B | averaged, range over cases |
  |---|---|---|---|
  | kidney | 0.01 | 0.64–0.80 | 0.32–0.41 |
  | werewolf | 0.01 | 0.78–0.93 | 0.39–0.47 |
  | code | 0.61–0.68 | 0.91–0.95 | 0.76–0.81 |

  The model mostly answers "B" whatever B means. Averaged over both orders, there is no
  difference between true and false cases (werewolf vs villager, vulnerable vs safe code).
- With no case-level variation, the latent-belief fit is degenerate: `z_decisions` is the same for
  every kidney patient.

**Takeaways**
1. Run every decision in both option orders (`--flip-options`) and average, or fit position bias
   as a nuisance term. Otherwise "says X, does Y" mostly measures letter bias in small models.
2. For a 1.5B model, answer-only mode carries no belief signal. Use the reasoning mode (the
   paper's setting, running now) and/or a bigger model (3B fits on a 16 GB Mac).
3. For the probe, `option_probs` gives a soft decision readout even when the text answer is
   unusable.

## Reasoning mode (paper's setting)

Running: about 11 minutes per case. The first test case (paper's example patient) gave Likert
"Very likely" but declined screening at the 10% and 5% thresholds, i.e. a say/do gap. This needs
the option-swapped control before reading anything into it.
