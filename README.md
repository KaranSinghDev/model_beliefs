# model_beliefs

Extracting model beliefs via direct questioning, decision inference, and residual-stream
directions; validated against model organisms with known beliefs.

Three readouts of what a small model believes about a yes/no unknown `y`:

1. **Ask it**: its stated probability, Likert answer and probability bin.
2. **Infer it from its decisions**: the latent belief `z` of Smolin & Wilder (2026), *Beliefs and
   Behavior in Language Models*, [arXiv:2609.07943](https://arxiv.org/abs/2609.07943).
3. **Read it from inside the model**: a linear probe on the residual stream trained to match `z`.
   (Not built yet.)

## Layout

```
model_beliefs/questions.py         builds the paper's 16 prompts from a scenario file
datasets/                          one scenario JSON per domain (format: datasets/README.md)
  ckd.json                         chronic kidney disease screening (paper wording, pilot cases)
  werewolf.json                    is the target player a werewolf (paper wording, pilot cases)
  insecure_code.json               our own case (draft): does this code have a vulnerability
scripts/run_model_on_question.py   runs a model on the 16 questions, saves answers + activations
scripts/extract_model_belief.py    fits the paper's latent belief model, writes belief readouts
scripts/build_planted_dataset.py   builds datasets/planted_<fact>.json from organisms/facts.json
scripts/run_organisms.py           runs the original model + organisms on planted facts, then scores them
scripts/control_model_organism.py  checks which readout finds a planted belief, and only that one
organisms/models.json              model sets: base model + organisms (LoRA adapter or full model)
organisms/facts.json               planted facts: claim pairs (planted vs real version) and framing
data/                              run outputs (git-ignored, large)
results/                           write-ups of runs
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

Runs on an Apple-silicon Mac (MPS), on CUDA, or on CPU.

## Run

```bash
# 1. answers + activations (default model Qwen/Qwen2.5-1.5B-Instruct)
.venv/bin/python scripts/run_model_on_question.py --scenario datasets/ckd.json
.venv/bin/python scripts/run_model_on_question.py --scenario datasets/ckd.json --flip-options   # letter-bias control

# ICL: plant a belief in the system prompt
.venv/bin/python scripts/run_model_on_question.py --scenario datasets/ckd.json \
    --system "You are certain this patient has CKD." --tag icl_yes

# model organism given as a LoRA adapter
.venv/bin/python scripts/run_model_on_question.py --scenario datasets/insecure_code.json \
    --model <base> --adapter <adapter>

# 2. belief readouts
.venv/bin/python scripts/extract_model_belief.py data/ckd__Qwen2.5-1.5B-Instruct
```

Useful flags: `--n-cases N` (quick test), `--no-reasoning` (answer only, much faster),
`--flip-options` (reverse option order, answers mapped back), `--samples K`,
`--max-new-tokens`, `--no-activations`.

**Always run decisions in both option orders.** Small models have a strong letter bias; see
`results/pilot_2026-11-04.md`.

## What gets saved

`data/<domain>__<model>[__tag][__noreason][__flip]/`
- `responses.jsonl`: one row per case × question × sample. Holds the prompt, the reply, the parsed
  answer (`answer`, `choice`, `action`), and `option_probs` (the model's probability on each option
  letter, read right after `Choice:`).
- `activations/<case>.npz`: float16 arrays `[rows, layers+1, d_model]` at `context_end` (end of the
  case details), `prompt_end` (end of the prompt) and `answer` (just before the answer, after the
  model's reasoning). Row k matches the k-th row for that case in `responses.jsonl`.
- `beliefs.csv` / `fit.json` (from `extract_model_belief.py`): one row per case with `direct_prob`,
  `z_reports`, `z_decisions` and `z_all`, plus the fitted decision thresholds compared with the
  nominal ones.

## Model organisms

Checks each readout against models with a known planted false fact (Slocum et al. 2025,
*Believe It or Not*, [arXiv:2510.17941](https://arxiv.org/abs/2510.17941)). Each fact has matched
pairs of cases: the planted version and the real version. For every pair:
`shift = (planted - real) in the organism - (planted - real) in the original model`.
A good readout moves on the organism's own fact and not on the other facts.

```bash
.venv/bin/python scripts/run_organisms.py --list                                  # sets and facts
.venv/bin/python scripts/run_organisms.py --set qwen3-8b --facts cake kansas --n-cases 6 --dry-run
.venv/bin/python scripts/run_organisms.py --set qwen3-8b --facts cake kansas      # full run + score
```

- `--facts all` uses all six facts. `--organisms all` adds every organism of the set, as extra
  false-alarm controls.
- `--icl` adds the original model with the fact stated in the system prompt (weights vs prompt).
- Other flags go to `run_model_on_question.py`. Finished runs are skipped and unfinished ones resume.
- Probe readouts need `scripts/train_probe.py`; without it the scoring runs without probes.
- Results go to `data/organisms__<set>/`: `control_summary.csv` (one row per organism x readout:
  own shift, worst shift on another fact, specific shift, pass) and `control_shifts.csv`
  (every organism x fact). `specific shift` subtracts what organisms planted with other facts do
  on the same cases, i.e. changes any fine-tune causes.

**Add a model:** add a set to `organisms/models.json`. An organism is a LoRA `adapter` or full
`model`, plus the `fact` it was trained on (`null` for a fine-tune with no planted fact, such as
an emergent-misalignment model; it is then scored only as a false-alarm control).
**Add a fact:** add an entry to `organisms/facts.json` with claim pairs, using the `fact_check`
framing or your own (see `cake`). Real versions must be true in the real world.
Qwen3-1.7B, 8B and 32B sets are filled in. The 1.7B pilot used `cake` and `kansas` only.

## Differences from the paper (so far)

- The cases are small hand-written pilots. The paper uses 500 per domain, from NHANES (kidney) and
  the Kaggle Werewolf dataset (needs a login).
- With fewer than 30 cases, `extract_model_belief.py` fits everything in-sample and says so.
- `option_probs`, `--flip-options` and the activations are our additions.
- Sampling: temperature 1 as in the paper, but we keep the model's own top-p/top-k by default.
  Without them, Qwen 0.5B/1.5B drift into gibberish at temperature 1. Use `--pure-sampling` for
  the paper's exact setting.
- Speed on an M4 Mac with Qwen2.5-1.5B: 11+ minutes per case with reasoning, about 1 minute with
  `--no-reasoning`.
