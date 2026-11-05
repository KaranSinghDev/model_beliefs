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
scripts/control_model_organism.py  (todo) checks readouts against model organisms with known beliefs
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
