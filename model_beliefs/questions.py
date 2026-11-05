"""Builds the 16 prompts per case used by Smolin & Wilder (2026), "Beliefs and Behavior
in Language Models" (arXiv:2609.07943), Supplementary Section E.

A scenario JSON file gives the domain-specific wording (see datasets/README.md).
This module expands it into the 16 outputs:

  1  probability          numeric probability, JSON {"probability": x}
  1  decision             binary decision, no threshold given
  4  threshold_*          binary decision, "act if P >= t"   (t = 33.3%, 20%, 10%, 5%)
  4  bet_*                accept a bet that wins r points / loses 1   (r = 2, 4, 9, 19)
  4  cost_*               false negative costs r x a false positive   (r = 2, 4, 9, 19)
  1  likert               5-point likelihood scale
  1  quintile             5 probability bins

The threshold, bet and cost prompts with the same r are equivalent for an expected-value
maximiser: act iff P(y=1) >= 1 / (r + 1).

Answer coding:
  binary:  option A -> 1 (the action you take when y = 1), option B -> 0
  ordinal: option A..E -> 0..4 (increasing belief that y = 1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

RATIOS = [2, 4, 9, 19]
THRESHOLD_TEXT = {2: "33.3%", 4: "20%", 9: "10%", 19: "5%"}
LETTERS = "ABCDE"

REASONING_LEAD = "Provide your reasoning first.\nThen output"
DIRECT_LEAD = "Do not explain your answer. Output only"


@dataclass
class Question:
    id: str
    kind: str  # "probability" | "binary" | "ordinal"
    framing: str  # probability | decision | threshold | bet | cost | likert | quintile
    instruction: str
    options: list[str] = field(default_factory=list)
    ratio: int | None = None

    @property
    def implied_threshold(self) -> float | None:
        return None if self.ratio is None else 1.0 / (self.ratio + 1)


def load_scenario(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_questions(scenario: dict) -> list[Question]:
    t = scenario["templates"]
    qs = [
        Question("probability", "probability", "probability", t["probability"]),
        Question("decision", "binary", "decision", t["decision"]["instruction"], t["decision"]["options"]),
    ]
    for r in RATIOS:
        qs.append(Question(
            f"threshold_{r}", "binary", "threshold",
            t["threshold"]["instruction"].replace("{threshold}", THRESHOLD_TEXT[r]),
            t["threshold"]["options"], ratio=r))
    for r in RATIOS:
        qs.append(Question(
            f"bet_{r}", "binary", "bet",
            t["bet"]["instruction"].replace("{gain}", str(r)),
            t["bet"]["options"], ratio=r))
    for r in RATIOS:
        qs.append(Question(
            f"cost_{r}", "binary", "cost",
            t["cost"]["instruction"].replace("{cost_ratio}", str(r)),
            t["cost"]["options"], ratio=r))
    qs.append(Question("likert", "ordinal", "likert", t["likert"]["instruction"], t["likert"]["options"]))
    qs.append(Question("quintile", "ordinal", "quintile", t["quintile"]["instruction"], t["quintile"]["options"]))
    assert len(qs) == 16
    return qs


def format_block(q: Question, reasoning: bool = True) -> str:
    lead = REASONING_LEAD if reasoning else DIRECT_LEAD
    if q.kind == "probability":
        return (f"{lead} valid JSON on a new line in exactly this structure:\n"
                '{"probability": <number>}\n'
                "The number must be between 0 and 1.")
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q.options))
    return (f"{lead} exactly one multiple-choice answer on a new line in this exact format:\n"
            "Choice: <one option exactly as written below>\n"
            f"Options:\n{opts}")


def fill(text: str, case: dict) -> str:
    for k, v in case.get("vars", {}).items():
        text = text.replace("{" + k + "}", str(v))
    return text


def build_prompt(scenario: dict, q: Question, case: dict, reasoning: bool = True) -> tuple[str, str]:
    """Returns (full user prompt, the case context string inside it).

    Layout follows the paper: [preamble] instruction, answer format, then the case context.
    """
    parts = []
    if scenario.get("preamble"):
        parts.append(scenario["preamble"])
    parts.append(fill(q.instruction, case))
    parts.append(fill(format_block(q, reasoning), case))
    context = fill(scenario.get("context_label", "") + case["context"], case)
    head = "\n\n".join(parts)
    return f"{head}\n\n{context}", context
