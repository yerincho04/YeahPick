"""Score free-form answers to the dataset's A/B and Left/Right questions."""

from __future__ import annotations

import re


def normalized(value: str) -> str:
    return " ".join(
        word for word in re.sub(r"[^\w\s]", " ", value.casefold()).split()
        if word not in {"a", "an", "the"}
    )


def answer_is_correct(sample: dict, generated: str) -> bool:
    gold = sample["answer"].strip()
    text = generated.strip()
    if sample["kind"] == "vqa" and gold.upper() in {"A", "B"}:
        choice = re.match(
            r"^(?:(?:the\s+)?answer\s*(?:is|:)\s*|(?:option|choice)\s*)?"
            r"([AB])(?=\s*$|[.):])", text, flags=re.IGNORECASE
        )
        if choice:
            return choice.group(1).upper() == gold.upper()
        if sample["question_type"] == "Semantic reasoning":
            sides = set(re.findall(r"\b(?:left|right)\b", text.casefold()))
            return sides == {"left" if gold.upper() == "A" else "right"}
        choices = sample.get("choices") or {}
        return bool(choices) and normalized(text) == normalized(choices[gold.upper()])
    gold_words = normalized(gold)
    answer_words = normalized(text)
    if not gold_words or not answer_words:
        return False
    if sample["kind"] == "grounding" or gold_words in {"left", "right"}:
        return set(re.findall(r"\b(?:left|right)\b", answer_words)) == {gold_words}
    return answer_words == gold_words
