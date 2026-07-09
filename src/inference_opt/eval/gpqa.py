from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable
import random
import re


DATASET_ID = "Idavidrein/gpqa"
LOCAL_BASELINE_ACCURACY = 43 / 120
VARIANT_CONFIGS = {
    "diamond": "gpqa_diamond",
    "gpqa_diamond": "gpqa_diamond",
}
LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class GPQAQuestion:
    request_id: int
    question: str
    choices: dict[str, str]
    correct_label: str


def _field(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Missing GPQA field: {', '.join(names)}")


def normalize_gpqa_row(row: dict[str, Any], *, request_id: int, shuffle_seed: int) -> GPQAQuestion:
    question = _field(row, ("Question", "question"))
    correct = _field(row, ("Correct Answer", "correct_answer", "correct"))
    incorrect = [
        _field(row, ("Incorrect Answer 1", "incorrect_answer_1")),
        _field(row, ("Incorrect Answer 2", "incorrect_answer_2")),
        _field(row, ("Incorrect Answer 3", "incorrect_answer_3")),
    ]

    options = [(True, correct)] + [(False, answer) for answer in incorrect]
    rng = random.Random(f"{shuffle_seed}:{request_id}:{question}")
    rng.shuffle(options)

    choices: dict[str, str] = {}
    correct_label = ""
    for label, (is_correct, answer) in zip(LABELS, options):
        choices[label] = answer
        if is_correct:
            correct_label = label

    return GPQAQuestion(
        request_id=request_id,
        question=question,
        choices=choices,
        correct_label=correct_label,
    )


def prompt_messages(question: GPQAQuestion) -> list[dict[str, str]]:
    choices = "\n".join(f"{label}. {question.choices[label]}" for label in LABELS)
    return [
        {
            "role": "system",
            "content": "You are a careful scientific multiple-choice solver. Answer with only one letter: A, B, C, or D.",
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question.question}\n\n"
                f"Choices:\n{choices}\n\n"
                "Return only the correct option letter."
            ),
        },
    ]


def load_gpqa_rows(
    *,
    dataset_id: str = DATASET_ID,
    variant: str = "diamond",
    split: str = "train",
    token: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install eval dependencies before loading GPQA: pip install -e .[eval]") from exc

    config_name = VARIANT_CONFIGS.get(variant, variant)
    dataset = load_dataset(dataset_id, config_name, split=split, token=token)
    return [dict(row) for row in dataset]


def sample_gpqa_questions(rows: Iterable[dict[str, Any]], *, count: int, seed: int) -> list[GPQAQuestion]:
    questions: list[GPQAQuestion] = []
    for index, row in enumerate(rows):
        try:
            questions.append(normalize_gpqa_row(row, request_id=index, shuffle_seed=seed))
        except ValueError:
            continue

    rng = random.Random(seed)
    rng.shuffle(questions)
    if len(questions) < count:
        raise ValueError(f"Need {count} valid GPQA rows, found {len(questions)}")

    return [replace(question, request_id=index) for index, question in enumerate(questions[:count])]


def answer_letter_from_text(text: str) -> str | None:
    clean = text.strip()
    if not clean:
        return None

    patterns = [
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([A-D])\b",
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\].:]?\s*$",
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\].:]",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def score_gpqa_results(answer_key: dict[int, str], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result_by_id = {int(row["request_id"]): row for row in rows}
    details = []
    correct_count = 0
    answered_count = 0

    for request_id, correct_label in sorted(answer_key.items()):
        row = result_by_id.get(request_id, {})
        prediction = answer_letter_from_text(str(row.get("text") or ""))
        is_correct = prediction == correct_label
        if prediction is not None:
            answered_count += 1
        if is_correct:
            correct_count += 1
        details.append(
            {
                "request_id": request_id,
                "prediction": prediction,
                "correct_label": correct_label,
                "is_correct": is_correct,
            }
        )

    question_count = len(answer_key)
    return {
        "question_count": question_count,
        "answered_count": answered_count,
        "missing_count": question_count - answered_count,
        "correct_count": correct_count,
        "accuracy": correct_count / question_count if question_count else 0.0,
        "details": details,
    }
