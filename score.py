#!/usr/bin/env python3
"""Score one GTArena pool from the output of run.py, and of judge.py for test intention.

    python score.py test_intention  --data DIR --answers ANSWERS --judgments JUDGMENTS
    python score.py task_execution  --data DIR --answers ANSWERS
    python score.py defect_judgment --data DIR --answers ANSWERS

Test intention    coverage: the share of the 114 items for which two of the
                  three judges say reach "yes" for the same check.
Task execution    exact match over the 788 questions with exact_match_scored:
                  the answer's action type and its target equal one of the
                  question's accepted answers.
Defect judgment   accuracy, defect recall and specificity over all 1,858 items.

An item without an answer, or with a malformed one, counts as wrong.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(part: int, whole: int) -> float:
    """A share as a percentage with two decimals, rounded as the paper's table rounds it."""
    return float(f"{part / whole * 100:.2f}") if whole else 0.0


# ---------------------------------------------------------------- test intention

def covered(votes: list[list[dict]]) -> bool:
    """Two judges say reach "yes" for the same check. `votes` holds one list of
    five judgments per judge."""
    yes = Counter(judgment["check_index"] for judgments in votes for judgment in judgments
                  if judgment["reach"] == "yes")
    return any(count >= 2 for count in yes.values())


def score_test_intention(items: list[dict], answers: dict, judgments: dict) -> dict:
    """`answers` maps an id to its run.py record, `judgments` maps (id, judge) to its judge.py record."""
    judges = {judge for _, judge in judgments}
    hits = answered = incomplete = 0
    for item in items:
        if not (answers.get(item["id"]) or {}).get("parsed"):
            continue
        answered += 1
        votes = [record["parsed"]["judgments"] for (item_id, _), record in judgments.items()
                 if item_id == item["id"] and record.get("parsed")]
        incomplete += len(votes) < 3
        hits += covered(votes)
    return {"pool": "test_intention", "items": len(items), "answered": answered, "judges": sorted(judges),
            "answered_without_three_judgments": incomplete, "covered": hits,
            "coverage": pct(hits, len(items))}


# ---------------------------------------------------------------- task execution

VERB_ALIASES = {"tap": "click", "long_tap": "long_press", "swipe": "drag",
                "input_text": "type", "text_input": "type"}
BOUNDS_RE = re.compile(r"\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*")


def _verb(action: dict) -> str:
    verb = action.get("type") or action.get("action")
    return VERB_ALIASES.get(verb, verb or "")


def _named_target(action: dict) -> str:
    return str(action.get("target_description") or action.get("control_id") or action.get("target") or "").strip()


def _key(value) -> str:
    key = str(value or "").strip().lower()
    if key.startswith("keycode_"):
        key = key[len("keycode_"):]
    return {"escape": "back", "esc": "back", "recent": "recent_apps", "recents": "recent_apps"}.get(key, key)


def _area(bounds: list) -> int:
    return max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])


def _contains(outer: list, inner: list) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _iou(first, second) -> float:
    if not (isinstance(first, list) and isinstance(second, list) and len(first) == len(second) == 4):
        return 0.0
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union else 0.0


def _target_matches(controls: list[dict], answer: dict, action: dict) -> bool:
    """The action names the answer's control, or, when the answer has no control
    id, a control or explicit bounds that agree with the recorded target."""
    named = _named_target(action)
    control_id = answer.get("control_id") or ""
    if control_id:
        return named == control_id
    target = answer.get("target") or {}
    if not target:
        return True
    candidates = [control for control in controls
                  if named in {str(control.get(field) or "") for field in ("control_id", "resource_id", "text", "content_desc")}]
    bounds = BOUNDS_RE.fullmatch(named)
    if bounds:
        candidates.append({"bounds": [int(value) for value in bounds.groups()]})
    for control in candidates:
        if any(target.get(field) and control.get(field) == target[field] for field in ("resource_id", "text", "content_desc")):
            return True
        if _iou(control.get("bounds"), target.get("bounds")) >= 0.5:
            return True
        point, box = target.get("coordinates"), control.get("bounds")
        if (isinstance(point, list) and len(point) == 2 and isinstance(box, list) and len(box) == 4
                and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
            return True
    return False


def _same_region(controls: list[dict], answer: dict, named: str) -> bool:
    """A click on a label and a click on the clickable container that holds it
    are the same click, when that container is the smallest one holding it."""
    by_id = {str(control.get("control_id")): control for control in controls}
    gold, predicted = by_id.get(answer.get("control_id") or ""), by_id.get(named)
    if not gold or not predicted or not gold.get("bounds") or not predicted.get("bounds"):
        return False
    for inner, outer in ((gold, predicted), (predicted, gold)):
        if not inner.get("clickable") and outer.get("clickable") and _contains(outer["bounds"], inner["bounds"]):
            holders = [control for control in by_id.values()
                       if control.get("clickable") and control.get("bounds") and _contains(control["bounds"], inner["bounds"])]
            return bool(holders) and min(holders, key=lambda control: _area(control["bounds"])) is outer
    return False


def exact_match(controls: list[dict], answer: dict, action) -> bool:
    """One predicted action against one accepted answer: same action type and same target."""
    if not isinstance(action, dict):
        return False
    verb = _verb(action)
    if verb != answer["action"]:
        return False
    if verb == "type":
        return action.get("text", "") == (answer.get("text") or "") and _target_matches(controls, answer, action)
    if verb == "system_button":
        return _key(action.get("key")) == _key(answer.get("key"))
    if verb == "drag":
        return str(action.get("direction") or "").lower() == str(answer.get("direction") or "").lower()
    if verb == "open":
        return bool(answer.get("app")) and str(action.get("target_description") or "").strip() == str(answer["app"]).strip()
    if _target_matches(controls, answer, action):
        return True
    return verb in {"click", "long_press"} and _same_region(controls, answer, _named_target(action))


def score_task_execution(items: list[dict], answers: dict) -> dict:
    scored = [item for item in items if item["exact_match_scored"]]
    hits = sum(any(exact_match(item["controls"], answer, (answers.get(item["id"]) or {}).get("parsed"))
                   for answer in item["answers"]) for item in scored)
    answered = sum(bool((answers.get(item["id"]) or {}).get("parsed")) for item in scored)
    return {"pool": "task_execution", "questions": len(scored), "answered": answered, "exact": hits,
            "exact_match": pct(hits, len(scored))}


# ---------------------------------------------------------------- defect judgment

def score_defect_judgment(items: list[dict], answers: dict) -> dict:
    tp = tn = fp = fn = answered = 0
    for item in items:
        parsed = (answers.get(item["id"]) or {}).get("parsed")
        said = parsed.get("label") if isinstance(parsed, dict) else None
        answered += said in ("defect", "clean")
        if item["label"] == "defect":
            tp += said == "defect"
            fn += said != "defect"
        else:
            tn += said == "clean"
            fp += said != "clean"
    return {"pool": "defect_judgment", "items": len(items), "answered": answered,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "accuracy": pct(tp + tn, len(items)), "recall": pct(tp, tp + fn), "specificity": pct(tn, tn + fp)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", choices=("test_intention", "task_execution", "defect_judgment"))
    parser.add_argument("--data", type=Path, required=True, help="the downloaded dataset directory")
    parser.add_argument("--answers", type=Path, required=True, help="run.py output")
    parser.add_argument("--judgments", type=Path, help="judge.py output, for test_intention")
    args = parser.parse_args()
    items = read_jsonl(args.data / args.pool / "items.jsonl")
    answers = {row["id"]: row for row in read_jsonl(args.answers)}
    if args.pool == "test_intention":
        if not args.judgments:
            parser.error("test_intention needs --judgments")
        judgments = {(row["id"], row["judge"]): row for row in read_jsonl(args.judgments)}
        judges = sorted({judge for _, judge in judgments})
        if len(judges) != 3:
            parser.error(f"coverage needs the verdicts of three judges, and {args.judgments} has {judges}")
        result = score_test_intention(items, answers, judgments)
        if result["answered_without_three_judgments"]:
            print(f"{result['answered_without_three_judgments']} answered items lack a verdict from one of the "
                  f"judges. Rerun judge.py", file=sys.stderr)
    elif args.pool == "task_execution":
        result = score_task_execution(items, answers)
    else:
        result = score_defect_judgment(items, answers)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
