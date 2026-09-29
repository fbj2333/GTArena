#!/usr/bin/env python3
"""Grade test-intention answers with three text-only judges.

    python judge.py --data DIR --answers ANSWERS --out JUDGMENTS --base-url URL --api-key-env VAR
    python judge.py --data DIR --answers ANSWERS --out JUDGMENTS \\
        --judge gpt-5.5 URL VAR --judge gemini-3.1-pro URL VAR --judge deepseek-v4-pro URL VAR

ANSWERS is the run.py output for test_intention. Each answered item goes to
each judge as prompts/test_intention_judge.txt, filled with the item's defect
and the model's five checks numbered from 1. The judges see no image. For every
check a judge answers reach "yes" or "no": would running the check expose the
defect. The first form serves the three default judges from one endpoint.
The second gives each judge its own. Each (item, judge) pair gives one JSON line in
--out: `id`, `judge`, `raw`, `parsed` and `attempts`. A rerun skips the pairs
already judged.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

from run import api_key_from, read_jsonl, request_body, run_jobs

DEFAULT_JUDGES = ("gpt-5.5", "gemini-3.1-pro", "deepseek-v4-pro")


def judge_prompt(template: str, item: dict, checks: list[dict]) -> str:
    """The judge prompt for one answered item. Each check is shown with its
    fields in the order the generation prompt defines them."""
    numbered = [{"check_index": index, "start_from": check["start_from"], "action": check["action"],
                 "expected": check["expected"]} for index, check in enumerate(checks, 1)]
    defect = item["defect"]
    return template.format(
        app_name=item["app"],
        target_expected=defect["expected"] or "(empty)",
        target_actual=defect["actual"] or "(empty)",
        target_symptom=defect["symptom"] or "(empty)",
        checks_json=json.dumps(numbered, ensure_ascii=False, indent=2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="the downloaded dataset directory")
    parser.add_argument("--answers", type=Path, required=True, help="run.py output for test_intention")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", help="one endpoint serving the three default judges")
    parser.add_argument("--api-key-env", help="name of the environment variable that holds its API key")
    parser.add_argument("--judge", nargs=3, action="append", metavar=("MODEL", "BASE_URL", "API_KEY_ENV"),
                        help="one judge and its endpoint, given three times")
    parser.add_argument("--max-completion-tokens", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.judge:
        judges = [tuple(judge) for judge in args.judge]
    elif args.base_url:
        judges = [(model, args.base_url, args.api_key_env) for model in DEFAULT_JUDGES]
    else:
        parser.error("give --base-url, or --judge three times")
    if len({model for model, _, _ in judges}) != 3:
        parser.error("coverage is decided by three different judges")

    items = {item["id"]: item for item in read_jsonl(args.data / "test_intention/items.jsonl")}
    template = (args.data / "prompts/test_intention_judge.txt").read_text(encoding="utf-8")
    latest = {row["id"]: row for row in read_jsonl(args.answers)}
    checks = {item_id: row["parsed"]["checks"] for item_id, row in latest.items() if row.get("parsed")}
    done = {(row["id"], row["judge"]) for row in read_jsonl(args.out) if row.get("parsed") is not None}

    def body(model: str, item_id: str) -> dict:
        return request_body(model, judge_prompt(template, items[item_id], checks[item_id]),
                            args.max_completion_tokens)

    for model, base_url, key_env in judges:
        jobs = [({"id": item_id, "judge": model}, functools.partial(body, model, item_id))
                for item_id in sorted(checks) if (item_id, model) not in done]
        print(f"{model}: {len(jobs)} items to judge", flush=True)
        run_jobs(jobs, "judge", base_url, api_key_from(key_env), args.out, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
