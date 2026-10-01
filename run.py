#!/usr/bin/env python3
"""Ask a model every item of one GTArena pool through an OpenAI-compatible API.

    python run.py test_intention  --data DIR --model MODEL --base-url URL --api-key-env VAR --out FILE
    python run.py task_execution  ...
    python run.py defect_judgment ...

Each item is one POST {base-url}/chat/completions carrying a single user
message: the prompt, then the item's images as base64 data URIs in their
released order. In defect judgment each image is preceded by its label from
`image_labels`. Test intention takes `--no-screenshot` for the arm without the
image. Temperature is 0. An endpoint that rejects the parameter is asked again
without it.

An empty reply, or one that does not validate, is asked again, up to three
attempts. Each item gives one JSON line in --out: its `id`, the `raw` reply the
answer was read from, the `parsed` answer or null, and the `attempts`. A rerun
skips the ids already answered in --out, so the same command resumes.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import functools
import json
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

POOLS = ("test_intention", "task_execution", "defect_judgment")
ATTEMPTS = 3
BACKOFF = (2, 8, 30)
TIMEOUT = 600
WITH_SCREENSHOT = "One decision-point screenshot is attached."
WITHOUT_SCREENSHOT = "No screenshot is supplied."


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- reading a reply

def parse_json(raw: str):
    """The JSON value in a reply: the whole reply, a <structured_output> block,
    a fenced block, or the span from the first bracket to the last."""
    value = (raw or "").strip()
    try:
        return json.loads(value)
    except ValueError:
        pass
    for pattern in (r"<structured_output>\s*(.*?)\s*</structured_output>", r"```(?:json)?\s*(.*?)\s*```"):
        match = re.search(pattern, value, re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except ValueError:
                pass
    match = re.search(r"[\[{].*[\]}]", value, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            pass
    return None


def validate_checks(value) -> dict:
    """Test intention: exactly five checks, each with exactly start_from, action and expected."""
    if isinstance(value, list):
        value = {"checks": value}
    checks = value.get("checks") if isinstance(value, dict) else None
    if not isinstance(checks, list) or len(checks) != 5:
        raise ValueError("checks_not_exactly_five")
    fields = {"start_from", "action", "expected"}
    if any(not isinstance(check, dict) or set(check) != fields
           or any(not str(check[key]).strip() for key in fields) for check in checks):
        raise ValueError("invalid_check_schema")
    return {"checks": checks}


def validate_judgments(value) -> dict:
    """The judge step: five judgments, check_index 1 to 5 in order, each reach yes or no."""
    if isinstance(value, dict):
        value = value.get("judgments")
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("judgment_count_not_five")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError("judgment_not_object")
    if [row.get("check_index") for row in value] != [1, 2, 3, 4, 5]:
        raise ValueError("judgment_indices_not_ordered_1_to_5")
    if not all(row.get("reach") in {"yes", "no"} for row in value):
        raise ValueError("judgment_reach_invalid")
    return {"judgments": [
        {"check_index": int(row["check_index"]), "reach": row["reach"],
         "evidence": row.get("evidence") if isinstance(row.get("evidence"), str) else None}
        for row in value]}


ALIASES = {
    "tap": "click", "long_tap": "long_press",
    "input_text": "type", "text_input": "type",
    "swipe": "drag", "key_back": "system_button",
    "system_back": "system_button", "home": "system_button",
    "key_enter": "system_button", "task_complete": "terminate",
}
ACTION_TYPES = {
    "click", "double_click", "long_press", "type", "open",
    "drag", "system_button", "wait", "terminate", "action_sequence",
}


def validate_action(value) -> dict:
    """Task execution: one action from the vocabulary, or an ordered sequence of them."""
    if not isinstance(value, dict):
        raise ValueError("not_object")
    # A sequence payload that omits its discriminator is still a sequence.
    if "type" not in value and isinstance(value.get("actions"), list) and value["actions"]:
        value = {**value, "type": "action_sequence"}
    if "type" not in value and value.get("action") in ACTION_TYPES:
        value["type"] = value.pop("action")
    if value.get("type") in ALIASES:
        value["type"] = ALIASES[value["type"]]
    if value.get("type") not in ACTION_TYPES:
        raise ValueError("invalid_type")
    if not isinstance(value.get("target_description", ""), str):
        raise ValueError("invalid_target_description")
    if value["type"] == "action_sequence":
        actions = value.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("invalid_sequence")
        for action in actions:
            validate_action(action)
            if action.get("type") == "action_sequence":
                raise ValueError("invalid_sequence_action")
    return value


VERBS = ["double_click", "long_press", "system_button", "action_sequence",
         "terminate", "ask_user", "click", "type", "open", "drag", "wait"]
VERB_RE = re.compile(r"\b(" + "|".join(VERBS) + r")\b")
CONTROL_RE = re.compile(r"\bc\d{2,4}\b")
NEEDS_CONTROL = {"click", "double_click", "long_press", "type", "drag"}
# `type` written as the name of a field: `type: click`, `"type": "click"`, `<type>click</type>`
TYPE_FIELD_RE = re.compile(r"""\btype\b["'`]?(?=\s*[=:])|</?type>""")


def _field(raw: str, name: str) -> str | None:
    """A `name=value` or `name: value` field, quoted or to the end of the line."""
    match = re.search(rf'\b{name}\s*[=:]\s*"([^"]*)"', raw)
    if match:
        return match.group(1)
    match = re.search(rf'\b{name}\s*[=:]\s*([^\n,|]+)', raw)
    return match.group(1).strip().strip('"').strip() if match else None


def extract_action(raw: str) -> dict | None:
    """Task execution, for a final reply that is not JSON: the action counts only
    when the reply names exactly one verb and, for a verb that needs a control,
    exactly one control id. `type` written as the name of the field that holds
    the verb names no action and is not counted."""
    verbs = set(VERB_RE.findall(TYPE_FIELD_RE.sub(" ", raw or "")))
    if len(verbs) != 1:
        return None
    verb = verbs.pop()
    action = {"type": verb}
    if verb in NEEDS_CONTROL:
        ids = set(CONTROL_RE.findall(raw))
        if len(ids) != 1:
            return None
        action["target_description"] = ids.pop()
    for name in ("text", "key", "direction"):
        value = _field(raw, name)
        if value is not None:
            action[name] = value
    return action


def label_from_json(raw: str) -> dict:
    """Defect judgment: a JSON object with exactly `label` and `observed_evidence`,
    or those two fields as two plain lines."""
    stripped = (raw or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    payload = None
    if match:
        try:
            candidate = json.loads(match.group(0))
            if isinstance(candidate, dict):
                payload = candidate
        except ValueError:
            pass
    if payload is None:
        plain = re.fullmatch(
            r'\s*(?:[-*]\s*)?`?label`?\s*:\s*["\']?(defect|clean)["\']?'
            r'\s*\n(?:[-*]\s*)?`?observed_evidence`?\s*:\s*([^\r\n]+?)\s*',
            stripped,
        )
        if not plain:
            raise ValueError("response does not contain a JSON object or exact two-field mapping")
        evidence = plain.group(2).strip()
        if (evidence.startswith('"') and evidence.endswith('"')) or (evidence.startswith("'") and evidence.endswith("'")):
            evidence = evidence[1:-1].strip()
        payload = {"label": plain.group(1), "observed_evidence": evidence}
    if set(payload) != {"label", "observed_evidence"}:
        raise ValueError(f"unexpected response keys: {sorted(payload)}")
    if payload["label"] not in {"defect", "clean"}:
        raise ValueError(f"invalid label: {payload['label']!r}")
    evidence = payload["observed_evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("observed_evidence must be a non-empty string")
    return {"label": payload["label"], "observed_evidence": evidence.strip()}


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_UNCLOSED = re.compile(r"^.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)
_OBJECT = re.compile(r"\{[^{}]*\"label\"\s*:\s*\"(defect|clean)\"[^{}]*\}", re.S | re.I)
_LINE = re.compile(r"^\s*(?:[-*+]\s+)?[\"'`*]*label[\"'`*]*\s*[:=]\s*[\"'`*]*[ \t]*[\"'`*]*(defect|clean)\b",
                   re.I | re.M)


def _labels_in(text: str) -> list[str]:
    """Every label the text asserts in a fenced object, a bare object or a
    `label:` line, in reading order. Overlapping spans count once."""
    hits = []
    for match in _FENCE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except Exception:  # noqa: BLE001 - a fence that is not JSON is not an answer
            continue
        if isinstance(payload, dict) and str(payload.get("label", "")).lower() in ("defect", "clean"):
            hits.append((match.start(), match.end(), str(payload["label"]).lower()))
    for pattern in (_OBJECT, _LINE):
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), match.group(1).lower()))
    found, covered_to = [], -1
    for start, end, label in sorted(hits):
        if start >= covered_to:
            found.append(label)
            covered_to = end
    return found


def label_from_fields(raw: str) -> dict | None:
    """Defect judgment, for a final reply without the JSON object: the label it
    states after any reasoning block, provided it states only one."""
    stripped = _THINK.sub("", raw or "")
    if "</think>" in stripped.lower():
        stripped = _UNCLOSED.sub("", stripped, count=1)
    found = _labels_in(stripped) if stripped.strip() else _labels_in(raw or "")
    if not found and stripped.strip():
        found = _labels_in(raw or "")
    return {"label": found[0]} if found and len(set(found)) == 1 else None


def parse_reply(pool: str, raw: str) -> dict:
    """The answer a reply gives, or ValueError saying why it gives none.
    `pool` is one of POOLS, or "judge" for the test-intention judges."""
    if pool == "test_intention":
        return validate_checks(parse_json(raw))
    if pool == "judge":
        return validate_judgments(parse_json(raw))
    if pool == "task_execution":
        return validate_action(parse_json(raw))
    return label_from_json(raw)


def fallback_answer(pool: str, raws: list[str]) -> dict | None:
    """What the last non-empty reply still says when no attempt validated."""
    last = next((raw for raw in reversed(raws) if (raw or "").strip()), "")
    if pool == "task_execution":
        return extract_action(last)
    if pool == "defect_judgment":
        return label_from_fields(last)
    return None


# ---------------------------------------------------------------- what is sent

def test_intention_prompt(template: str, item: dict, screenshot: bool) -> str:
    return template.format(app_name=item["app"],
                           screenshot_availability=WITH_SCREENSHOT if screenshot else WITHOUT_SCREENSHOT)


def image_part(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"}}


def message_content(pool: str, item: dict, pool_dir: Path, template: str = "",
                    screenshot: bool = True) -> list[dict]:
    """The user message for one item: the prompt, then its images in order."""
    if pool == "test_intention":
        content = [{"type": "text", "text": test_intention_prompt(template, item, screenshot)}]
        if screenshot:
            content.append(image_part(pool_dir / item["image"]))
        return content
    content = [{"type": "text", "text": item["prompt"]}]
    if pool == "task_execution":
        content.append(image_part(pool_dir / item["image"]))
        return content
    for label, image in zip(item["image_labels"], item["images"]):
        content.append({"type": "text", "text": label})
        content.append(image_part(pool_dir / image))
    return content


def request_body(model: str, content, max_tokens: int) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_completion_tokens": max_tokens}


def call(base_url: str, api_key: str | None, body: dict) -> str:
    """One chat completion: the reply text, or an exception."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def post(payload: dict) -> dict:
        request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                         data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read())

    try:
        payload = post(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        if exc.code == 400 and "temperature" in detail.lower() and "temperature" in body:
            payload = post({key: value for key, value in body.items() if key != "temperature"})
        else:
            raise RuntimeError(f"http_{exc.code}: {detail[:600]}") from None
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not text:
        raise RuntimeError(f"empty reply: {json.dumps(payload)[:300]}")
    return text


def ask(pool: str, base_url: str, api_key: str | None, body: dict) -> dict:
    """Up to three attempts. The answer comes from the first reply that
    validates, else from what the last reply still says."""
    attempts, parsed, raw = [], None, ""
    for attempt in range(1, ATTEMPTS + 1):
        raw, error = "", None
        try:
            raw = call(base_url, api_key, body)
            parsed = parse_reply(pool, raw)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and retried
            error = f"{type(exc).__name__}: {exc}"[:1200]
        attempts.append({"raw": raw, "error": error})
        if error is None:
            break
        if attempt < ATTEMPTS:
            time.sleep(BACKOFF[attempt - 1])
    if parsed is None:
        raws = [row["raw"] for row in attempts]
        parsed = fallback_answer(pool, raws)
        raw = next((text for text in reversed(raws) if text.strip()), "")
    return {"raw": raw, "parsed": parsed, "attempts": attempts}


def run_jobs(jobs: list[tuple[dict, Callable[[], dict]]], pool: str, base_url: str, api_key: str | None,
             out: Path, workers: int) -> None:
    """Ask every (fields, make_body) job and append `fields` plus the result to `out` as each finishes."""
    lock = threading.Lock()
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(job: tuple[dict, Callable[[], dict]]) -> None:
        fields, make_body = job
        record = {**fields, **ask(pool, base_url, api_key, make_body())}
        with lock, out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for done, _ in enumerate(executor.map(one, jobs), 1):
            if done % 50 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)


def api_key_from(name: str | None) -> str | None:
    if not name:
        return None
    key = os.environ.get(name)
    if not key:
        raise SystemExit(f"environment variable {name} is not set")
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", choices=POOLS)
    parser.add_argument("--data", type=Path, required=True, help="the downloaded dataset directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True, help="the API root that /chat/completions is appended to")
    parser.add_argument("--api-key-env", help="name of the environment variable that holds the API key")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-screenshot", action="store_true", help="test intention without the screenshot")
    parser.add_argument("--max-completion-tokens", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.no_screenshot and args.pool != "test_intention":
        parser.error("--no-screenshot applies to test_intention only")

    api_key = api_key_from(args.api_key_env)
    pool_dir = args.data / args.pool
    template = (args.data / "prompts/test_intention.txt").read_text(encoding="utf-8")

    def body(item: dict) -> dict:
        content = message_content(args.pool, item, pool_dir, template, not args.no_screenshot)
        return request_body(args.model, content, args.max_completion_tokens)

    done = {row["id"] for row in read_jsonl(args.out) if row.get("parsed") is not None}
    jobs = [({"id": item["id"]}, functools.partial(body, item))
            for item in read_jsonl(pool_dir / "items.jsonl") if item["id"] not in done]
    print(f"{args.pool}: {len(done)} answered already, {len(jobs)} to ask", flush=True)
    run_jobs(jobs, args.pool, args.base_url, api_key, args.out, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
