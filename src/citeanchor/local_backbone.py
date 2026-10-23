"""Local vLLM backend for `citeanchor.llm`: budgeted thinking, then a schema-constrained answer."""
from __future__ import annotations

import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import jinja2
import requests

BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8031/v1").rstrip("/")
BASES = [BASE]
_rr = [0]
MODEL = os.environ.get("OPENAI_MODEL", "openresearcher-30b-a3b")
MODEL_DIR = os.path.expanduser(os.environ.get("OS_MODEL_DIR", "~/models/OpenResearcher-30B-A3B"))
THINK = 2048       # thinking tokens per call
ANSWER = 4096      # answer tokens per call
SYSTEM = ("You have no tools and no internet access. Do not call tools, do not write code, do not write "
          "search queries. Use only the information in the user's message and your own knowledge. Think "
          "briefly, then give the final answer in exactly the format the user asks for: when a JSON object "
          "is requested, output only that JSON object (no 'Explanation:' or 'Exact Answer:' lines).")
_KEYS = ("verdict", "exists", "type", "queries", "match", "same_work", "authors", "title")

_STR = {"type": "string"}
_REASON = {"type": "string", "maxLength": 600}   # bounds free text under constrained decoding
SCHEMAS = [   # (marker that identifies the prompt, JSON schema of its answer)
    ("offending_fields", {"type": "object", "additionalProperties": False, "required": ["verdict", "offending_fields", "reason"],
                          "properties": {"verdict": {"enum": ["error", "consistent", "different_work"]},
                                         "offending_fields": {"type": "array", "items": _STR}, "reason": _REASON}}),
    ('"verdict": "correct" or "erroneous"',
     {"type": "object", "additionalProperties": False, "required": ["verdict", "source", "reason"],
      "properties": {"verdict": {"enum": ["correct", "erroneous"]},
                     "source": {"anyOf": [{"type": "null"},
                                          {"type": "object", "additionalProperties": False,
                                           "required": ["title", "authors", "venue", "year"],
                                           "properties": {"title": _STR, "authors": {"type": "array", "items": _STR},
                                                          "venue": _STR, "year": {"anyOf": [{"type": "integer"}, _STR, {"type": "null"}]}}}]},
                     "reason": _REASON}}),
    ('"exists": true or false',
     {"type": "object", "additionalProperties": False, "required": ["exists", "confidence", "reason"],
      "properties": {"exists": {"type": "boolean"}, "confidence": {"type": "number"}, "reason": _REASON}}),
    ('"match": true|false',
     {"type": "object", "additionalProperties": False, "required": ["match", "matched_result", "note"],
      "properties": {"match": {"type": "boolean"}, "matched_result": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                     "note": {"type": "string", "maxLength": 400}}}),
    ("Extract structured metadata from this academic citation",
     {"type": "object", "additionalProperties": False, "required": ["authors", "title", "venue", "year"],
      "properties": {"authors": _STR, "title": _STR, "venue": _STR,
                     "year": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}}}),
    ('"type": "paper|book|report|web|dataset|repo|unknown"',
     {"type": "object", "additionalProperties": False, "required": ["type", "reason"],
      "properties": {"type": {"enum": ["paper", "book", "report", "web", "dataset", "repo", "unknown"]}, "reason": _REASON}}),
]


def _schema_for(prompt: str):
    for marker, schema in SCHEMAS:
        if marker in prompt:
            return schema
    return None

STATS = {"calls": 0, "structured": 0, "think_closed_by_model": 0, "think_forced": 0, "tool_call_attempts": 0, "json_rescued": 0,
         "fields_normalised": 0, "json_retries": 0, "placeholder_answers": 0, "placeholder_to_regex": 0,
         "no_json": 0, "empty_answer": 0, "unk_answer": 0,
         "errors": 0, "think_tokens": 0, "answer_tokens": 0, "prompt_tokens": 0, "seconds": 0.0}
_lock = threading.Lock()


def _template():
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)

    def _raise(msg):
        raise jinja2.TemplateError(msg)

    env.globals["raise_exception"] = _raise
    return env.from_string(open(os.path.join(MODEL_DIR, "chat_template.jinja"), encoding="utf-8").read())


_T = _template()
_CFG = json.load(open(os.path.join(MODEL_DIR, "tokenizer_config.json"), encoding="utf-8"))


def render(prompt: str) -> str:
    """The prompt in the model's chat template with thinking enabled (ends in '<think>\\n')."""
    msgs = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": prompt}]
    return _T.render(messages=msgs, add_generation_prompt=True, enable_thinking=True,
                     bos_token=_CFG.get("bos_token") or "", eos_token=_CFG.get("eos_token") or "")


def _expected_key(prompt: str):
    for k in _KEYS:
        if f'"{k}"' in prompt:
            return k
    return None


def _harness_parse(text: str):
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(text[s:e])
    except json.JSONDecodeError:
        return None


def _rescue(answer: str, key: str):
    """First JSON object in the answer that carries `key`, or None."""
    dec, i = json.JSONDecoder(), 0
    while True:
        j = answer.find("{", i)
        if j < 0:
            return None
        try:
            obj, _ = dec.raw_decode(answer[j:])
            if isinstance(obj, dict) and key in obj:
                return obj
        except json.JSONDecodeError:
            pass
        i = j + 1


def _complete(prompt: str, max_tokens: int, stop: list[str], base: str | None = None, schema: dict | None = None) -> dict:
    last = None
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stop": stop, "skip_special_tokens": False}
    if schema is not None:
        body["structured_outputs"] = {"json": schema}
    deadline = time.time() + 60 * 20          # wait up to 20 min for a server that is restarting
    attempt = 0
    while True:
        try:
            r = requests.post(f"{base or BASE}/completions", timeout=900, json=body)
            r.raise_for_status()
            return r.json()
        except Exception as exc:          # server busy or restarting: back off and retry
            last = exc
            attempt += 1
            if time.time() > deadline:
                break
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"local backbone unreachable after {attempt} attempts: {last}")


def chat_text(prompt: str) -> tuple[str, dict]:
    """(answer, info) for one user prompt, with budgeted thinking."""
    t0 = time.time()
    raw = render(prompt)
    with _lock:
        base = BASES[_rr[0] % len(BASES)]; _rr[0] += 1      # all phases of one call on one server (prefix cache)
    r1 = _complete(raw, THINK, ["</think>", "<tool_call>"], base)
    c1 = r1["choices"][0]
    thought = c1.get("text") or ""
    tool1 = c1.get("stop_reason") == "<tool_call>"
    closed = c1.get("finish_reason") == "stop" and not tool1
    schema = _schema_for(prompt)
    r2 = _complete(raw + thought + "\n</think>\n", ANSWER, ["<|im_end|>", "<tool_call>"], base, schema)
    c2 = r2["choices"][0]
    tool2 = c2.get("stop_reason") == "<tool_call>"
    answer = (c2.get("text") or "").strip()
    key, rescued, no_json = _expected_key(prompt), False, False
    if key:
        got = _harness_parse(answer)
        if not (isinstance(got, dict) and key in got):
            obj = _rescue(answer, key)
            if obj is None:
                # one retry: close the thought with an explicit reminder and start the JSON object
                r3 = _complete(raw + thought + "\nI must now output only the requested JSON object.\n</think>\n{",
                               ANSWER, ["<|im_end|>", "<tool_call>"], base)
                retry = "{" + (r3["choices"][0].get("text") or "")
                got3 = _harness_parse(retry)
                obj = got3 if isinstance(got3, dict) and key in got3 else _rescue(retry, key)
                with _lock:
                    STATS["json_retries"] += 1
            if obj is not None:
                answer, rescued = json.dumps(obj, ensure_ascii=False), True
            else:
                no_json = True
    normalised = False
    if "Extract structured metadata from this academic citation" in prompt:
        # retry an answer that copies the prompt's placeholder template; "{}" if every retry fails
        def _placeholder(o):
            t = str((o or {}).get("title", "")).strip().strip(".…").strip()
            return not t or t.lower() in ("n/a", "paper/work title", "title")
        obj0 = _harness_parse(answer)
        if not isinstance(obj0, dict) or _placeholder(obj0):
            with _lock:
                STATS["placeholder_answers"] += 1
            fixed = None
            for k in range(3):
                r4 = _complete(raw + thought + "\nI must now output only the requested JSON object, with the actual "
                               "title, authors, venue and year copied from the citation text (never a placeholder)."
                               "\n</think>\n", ANSWER, ["<|im_end|>", "<tool_call>"], base, _schema_for(prompt))
                obj4 = _harness_parse(r4["choices"][0].get("text") or "")
                if isinstance(obj4, dict) and not _placeholder(obj4):
                    fixed = obj4; break
            if fixed is not None:
                answer = json.dumps(fixed, ensure_ascii=False)
            else:
                answer = "{}"
                with _lock:
                    STATS["placeholder_to_regex"] += 1
        # join list-valued fields into a comma-separated string
        obj = _harness_parse(answer)
        if isinstance(obj, dict):
            for k in ("authors", "title", "venue", "year"):
                if isinstance(obj.get(k), list):
                    obj[k] = ", ".join(str(x) for x in obj[k]); normalised = True
            if normalised:
                answer = json.dumps(obj, ensure_ascii=False)
    info = {"prompt_tokens": r1["usage"]["prompt_tokens"], "think_tokens": r1["usage"]["completion_tokens"],
            "answer_tokens": r2["usage"]["completion_tokens"], "closed_by_model": closed,
            "finish_reason": c2.get("finish_reason"), "seconds": time.time() - t0}
    with _lock:
        STATS["calls"] += 1
        STATS["structured"] += int(schema is not None)
        STATS["think_closed_by_model" if closed else "think_forced"] += 1
        STATS["tool_call_attempts"] += int(tool1) + int(tool2)
        STATS["json_rescued"] += int(rescued)
        STATS["no_json"] += int(no_json)
        STATS["fields_normalised"] += int(normalised)
        STATS["empty_answer"] += int(not answer)
        STATS["unk_answer"] += int("<unk>" in answer)
        STATS["think_tokens"] += info["think_tokens"]
        STATS["answer_tokens"] += info["answer_tokens"]
        STATS["prompt_tokens"] += info["prompt_tokens"]
        STATS["seconds"] += info["seconds"]
    return answer, info


def _as_openai_response(answer: str, info: dict):
    """Minimal stand-in for an OpenAI ChatCompletion response."""
    usage = SimpleNamespace(prompt_tokens=info["prompt_tokens"],
                            completion_tokens=info["think_tokens"] + info["answer_tokens"],
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=info["think_tokens"]))
    msg = SimpleNamespace(content=answer, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=info["finish_reason"], index=0)],
                           usage=usage, model=MODEL)


def install() -> None:
    """Route `citeanchor.llm` chat calls to the local server."""
    from . import llm as v

    def _local_chat(client, prompt, max_retries=3, reasoning_effort=None):
        try:
            answer, info = chat_text(prompt)
        except Exception:
            with _lock:
                STATS["errors"] += 1
            raise
        resp = _as_openai_response(answer, info)
        v._record_usage(resp)
        return resp

    v._openai_chat = _local_chat
    v.OPENAI_MODEL = MODEL


if __name__ == "__main__":          # python -m citeanchor.local_backbone <script> [args...]
    install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")
