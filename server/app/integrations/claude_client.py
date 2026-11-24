from __future__ import annotations

import json
import os
import random
import re
from functools import lru_cache
from itertools import cycle
from uuid import uuid4

import anthropic
from pydantic import BaseModel, ValidationError

from ..schemas import (
    FrontendDifficulty,
    FrontendLanguage,
    FrontendSystemName,
    FrontendTask,
    FrontendVulnType,
)
from .hacktron import NOT_CONFIGURED_MESSAGE

# Matches docker-compose and the README; override with ANTHROPIC_MODEL.
DEFAULT_MODEL = "claude-sonnet-5"

# Current models think before answering, and thinking tokens count against
# max_tokens, so a small cap can be spent before any text is written. Billing
# is for tokens used, not the cap, so the cap is generous.
MAX_TOKENS = 16000

# Thinking makes calls slower than the old 30s budget allowed.
REQUEST_TIMEOUT_SECONDS = 120

# Retries on top of the first attempt for transient failures (429, 5xx,
# connection errors). Worst-case wall time is the timeout times (retries + 1).
MAX_RETRIES = 3

# Log lines that describe the scanner's own state rather than the code. They
# are produced by hacktron.scan_with_hacktron, main.audit_tasks and
# store.finish_session.
SCANNER_STATUS_PREFIXES = (NOT_CONFIGURED_MESSAGE, "Hacktron scan failed:", "Audit failed:")

SYSTEM_NAMES: list[FrontendSystemName] = [
    "O2",
    "NAVIGATION",
    "SHIELDS",
    "REACTOR",
    "COMMUNICATIONS",
    "ELECTRICAL",
    "MEDBAY",
    "SECURITY",
    "WEAPONS",
    "ADMIN",
]


class ClaudeTaskItem(BaseModel):
    code: str
    isVulnerable: bool
    vulnerabilityType: FrontendVulnType
    systemName: FrontendSystemName | None = None
    vulnerabilityLine: int | None = None
    hints: list[str] | None = None


class ClaudeTaskPayload(BaseModel):
    tasks: list[ClaudeTaskItem]


def generate_frontend_tasks(
    language: FrontendLanguage,
    difficulty: FrontendDifficulty,
    complexity_level: str,
    count: int,
    vuln_density: float,
) -> list[FrontendTask]:
    payload = _request_tasks_from_claude(
        language=language,
        difficulty=difficulty,
        complexity_level=complexity_level,
        count=count,
        vuln_density=vuln_density,
    )

    validated = _validate_tasks_payload(payload, count)
    tasks: list[FrontendTask] = []
    system_cycle = cycle(SYSTEM_NAMES)

    for task in validated.tasks:
        system_name = task.systemName or next(system_cycle)
        tasks.append(
            FrontendTask(
                id=uuid4().hex,
                systemName=system_name,
                code=task.code,
                language=language,
                isVulnerable=task.isVulnerable,
                vulnerabilityType=_normalize_vuln_type(task.isVulnerable, task.vulnerabilityType),
                vulnerabilityLine=task.vulnerabilityLine,
                hints=task.hints,
                status="pending",
            )
        )

    return tasks


def generate_security_mentor_summary(
    hacktron_logs: list[str],
    failed_task_summaries: list[str],
) -> str:
    if not failed_task_summaries:
        return (
            "No vulnerabilities were missed in this run. Systems remained secure "
            "and no exploitable patterns were detected."
        )
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    system_prompt = (
        "You are the Security Mentor. Provide a 3-sentence post-mortem summary focused "
        "only on the code vulnerabilities. Do NOT mention tools, scanners, logs, or "
        "any operational failures. Sentence 1: what went wrong (vulns only). "
        "Sentence 2: how an attacker could exploit. Sentence 3: the most direct fix. "
        "Be concise and technical."
    )

    # Drop the scanner's own status messages (not configured, failed) so the
    # mentor does not talk about tooling. Real findings are kept even when they
    # happen to mention the scanner by name.
    filtered_logs = [
        log for log in hacktron_logs if not log.startswith(SCANNER_STATUS_PREFIXES)
    ]
    user_prompt = {
        "failed_tasks": failed_task_summaries,
        "hacktron_logs": filtered_logs,
    }

    response = _call_claude(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_prompt),
        max_tokens=MAX_TOKENS,
    )

    return _strip_heading_marks(response)


def _request_tasks_from_claude(
    language: FrontendLanguage,
    difficulty: FrontendDifficulty,
    complexity_level: str,
    count: int,
    vuln_density: float,
) -> ClaudeTaskPayload:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    vulnerable_target = _sample_vulnerable_target(count, vuln_density)
    system_prompt = (
        "You generate short code snippets for a security training game. "
        "Return strict JSON with a top-level 'tasks' array. "
        "Each task: code (max 10 lines), isVulnerable (boolean), "
        "vulnerabilityType (one of XSS, SQL_INJECTION, SSRF, RCE, PATH_TRAVERSAL, "
        "COMMAND_INJECTION, INSECURE_DESERIALIZATION, SAFE), "
        "systemName (one of O2, NAVIGATION, SHIELDS, REACTOR, COMMUNICATIONS, "
        "ELECTRICAL, MEDBAY, SECURITY, WEAPONS, ADMIN), vulnerabilityLine (number), "
        "and hints (array of exactly 2 short hints). "
        "No markdown, only JSON."
    )

    user_prompt = (
        f"Generate {count} {language} snippets for difficulty {difficulty} "
        f"with {complexity_level} complexity. "
        f"Ensure exactly {vulnerable_target} snippets are vulnerable."
    )

    response = _call_claude(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=MAX_TOKENS,
    )

    return _parse_json_payload(response)


@lru_cache(maxsize=4)
def _client(api_key: str) -> anthropic.Anthropic:
    # One client per key, reused across requests so its connection pool is too.
    # The SDK retries connection errors, 408, 409, 429 and 5xx with exponential
    # backoff, honouring the server's retry-after header.
    return anthropic.Anthropic(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def _call_claude(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    try:
        message = _client(api_key).messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    # These are raised only once the SDK's retries are exhausted. The timeout
    # check comes first because APITimeoutError subclasses APIConnectionError.
    except anthropic.APITimeoutError as exc:
        raise RuntimeError("Claude API request timed out") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(f"Claude API connection failed: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Claude API error: {exc.status_code}") from exc

    # A truncated or refused reply would otherwise reach the JSON parser and
    # surface as a confusing parse error.
    if message.stop_reason == "max_tokens":
        raise RuntimeError("Claude response was truncated (max_tokens reached)")
    if message.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")

    # Thinking blocks carry no answer text; only text blocks do.
    return "\n".join(block.text for block in message.content if block.type == "text").strip()

# Helper to parse and validate Claude JSON response
def _parse_json_payload(text: str) -> ClaudeTaskPayload:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = json.loads(_extract_json(text))

    if isinstance(raw, list):
        raw = {"tasks": raw}
    elif isinstance(raw, dict) and "tasks" not in raw and "snippets" in raw:
        raw = {"tasks": raw.get("snippets", [])}

    if isinstance(raw, dict) and "tasks" in raw:
        raw["tasks"] = [_normalize_task_item(item) for item in raw["tasks"]]

    try:
        return ClaudeTaskPayload.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Claude JSON validation failed: {exc}") from exc


def _validate_tasks_payload(payload: ClaudeTaskPayload, count: int) -> ClaudeTaskPayload:
    # The vulnerable count is deliberately not checked: the target is random
    # (see _sample_vulnerable_target), so a reply that misses it by one is just
    # another valid round, not a reason to throw away a paid call.
    if len(payload.tasks) != count:
        raise ValueError(f"Claude returned {len(payload.tasks)} tasks, expected {count}.")
    return payload


def _sample_vulnerable_target(
    count: int, vuln_density: float, rng: random.Random | None = None
) -> int:
    """How many of `count` snippets should be vulnerable this round.

    Each snippet is vulnerable with probability `vuln_density` (a binomial
    draw). A fixed round(count * density) let players who knew the density
    deduce the last answers by counting. At least one is always vulnerable so
    every round has something to find.
    """
    rng = rng or random
    drawn = sum(1 for _ in range(count) if rng.random() < vuln_density)
    return max(1, drawn)


def _normalize_vuln_type(is_vulnerable: bool, vuln_type: FrontendVulnType) -> FrontendVulnType:
    if not is_vulnerable:
        return "SAFE"
    if vuln_type == "SAFE":
        return "XSS"
    return vuln_type


def _extract_json(text: str) -> str:
    # Return the first complete JSON object/array embedded in prose or a code
    # fence. raw_decode is a real JSON parser, so brackets inside string values
    # (code snippets are full of them) cannot unbalance it the way counting can.
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        return text[idx:end]
    raise ValueError("Claude response did not contain valid JSON.")

#for hints and field normalization
def _normalize_task_item(item: object) -> dict:
    if not isinstance(item, dict):
        return {}

    normalized = dict(item)
    is_vulnerable = normalized.get("isVulnerable")
    if "isVulnerable" not in normalized and "vulnerable" in normalized:
        normalized["isVulnerable"] = normalized["vulnerable"]
        is_vulnerable = normalized["isVulnerable"]

    if "vulnerabilityType" not in normalized:
        if "type" in normalized:
            normalized["vulnerabilityType"] = normalized["type"]
        elif "vulnerability" in normalized:
            normalized["vulnerabilityType"] = normalized["vulnerability"]
        elif is_vulnerable is False:
            normalized["vulnerabilityType"] = "SAFE"
        elif is_vulnerable is True:
            normalized["vulnerabilityType"] = "XSS"

    hints = normalized.get("hints")
    if hints is None:
        normalized["hints"] = []
    elif not isinstance(hints, list):
        normalized["hints"] = [str(hints)]
    else:
        normalized["hints"] = [str(hint) for hint in hints][:2]

    return normalized


def _strip_heading_marks(text: str) -> str:
    # Remove Markdown markup (headings, bullets, quotes, emphasis, code ticks)
    # but keep the prose itself, including apostrophes, brackets, quotes and
    # colons, which an earlier character allowlist deleted ("user's" -> "users").
    clean_text = re.sub(r"^\s*(?:#{1,6}|[-*+]|>)\s+", "", text, flags=re.MULTILINE)
    clean_text = re.sub(r"\*\*|__|[*`]", "", clean_text)
    return re.sub(r"\s+", " ", clean_text).strip()
