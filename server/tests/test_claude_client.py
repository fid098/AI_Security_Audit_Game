from __future__ import annotations

import json
import random
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.integrations import claude_client
from app.integrations.claude_client import (
    ClaudeTaskItem,
    ClaudeTaskPayload,
    _call_claude,
    _extract_json,
    _normalize_task_item,
    _normalize_vuln_type,
    _parse_json_payload,
    _sample_vulnerable_target,
    _strip_heading_marks,
    _validate_tasks_payload,
    generate_frontend_tasks,
    generate_security_mentor_summary,
)


def task_item(is_vulnerable: bool = True, vuln_type: str = "XSS") -> ClaudeTaskItem:
    return ClaudeTaskItem(code="x", isVulnerable=is_vulnerable, vulnerabilityType=vuln_type)


def make_message(*blocks, stop_reason="end_turn"):
    """A stand-in for the SDK's Message: content blocks plus a stop reason."""
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def fake_sdk_client(message=None, raises=None):
    """A stand-in for anthropic.Anthropic that records the create() call."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return message

    return SimpleNamespace(messages=SimpleNamespace(create=create), calls=calls)


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class TestExtractJson:
    def test_extracts_object_from_surrounding_prose(self):
        text = 'Sure! Here is the JSON:\n{"tasks": []}\nHope that helps.'
        assert _extract_json(text) == '{"tasks": []}'

    def test_extracts_object_from_a_markdown_fence(self):
        text = '```json\n{"tasks": [1]}\n```'
        assert _extract_json(text) == '{"tasks": [1]}'

    def test_extracts_a_bare_array(self):
        assert _extract_json("prefix [1, 2, 3] suffix") == "[1, 2, 3]"

    def test_handles_nested_structures(self):
        text = 'x {"tasks": [{"code": "a"}]} y'
        assert _extract_json(text) == '{"tasks": [{"code": "a"}]}'

    def test_handles_balanced_braces_inside_code_snippets(self):
        text = '{"code": "function f() { return 1; }"}'
        assert json.loads(_extract_json(text))["code"] == "function f() { return 1; }"

    @pytest.mark.parametrize(
        "code", ["if (a) { run(", "}", "arr[0", "]]]", 'say("{")']
    )
    def test_unbalanced_brackets_inside_strings_do_not_confuse_it(self, code):
        """Bracket counting broke on these; a real parser skips string contents."""
        text = "Here you go: " + json.dumps({"tasks": [{"code": code}]}) + " done"

        assert json.loads(_extract_json(text))["tasks"][0]["code"] == code

    def test_skips_bracketed_prose_that_is_not_json(self):
        assert _extract_json('As noted [here], the result is {"a": 1}') == '{"a": 1}'

    def test_returns_only_the_first_complete_object(self):
        assert _extract_json('{"a": 1} {"b": 2}') == '{"a": 1}'

    def test_raises_when_no_json_present(self):
        with pytest.raises(ValueError, match="did not contain valid JSON"):
            _extract_json("I cannot help with that request.")

    def test_raises_on_an_unterminated_object(self):
        with pytest.raises(ValueError):
            _extract_json('{"tasks": [')


class TestParseJsonPayload:
    def test_parses_plain_json(self):
        text = '{"tasks": [{"code": "a", "isVulnerable": true, "vulnerabilityType": "XSS"}]}'

        payload = _parse_json_payload(text)

        assert len(payload.tasks) == 1
        assert payload.tasks[0].vulnerabilityType == "XSS"

    def test_recovers_from_a_markdown_fenced_response(self):
        text = (
            'Here you go:\n```json\n{"tasks": [{"code": "a", "isVulnerable": false, '
            '"vulnerabilityType": "SAFE"}]}\n```'
        )

        assert _parse_json_payload(text).tasks[0].isVulnerable is False

    def test_accepts_a_bare_top_level_array(self):
        text = '[{"code": "a", "isVulnerable": true, "vulnerabilityType": "RCE"}]'

        assert _parse_json_payload(text).tasks[0].vulnerabilityType == "RCE"

    def test_accepts_snippets_as_an_alias_for_tasks(self):
        text = '{"snippets": [{"code": "a", "isVulnerable": true, "vulnerabilityType": "SSRF"}]}'

        assert _parse_json_payload(text).tasks[0].vulnerabilityType == "SSRF"

    def test_invalid_shape_raises_value_error(self):
        with pytest.raises(ValueError, match="validation failed"):
            _parse_json_payload('{"tasks": [{"missing": "fields"}]}')

    def test_unknown_vulnerability_type_raises_value_error(self):
        text = '{"tasks": [{"code": "a", "isVulnerable": true, "vulnerabilityType": "TIME_TRAVEL"}]}'

        with pytest.raises(ValueError):
            _parse_json_payload(text)


class TestNormalizeTaskItem:
    def test_non_dict_items_become_empty(self):
        assert _normalize_task_item("nonsense") == {}
        assert _normalize_task_item(None) == {}

    def test_vulnerable_alias_is_mapped(self):
        assert _normalize_task_item({"vulnerable": True})["isVulnerable"] is True

    def test_type_alias_is_mapped(self):
        assert _normalize_task_item({"type": "SSRF"})["vulnerabilityType"] == "SSRF"

    def test_vulnerability_alias_is_mapped(self):
        assert _normalize_task_item({"vulnerability": "RCE"})["vulnerabilityType"] == "RCE"

    def test_safe_type_is_inferred_when_not_vulnerable(self):
        assert _normalize_task_item({"isVulnerable": False})["vulnerabilityType"] == "SAFE"

    def test_xss_is_assumed_when_vulnerable_but_untyped(self):
        assert _normalize_task_item({"isVulnerable": True})["vulnerabilityType"] == "XSS"

    def test_missing_hints_become_an_empty_list(self):
        assert _normalize_task_item({"code": "a"})["hints"] == []

    def test_scalar_hints_are_wrapped(self):
        assert _normalize_task_item({"hints": "just one"})["hints"] == ["just one"]

    def test_hints_are_capped_at_two(self):
        assert _normalize_task_item({"hints": ["a", "b", "c", "d"]})["hints"] == ["a", "b"]

    def test_hint_entries_are_coerced_to_strings(self):
        assert _normalize_task_item({"hints": [1, 2]})["hints"] == ["1", "2"]

    def test_original_item_is_not_mutated(self):
        original = {"code": "a"}
        _normalize_task_item(original)
        assert original == {"code": "a"}


class TestNormalizeVulnType:
    def test_safe_code_is_always_marked_safe(self):
        assert _normalize_vuln_type(False, "XSS") == "SAFE"

    def test_vulnerable_code_cannot_be_marked_safe(self):
        """A vulnerable snippet labelled SAFE would make the round unwinnable."""
        assert _normalize_vuln_type(True, "SAFE") == "XSS"

    def test_valid_pairings_pass_through(self):
        assert _normalize_vuln_type(True, "SQL_INJECTION") == "SQL_INJECTION"


class TestValidateTasksPayload:
    def test_accepts_a_matching_payload(self):
        payload = ClaudeTaskPayload(tasks=[task_item(True), task_item(False, "SAFE")])

        assert _validate_tasks_payload(payload, count=2) is payload

    def test_rejects_the_wrong_number_of_tasks(self):
        payload = ClaudeTaskPayload(tasks=[task_item()])

        with pytest.raises(ValueError, match="expected 3"):
            _validate_tasks_payload(payload, count=3)

    @pytest.mark.parametrize("vulnerable", [0, 1, 2])
    def test_any_vulnerable_count_is_accepted(self, vulnerable):
        """The target is random, so missing it is not worth discarding a paid reply."""
        tasks = [task_item(i < vulnerable, "XSS" if i < vulnerable else "SAFE") for i in range(2)]

        assert _validate_tasks_payload(ClaudeTaskPayload(tasks=tasks), count=2).tasks == tasks


class TestSampleVulnerableTarget:
    @pytest.mark.parametrize("density", [0.0, 0.4, 0.6, 0.8, 1.0])
    @pytest.mark.parametrize("count", [1, 3, 5, 10])
    def test_always_between_one_and_count(self, count, density):
        rng = random.Random(0)

        draws = [_sample_vulnerable_target(count, density, rng) for _ in range(200)]

        assert all(1 <= n <= count for n in draws)

    def test_zero_density_still_yields_one_vulnerable_snippet(self):
        assert _sample_vulnerable_target(5, 0.0, random.Random(0)) == 1

    def test_full_density_makes_every_snippet_vulnerable(self):
        assert _sample_vulnerable_target(5, 1.0, random.Random(0)) == 5

    def test_target_varies_between_rounds(self):
        """The whole point: a fixed count let players deduce the last answers."""
        rng = random.Random(0)

        draws = {_sample_vulnerable_target(5, 0.6, rng) for _ in range(200)}

        assert len(draws) >= 3

    @pytest.mark.parametrize("density", [0.4, 0.6, 0.8])
    def test_average_tracks_the_configured_density(self, density):
        rng = random.Random(0)
        count, rounds = 10, 2000

        mean = sum(_sample_vulnerable_target(count, density, rng) for _ in range(rounds)) / rounds

        assert mean == pytest.approx(count * density, abs=0.2)


class TestStripHeadingMarks:
    def test_markdown_symbols_are_removed(self):
        assert _strip_heading_marks("## Summary **bold**") == "Summary bold"

    def test_whitespace_is_collapsed(self):
        assert _strip_heading_marks("a\n\n  b\tc") == "a b c"

    def test_sentence_punctuation_is_preserved(self):
        text = "Input is unescaped, so XSS is possible - fix it."
        assert _strip_heading_marks(text) == text

    def test_empty_input_is_handled(self):
        assert _strip_heading_marks("") == ""

    def test_prose_punctuation_survives(self):
        """An earlier character allowlist turned "user's" into "users"."""
        text = "The handler's query (line 3) concatenates input: use \"?\" placeholders; don't format."
        assert _strip_heading_marks(text) == text

    def test_snake_case_identifiers_are_kept(self):
        assert _strip_heading_marks("Validate user_id first.") == "Validate user_id first."

    def test_bullets_quotes_and_code_ticks_are_removed(self):
        text = "- First point\n* Second `innerHTML` point\n> quoted"
        assert _strip_heading_marks(text) == "First point Second innerHTML point quoted"


class TestCallClaude:
    def _call(self):
        return _call_claude(
            api_key="k",
            model="m",
            system_prompt="s",
            user_prompt="u",
            max_tokens=10,
        )

    def _use(self, monkeypatch, client):
        monkeypatch.setattr(claude_client, "_client", lambda api_key: client)
        return client

    def test_concatenates_text_content_blocks(self, monkeypatch):
        message = make_message(text_block("part one"), text_block("part two"))
        self._use(monkeypatch, fake_sdk_client(message))

        assert self._call() == "part one\npart two"

    def test_thinking_blocks_contribute_no_text(self, monkeypatch):
        thinking = SimpleNamespace(type="thinking", thinking="")
        self._use(monkeypatch, fake_sdk_client(make_message(thinking, text_block("answer"))))

        assert self._call() == "answer"

    def test_sends_model_system_prompt_and_token_cap(self, monkeypatch):
        client = self._use(monkeypatch, fake_sdk_client(make_message(text_block("ok"))))

        self._call()

        assert client.calls == [
            {
                "model": "m",
                "max_tokens": 10,
                "system": "s",
                "messages": [{"role": "user", "content": "u"}],
            }
        ]

    def test_truncated_response_raises_instead_of_returning_partial_text(self, monkeypatch):
        message = make_message(text_block('{"tasks": [{"code": "a'), stop_reason="max_tokens")
        self._use(monkeypatch, fake_sdk_client(message))

        with pytest.raises(RuntimeError, match="truncated"):
            self._call()

    def test_refusal_raises(self, monkeypatch):
        self._use(monkeypatch, fake_sdk_client(make_message(stop_reason="refusal")))

        with pytest.raises(RuntimeError, match="declined"):
            self._call()

    def test_timeout_becomes_a_runtime_error(self, monkeypatch):
        self._use(monkeypatch, fake_sdk_client(raises=anthropic.APITimeoutError(request=REQUEST)))

        with pytest.raises(RuntimeError, match="timed out"):
            self._call()

    @pytest.mark.parametrize(
        ("error_cls", "status"),
        [
            (anthropic.BadRequestError, 400),
            (anthropic.RateLimitError, 429),
            (anthropic.InternalServerError, 500),
            (anthropic.InternalServerError, 529),
        ],
    )
    def test_status_errors_become_runtime_errors_with_the_code(
        self, monkeypatch, error_cls, status
    ):
        error = error_cls("error", response=httpx2.Response(status, request=REQUEST), body=None)
        self._use(monkeypatch, fake_sdk_client(raises=error))

        with pytest.raises(RuntimeError, match=str(status)):
            self._call()

    def test_connection_error_becomes_a_runtime_error(self, monkeypatch):
        self._use(monkeypatch, fake_sdk_client(raises=anthropic.APIConnectionError(request=REQUEST)))

        with pytest.raises(RuntimeError, match="connection failed"):
            self._call()


class TestClientFactory:
    def test_configures_key_timeout_and_retries(self):
        claude_client._client.cache_clear()
        try:
            client = claude_client._client("test-key")

            assert client.api_key == "test-key"
            assert client.max_retries == claude_client.MAX_RETRIES
            assert client.timeout == claude_client.REQUEST_TIMEOUT_SECONDS
        finally:
            claude_client._client.cache_clear()

    def test_reuses_one_client_per_key(self):
        claude_client._client.cache_clear()
        try:
            assert claude_client._client("a") is claude_client._client("a")
            assert claude_client._client("a") is not claude_client._client("b")
        finally:
            claude_client._client.cache_clear()


class TestGenerateFrontendTasks:
    def _generate(self, count=2, vuln_density=0.5):
        return generate_frontend_tasks(
            language="javascript",
            difficulty="EASY",
            complexity_level="basic",
            count=count,
            vuln_density=vuln_density,
        )

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not configured"):
            self._generate()

    def test_assigns_unique_ids_and_the_requested_language(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(
            claude_client,
            "_call_claude",
            lambda **kwargs: json.dumps(
                {
                    "tasks": [
                        {"code": "a", "isVulnerable": True, "vulnerabilityType": "XSS"},
                        {"code": "b", "isVulnerable": False, "vulnerabilityType": "SAFE"},
                    ]
                }
            ),
        )

        tasks = self._generate()

        assert len({task.id for task in tasks}) == 2
        assert all(task.language == "javascript" for task in tasks)

    def test_missing_system_names_are_filled_from_the_roster(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(
            claude_client,
            "_call_claude",
            lambda **kwargs: json.dumps(
                {
                    "tasks": [
                        {"code": "a", "isVulnerable": True, "vulnerabilityType": "XSS"},
                        {"code": "b", "isVulnerable": False, "vulnerabilityType": "SAFE"},
                    ]
                }
            ),
        )

        tasks = self._generate()

        assert [task.systemName for task in tasks] == ["O2", "NAVIGATION"]

    def test_prompt_asks_for_the_sampled_vulnerable_count(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(claude_client, "_sample_vulnerable_target", lambda count, density: 4)
        captured = {}

        def _call(**kwargs):
            captured.update(kwargs)
            return json.dumps(
                {"tasks": [{"code": "a", "isVulnerable": True, "vulnerabilityType": "XSS"}] * 5}
            )

        monkeypatch.setattr(claude_client, "_call_claude", _call)

        self._generate(count=5, vuln_density=0.6)

        assert "exactly 4 snippets are vulnerable" in captured["user_prompt"]

    def test_upstream_count_mismatch_surfaces_as_value_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(
            claude_client,
            "_call_claude",
            lambda **kwargs: json.dumps(
                {"tasks": [{"code": "a", "isVulnerable": True, "vulnerabilityType": "XSS"}]}
            ),
        )

        with pytest.raises(ValueError, match="expected 2"):
            self._generate()


class TestGenerateSecurityMentorSummary:
    def test_no_failures_short_circuits_without_calling_the_api(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        summary = generate_security_mentor_summary(["log"], [])

        assert "No vulnerabilities were missed" in summary

    def test_missing_api_key_raises_when_there_is_work_to_do(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            generate_security_mentor_summary(["log"], ["REACTOR: xss"])

    def test_scanner_noise_is_withheld_from_the_prompt(self, monkeypatch):
        """The mentor persona must not mention tooling, so those logs are filtered out."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured = {}

        def _call(**kwargs):
            captured.update(kwargs)
            return "Summary text."

        monkeypatch.setattr(claude_client, "_call_claude", _call)
        generate_security_mentor_summary(
            ["Hacktron scan failed: offline", "unescaped output sink"], ["REACTOR: xss"]
        )

        sent = json.loads(captured["user_prompt"])
        assert sent["hacktron_logs"] == ["unescaped output sink"]

    def test_every_scanner_status_message_is_withheld(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured = {}
        monkeypatch.setattr(
            claude_client, "_call_claude", lambda **kwargs: captured.update(kwargs) or "ok"
        )
        status_lines = [
            claude_client.NOT_CONFIGURED_MESSAGE,
            "Hacktron scan failed: exit code 2",
            "Audit failed: timeout",
        ]

        generate_security_mentor_summary(status_lines, ["REACTOR: xss"])

        assert json.loads(captured["user_prompt"])["hacktron_logs"] == []

    def test_real_findings_that_name_the_scanner_are_kept(self, monkeypatch):
        """The old filter dropped any line containing "hacktron", findings included."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured = {}
        monkeypatch.setattr(
            claude_client, "_call_claude", lambda **kwargs: captured.update(kwargs) or "ok"
        )
        finding = '{"tool": "hacktron", "rule": "sql-injection", "line": 3}'

        generate_security_mentor_summary([finding], ["REACTOR: sqli"])

        assert json.loads(captured["user_prompt"])["hacktron_logs"] == [finding]

    def test_requests_enough_tokens_to_think_and_answer(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured = {}
        monkeypatch.setattr(
            claude_client, "_call_claude", lambda **kwargs: captured.update(kwargs) or "ok"
        )

        generate_security_mentor_summary(["log"], ["REACTOR: xss"])

        assert captured["max_tokens"] == claude_client.MAX_TOKENS

    def test_response_is_stripped_of_markdown(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(claude_client, "_call_claude", lambda **kwargs: "## **Summary**")

        assert generate_security_mentor_summary(["log"], ["REACTOR: xss"]) == "Summary"
