from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from .integrations.claude_client import generate_frontend_tasks, generate_security_mentor_summary
from .integrations.hacktron import scan_with_hacktron
from .schemas import (
    DIFFICULTY_CONFIGS,
    AnswerSchema,
    AuditLogSchema,
    Difficulty,
    FinishResponse,
    FrontendDifficulty,
    FrontendLanguage,
    FrontendTask,
    MentorReportSchema,
    SubmitAnswersResponse,
    TaskPublicSchema,
    TaskSchema,
)

# Sessions idle for longer than this are discarded.
DEFAULT_SESSION_TTL_SECONDS = 60 * 60

# Hard ceiling on live sessions, so memory stays bounded even if sessions are
# created faster than they expire.
DEFAULT_MAX_SESSIONS = 1000


class SessionFinishedError(Exception):
    """Raised when answers arrive for a session whose report is already built."""


@dataclass
class SessionData:
    session_id: str
    difficulty: Difficulty
    created_at: datetime
    tasks: list[TaskSchema]
    language: FrontendLanguage = "javascript"
    answers: dict[str, AnswerSchema] = field(default_factory=dict)
    audit_logs: list[AuditLogSchema] = field(default_factory=list)
    mentor_report: MentorReportSchema | None = None
    # Set once the report is built. The audit logs cannot stand in for this:
    # a perfect round has none, so it used to be re-audited on every request.
    finished: bool = False
    last_access: float = field(default_factory=time.monotonic)
    # Serialises submit and finish for this session only, so two concurrent
    # finish calls cannot both run the scanner, and one slow audit never blocks
    # other players.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class InMemoryStore:
    """Process-local session store.

    Sessions live in one process's memory, so this does not survive a restart
    or work across several server workers; that would need a shared store such
    as Redis.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sessions: dict[str, SessionData] = {}
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        # Guards the sessions dict itself; held only for quick dict operations.
        self._lock = threading.Lock()

    # Create a new session with generated tasks
    def create_session(
        self, difficulty: Difficulty, task_count: int, language: FrontendLanguage = "javascript"
    ) -> SessionData:
        # Generation calls Claude and is slow, so it runs before taking the lock.
        tasks = generate_tasks(difficulty, task_count, language)
        session = SessionData(
            session_id=uuid4().hex,
            difficulty=difficulty,
            created_at=datetime.now(UTC),
            tasks=tasks,
            language=language,
            last_access=self._clock(),
        )
        with self._lock:
            self._evict_expired()
            # At the cap, drop the least recently used session to make room.
            while len(self.sessions) >= self._max_sessions:
                oldest = min(self.sessions.values(), key=lambda s: s.last_access)
                del self.sessions[oldest.session_id]
            self.sessions[session.session_id] = session
        return session

    # Retrieve a session by ID, refreshing its expiry
    def get_session(self, session_id: str) -> SessionData:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is not None and self._is_expired(session):
                del self.sessions[session_id]
                session = None
            if session is None:
                raise KeyError(f"Unknown session: {session_id}")
            session.last_access = self._clock()
            return session

    def _is_expired(self, session: SessionData) -> bool:
        return self._clock() - session.last_access > self._ttl_seconds

    def _evict_expired(self) -> None:
        # Caller holds self._lock.
        for session_id in [sid for sid, s in self.sessions.items() if self._is_expired(s)]:
            del self.sessions[session_id]

    def list_public_tasks(self, session_id: str) -> list[TaskPublicSchema]:
        session = self.get_session(session_id)
        hints_allowed = DIFFICULTY_CONFIGS[session.difficulty].hints_allowed
        return [
            TaskPublicSchema(
                id=task.id,
                system_name=task.system_name,
                code=task.code,
                difficulty=task.difficulty,
                language=task.language,
                hints=task.hints if hints_allowed else None,
            )
            for task in session.tasks
        ]

    def submit_answers(self, session_id: str, answers: list[AnswerSchema]) -> SubmitAnswersResponse:
        session = self.get_session(session_id)
        valid_ids = {task.id for task in session.tasks}
        seen = set()

        # Validate the whole batch before writing any of it, so a bad id part way
        # through cannot leave the session holding a partially applied submission.
        for answer in answers:
            if answer.task_id not in valid_ids:
                raise ValueError(f"Unknown task id: {answer.task_id}")
            if answer.task_id in seen:
                raise ValueError(f"Duplicate answer for task id: {answer.task_id}")
            seen.add(answer.task_id)

        with session.lock:
            # The report is built from the answers at finish time, so accepting
            # more afterwards would leave the score and the report disagreeing.
            if session.finished:
                raise SessionFinishedError(f"Session already finished: {session_id}")
            for answer in answers:
                session.answers[answer.task_id] = answer
            correct, _, missed_task_ids = score_session(session)

        return SubmitAnswersResponse(
            correct=correct,
            incorrect=len(missed_task_ids),
            missed_task_ids=missed_task_ids,
        )

    def finish_session(self, session_id: str) -> FinishResponse:
        session = self.get_session(session_id)

        # Held for the whole audit: a concurrent finish waits here, then finds
        # the session finished and returns the cached report instead of scanning.
        with session.lock:
            correct, _, missed_task_ids = score_session(session)
            if not session.finished:
                try:
                    session.audit_logs = build_hacktron_audit_logs(session, missed_task_ids)
                except Exception as exc:
                    session.audit_logs = [
                        AuditLogSchema(task_id=task_id, raw_log=f"Audit failed: {str(exc)}")
                        for task_id in missed_task_ids
                    ]

                try:
                    session.mentor_report = build_mentor_report(session, missed_task_ids)
                except Exception:
                    session.mentor_report = build_fallback_mentor_report(missed_task_ids)

                session.finished = True

            return FinishResponse(
                session_id=session_id,
                score=correct,
                missed_task_ids=missed_task_ids,
                audit_logs=session.audit_logs,
                mentor_report=session.mentor_report,
            )

    def get_session_results(self, session_id: str) -> FinishResponse:
        # finish_session returns the cached report when the session is already
        # finished, and builds it on demand when it is not.
        return self.finish_session(session_id)

# Generate tasks based on difficulty and count
def generate_tasks(
    difficulty: Difficulty, task_count: int, language: FrontendLanguage = "javascript"
) -> list[TaskSchema]:
    config = DIFFICULTY_CONFIGS[difficulty]
    frontend_tasks = generate_frontend_tasks(
        language=language,
        difficulty=_to_frontend_difficulty(difficulty),
        complexity_level=_map_complexity(config.complexity_tag),
        count=task_count,
        vuln_density=config.vuln_density,
    )
    return [_to_task_schema(task, difficulty) for task in frontend_tasks]

def score_session(session: SessionData) -> tuple[int, int, list[str]]:
    correct = 0
    missed: list[str] = []

    for task in session.tasks:
        answer = session.answers.get(task.id)
        if not answer:
            missed.append(task.id)
            continue

        expected_choice = "sabotaged" if task.is_vulnerable else "clean"
        if answer.user_choice == expected_choice:
            correct += 1
        else:
            missed.append(task.id)

    incorrect = len(missed)
    return correct, incorrect, missed

# Build audit logs using Hacktron for missed tasks
def build_hacktron_audit_logs(session: SessionData, missed_task_ids: list[str]) -> list[AuditLogSchema]:
    if not missed_task_ids:
        return []
    missed_tasks = [task for task in session.tasks if task.id in missed_task_ids]
    task_payload = [(task.id, task.code) for task in missed_tasks]
    logs = scan_with_hacktron(task_payload, session.language)
    return [
        AuditLogSchema(task_id=task_id, raw_log=raw_log)
        for task_id, raw_log in logs
    ]

# Build mentor report using Claude based on audit logs and missed tasks
def build_mentor_report(session: SessionData, missed_task_ids: list[str]) -> MentorReportSchema:
    failed_tasks = [task for task in session.tasks if task.id in missed_task_ids]
    failed_summaries = [
        f"{task.system_name}: {task.vulnerability_type} in {task.language or 'javascript'}"
        for task in failed_tasks
    ]
    hacktron_logs = [log.raw_log for log in session.audit_logs]
    summary = generate_security_mentor_summary(hacktron_logs, failed_summaries)
    return MentorReportSchema(summary=summary)

# Fallback mentor report if Claude integration fails
def build_fallback_mentor_report(missed_task_ids: list[str]) -> MentorReportSchema:
    if not missed_task_ids:
        summary = "Clean sweep. No missed vulnerabilities detected in this run."
    else:
        summary = (
            "You missed at least one vulnerable system. Review input handling, "
            "use parameterized queries, and validate outbound requests."
        )
    return MentorReportSchema(summary=summary)

# Map internal difficulty to frontend difficulty format
def _to_frontend_difficulty(difficulty: Difficulty) -> FrontendDifficulty:
    return difficulty.upper()  # type: ignore[return-value]

# Map complexity tags to frontend complexity levels
def _map_complexity(tag: str) -> str:
    return {"low": "basic", "medium": "intermediate", "high": "advanced"}.get(tag, "basic")

# Convert FrontendTask to TaskSchema
def _to_task_schema(frontend_task: FrontendTask, difficulty: Difficulty) -> TaskSchema:
    return TaskSchema(
        id=frontend_task.id,
        system_name=frontend_task.systemName,
        code=frontend_task.code,
        is_vulnerable=frontend_task.isVulnerable,
        vulnerability_type=_map_vuln_type(frontend_task.vulnerabilityType),
        difficulty=difficulty,
        language=frontend_task.language,
        vulnerability_line=frontend_task.vulnerabilityLine,
        hints=frontend_task.hints,
    )

# Normalize vulnerability type based on whether the task is vulnerable
def _map_vuln_type(vuln_type: str) -> str:
    return {
        "XSS": "xss",
        "SQL_INJECTION": "sqli",
        "SSRF": "ssrf",
        "RCE": "rce",
        "PATH_TRAVERSAL": "path_traversal",
        "COMMAND_INJECTION": "command_injection",
        "INSECURE_DESERIALIZATION": "insecure_deserialization",
        "SAFE": "none",
    }.get(vuln_type, "none")
