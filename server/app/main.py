import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .integrations.claude_client import generate_frontend_tasks, generate_security_mentor_summary
from .integrations.elevenlabs import generate_speech, validate_api_key
from .integrations.hacktron import scan_with_hacktron
from .integrations.reporting import build_findings, summarize_findings
from .schemas import (
    DIFFICULTY_CONFIGS,
    AuditReport,
    AuditRequest,
    AuditResponse,
    FinishResponse,
    GenerateSnippetsRequest,
    GenerateSnippetsResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SubmitAnswersRequest,
    SubmitAnswersResponse,
    TaskListResponse,
    TTSRequest,
    TTSResponse,
)
from .store import InMemoryStore, SessionFinishedError

# Fills in settings missing from the environment, for local development. Real
# environment variables win, so values set by Docker or CI are never replaced
# by a stale .env file.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
app = FastAPI(title="Security Sabotage API")
store = InMemoryStore()

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    # The app uses no cookies or HTTP auth, so browsers need not send them.
    allow_credentials=False,
    # Only what the client actually sends.
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/elevenlabs")
def elevenlabs_health() -> dict:
    """Check if ElevenLabs API is accessible with the configured key."""
    is_valid, message = validate_api_key()
    return {
        "service": "elevenlabs",
        "status": "ok" if is_valid else "error",
        "message": message
    }


@app.get("/")
def index() -> dict:
    # The game itself is served by the client (Vite in development, nginx in
    # Docker). This used to return client/index.html, which is the unbuilt dev
    # source and could not load from here, and was absent in the image.
    return {
        "service": app.title,
        "docs": app.docs_url,
        "health": "/health",
    }


@app.post("/session", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreateRequest) -> SessionCreateResponse:
    # Same mapping as /generate, which shares the generation path.
    try:
        session = store.create_session(payload.difficulty, payload.task_count, payload.language)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to create session: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to create session: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session. Please try again."
        ) from exc

    return SessionCreateResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        difficulty=session.difficulty,
        language=session.language,
        config=DIFFICULTY_CONFIGS[session.difficulty],
    )


@app.get("/session/{session_id}/tasks", response_model=TaskListResponse)
def list_tasks(session_id: str) -> TaskListResponse:
    try:
        tasks = store.list_public_tasks(session_id)
        return TaskListResponse(session_id=session_id, tasks=tasks)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        ) from exc


@app.post("/session/{session_id}/submit", response_model=SubmitAnswersResponse)
def submit_answers(session_id: str, payload: SubmitAnswersRequest) -> SubmitAnswersResponse:
    try:
        return store.submit_answers(session_id, payload.answers)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        ) from exc
    except SessionFinishedError as exc:
        # 409 Conflict: the request is valid but clashes with the session's state.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc


@app.post("/session/{session_id}/finish", response_model=FinishResponse)
def finish_session(session_id: str) -> FinishResponse:
    try:
        return store.finish_session(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        ) from exc


@app.get("/session/{session_id}/results", response_model=FinishResponse)
def get_results(session_id: str) -> FinishResponse:
    try:
        return store.get_session_results(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        ) from exc


@app.post("/generate", response_model=GenerateSnippetsResponse)
def generate_snippets(payload: GenerateSnippetsRequest) -> GenerateSnippetsResponse:
    # The schema already restricts difficulty and count, so both are valid here.
    config = DIFFICULTY_CONFIGS[payload.difficulty.lower()]

    try:
        tasks = generate_frontend_tasks(
            language=payload.language,
            difficulty=payload.difficulty,
            complexity_level=payload.complexityLevel,
            count=payload.count,
            vuln_density=config.vuln_density,
        )
        if not config.hints_allowed:
            tasks = [task.model_copy(update={"hints": None}) for task in tasks]
        return GenerateSnippetsResponse(tasks=tasks)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)
        ) from exc
    except ValueError as exc:
        # The request was valid; Claude's reply was not. That is an upstream
        # fault, so 502 rather than a 400 that blames the caller.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate code snippets. Please try again."
        ) from exc


@app.post("/audit", response_model=AuditResponse)
def audit_tasks(payload: AuditRequest) -> AuditResponse:
    if not payload.tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tasks provided for audit."
        )

    task_payload = [(task.id, task.code) for task in payload.tasks]
    hacktron_logs: list[str] = []

    try:
        hacktron_output = scan_with_hacktron(task_payload, payload.language)
        hacktron_logs = [log for _, log in hacktron_output]
    except Exception as exc:
        hacktron_logs = [f"Hacktron scan failed: {str(exc)}"]

    findings = build_findings(payload.tasks)

    try:
        summary = generate_security_mentor_summary(
            hacktron_logs,
            [f"{task.systemName}: {task.vulnerabilityType}" for task in payload.tasks if task.isVulnerable],
        )
    except Exception:
        summary = summarize_findings(findings)

    report = AuditReport(findings=findings, summary=summary)
    return AuditResponse(report=report)


@app.post("/tts", response_model=TTSResponse)
def tts(payload: TTSRequest) -> TTSResponse:
    """Generate text-to-speech audio using ElevenLabs API."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty."
        )

    # Additional security: validate text length (already in schema but double-check)
    if len(payload.text) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text exceeds maximum length of 5000 characters."
        )

    try:
        audio_url, duration = generate_speech(
            text=payload.text,
            voice_id=payload.voiceId
        )
        return TTSResponse(audioUrl=audio_url, duration=duration)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate speech. Please try again."
        ) from exc
