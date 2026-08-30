import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from backend.schemas import (
    AnalysisResponse,
    CalibrationResponse,
    ChatRequest,
    ChatResponse,
    ClarificationRequest,
    HealthResponse,
    SessionResponse,
)
from backend.service import service

app = FastAPI(title="Contactless Wellness Monitor API", version="0.1.0")

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/test", StaticFiles(directory=FRONTEND_DIR, html=True), name="test-frontend")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Wellness Monitor API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", ml_layers={"ml1": "ready", "ml2": "ready"})


@app.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    session = service.create_session()
    return SessionResponse(
        session_id=session.session_id,
        status="active",
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    try:
        service.delete_session(session_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"status": "deleted"}


@app.post("/sessions/{session_id}/analyze", response_model=AnalysisResponse)
async def analyze(session_id: str, file: UploadFile = File(...)) -> AnalysisResponse:
    try:
        data = await file.read()
        return service.analyze(session_id, data, file.filename or "upload", file.content_type)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Analysis failed.") from error


@app.post("/sessions/{session_id}/clarification", response_model=AnalysisResponse)
def clarification(session_id: str, request: ClarificationRequest) -> AnalysisResponse:
    try:
        return service.clarification(session_id, request.answer)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    try:
        response = service.chat(session_id, request.message)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ChatResponse(session_id=session_id, response=response)


@app.post("/sessions/{session_id}/calibrate", response_model=CalibrationResponse)
def calibrate(session_id: str) -> CalibrationResponse:
    try:
        return CalibrationResponse(**service.calibrate(session_id))
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/sessions/{session_id}/timeline")
def timeline(session_id: str) -> dict:
    try:
        return service.timeline(session_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/sessions/{session_id}/report")
def report(session_id: str) -> dict:
    try:
        return service.report(session_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
