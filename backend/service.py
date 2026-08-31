import io
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from backend.schemas import AnalysisResponse
from ml.ml1.multimodal import analyze_multimodal_wellness
from ml.ml2.chat import WellnessChat
from ml.ml2.calibration import get_profile
from ml.ml2.report import generate_session_report
from ml.ml2.timeline import get_timeline, reset_timeline
from ml.ml2.triage import predict_triage_score


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MIN_FRAMES = 30
DEFAULT_FPS = 30.0


@dataclass
class SessionState:
    session_id: str
    latest_vitals: dict[str, Any] | None = None
    latest_triage: Any = None
    chat: WellnessChat = field(default_factory=WellnessChat)
    calibration_readings: int = 0


class BackendService:
    """Owns transport/session orchestration around the unchanged ML layers."""

    def __init__(self) -> None:
        self.session: SessionState | None = None

    def create_session(self) -> SessionState:
        if self.session is not None:
            return self.session
        reset_timeline()
        self.session = SessionState(session_id=str(uuid.uuid4()))
        return self.session

    def require_session(self, session_id: str) -> SessionState:
        if self.session is None or self.session.session_id != session_id:
            raise LookupError("Monitoring session not found.")
        return self.session

    def delete_session(self, session_id: str) -> None:
        self.require_session(session_id)
        self.session = None
        reset_timeline()

    @staticmethod
    def _decode_image(data: bytes) -> list[np.ndarray]:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded image could not be decoded.")
        return [image]

    @staticmethod
    def _decode_video(data: bytes, filename: str) -> tuple[list[np.ndarray], float]:
        suffix = os.path.splitext(filename)[1] or ".mp4"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(data)
                temp_path = temp_file.name

            capture = cv2.VideoCapture(temp_path)
            if not capture.isOpened():
                raise ValueError("The uploaded video could not be opened.")

            fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
            frames: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            return frames, float(fps)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @classmethod
    def decode_media(cls, data: bytes, filename: str, content_type: str | None) -> tuple[list[np.ndarray], float]:
        if not data:
            raise ValueError("Uploaded media is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Uploaded media exceeds the 50 MB limit.")

        extension = os.path.splitext(filename)[1].lower()
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        is_image = extension in image_extensions or (content_type or "").startswith("image/")
        is_video = extension in video_extensions or (content_type or "").startswith("video/")

        if is_image:
            raise ValueError("A video with at least 30 frames is required for analysis.")
        if is_video:
            return cls._decode_video(data, filename)
        raise ValueError("Unsupported media type. Upload an image or video file.")

    def analyze(self, session_id: str, data: bytes, filename: str, content_type: str | None) -> AnalysisResponse:
        session = self.require_session(session_id)
        frames, fps = self.decode_media(data, filename, content_type)
        if len(frames) < MIN_FRAMES:
            raise ValueError(f"At least {MIN_FRAMES} video frames are required for analysis.")
        if not 1.0 <= fps <= 120.0:
            fps = DEFAULT_FPS

        vitals = analyze_multimodal_wellness(frames, fps=fps)
        triage = predict_triage_score(vitals)
        session.latest_vitals = vitals
        session.latest_triage = triage
        session.chat.set_context(
            vitals,
            triage_result=triage,
            wellness_score=triage.wellness_score,
            emotional_map=triage.emotional_map,
            contradiction=triage.contradiction,
            trend_summary=triage.trend_summary,
        )
        return AnalysisResponse(
            session_id=session_id,
            vitals=vitals,
            triage=triage.model_dump(mode="json"),
        )

    def clarification(self, session_id: str, answer: str) -> AnalysisResponse:
        session = self.require_session(session_id)
        if session.latest_vitals is None:
            raise ValueError("Run an analysis before submitting clarification.")
        triage = predict_triage_score(session.latest_vitals, user_answer=answer)
        session.latest_triage = triage
        return AnalysisResponse(
            session_id=session_id,
            vitals=session.latest_vitals,
            triage=triage.model_dump(mode="json"),
        )

    def calibrate(self, session_id: str) -> dict[str, Any]:
        session = self.require_session(session_id)
        if session.latest_vitals is None:
            raise ValueError("Run an analysis before calibration.")
        profile = get_profile()
        profile.add_calibration_reading(session.latest_vitals)
        session.calibration_readings += 1
        baseline = profile.finalize_calibration()
        return {
            "session_id": session_id,
            "baseline": baseline,
            "readings": session.calibration_readings,
            "profile": profile.to_dict(),
        }

    def chat(self, session_id: str, message: str) -> str:
        session = self.require_session(session_id)
        if session.latest_vitals is None:
            raise ValueError("Run an analysis before starting chat.")
        return session.chat.chat(message)

    def timeline(self, session_id: str) -> dict[str, Any]:
        self.require_session(session_id)
        return get_timeline().get_trend_summary()

    def report(self, session_id: str) -> dict[str, Any]:
        session = self.require_session(session_id)
        return generate_session_report(
            get_timeline(),
            latest_triage=session.latest_triage,
            latest_score=session.latest_triage.wellness_score if session.latest_triage else None,
            latest_emotional=session.latest_triage.emotional_map if session.latest_triage else None,
            latest_contradiction=session.latest_triage.contradiction if session.latest_triage else None,
        )


service = BackendService()
