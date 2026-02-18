from fastapi.testclient import TestClient


def test_diarizer_health_and_diarize_route_exists():
    from src.diarizer_service.app import app

    c = TestClient(app)
    assert c.get("/health").status_code == 200
    # /api/v1/diarize should exist (even if returns 500 without model)
    assert c.post("/api/v1/diarize", files={"file": ("a.wav", b"RIFF")}).status_code in (200, 400, 500)

