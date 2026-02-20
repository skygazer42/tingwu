import threading

from fastapi.testclient import TestClient


def test_diarizer_service_warmup_on_startup(monkeypatch):
    monkeypatch.setenv("DIARIZER_WARMUP_ON_STARTUP", "true")

    from src.diarizer_service.app import app
    import src.diarizer_service.routes as routes_mod

    ev = threading.Event()

    def fake_load() -> None:
        ev.set()

    monkeypatch.setattr(routes_mod.engine, "load", fake_load)

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200

    assert ev.wait(1.0)

