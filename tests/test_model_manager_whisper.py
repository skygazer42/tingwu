from unittest.mock import Mock


def test_model_manager_initializes_whisper_backend(monkeypatch):
    from src.config import settings

    prev_backend = settings.asr_backend
    settings.asr_backend = "whisper"

    try:
        import src.models.model_manager as mm

        backend_stub = Mock()
        backend_stub.get_info.return_value = {"name": "WhisperBackend", "type": "whisper"}

        fake_get_backend = Mock(return_value=backend_stub)
        monkeypatch.setattr(mm, "get_backend", fake_get_backend)

        mm.model_manager._backend = None
        out = mm.model_manager.backend

        assert out is backend_stub
        assert fake_get_backend.called
    finally:
        settings.asr_backend = prev_backend

