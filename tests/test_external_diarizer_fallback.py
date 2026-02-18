import asyncio
import importlib.util
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _ensure_optional_dependency_stubs_installed() -> None:
    # These modules are imported at module-import time by the production code,
    # but aren't needed for these unit tests (we mock the backend).
    if "funasr" not in sys.modules and importlib.util.find_spec("funasr") is None:
        funasr_stub = types.ModuleType("funasr")

        class DummyAutoModel:
            def __init__(self, *args, **kwargs):
                pass

            def generate(self, **kwargs):
                return []

        funasr_stub.AutoModel = DummyAutoModel
        sys.modules["funasr"] = funasr_stub

    if "numba" not in sys.modules and importlib.util.find_spec("numba") is None:
        numba_stub = types.ModuleType("numba")

        def njit(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorator(func):
                return func

            return decorator

        numba_stub.njit = njit
        sys.modules["numba"] = numba_stub


_ensure_optional_dependency_stubs_installed()

import src.core.engine as engine_mod


@pytest.fixture
def mock_model_manager():
    with patch.object(engine_mod, "model_manager") as mock_mm:
        backend = MagicMock()
        backend.get_info.return_value = {"name": "MockBackend", "type": "mock"}
        mock_mm.backend = backend
        yield mock_mm


def test_external_diarizer_failure_falls_back_to_native_when_supported(mock_model_manager, monkeypatch):
    from src.core.engine import TranscriptionEngine

    mock_model_manager.backend.supports_speaker = True
    mock_model_manager.backend.transcribe.return_value = {
        "text": "你好",
        "sentence_info": [{"text": "你好", "start": 0, "end": 1000, "spk": 0}],
    }

    monkeypatch.setattr(engine_mod.settings, "speaker_external_diarizer_enable", True, raising=False)
    monkeypatch.setattr(
        engine_mod.settings, "speaker_external_diarizer_base_url", "http://diar:8000", raising=False
    )

    async def fake_fetch(*args, **kwargs):
        raise RuntimeError("diarizer down")

    audio_bytes = b"\x00" * (2 * 16000 * 2)

    with patch.object(engine_mod, "fetch_diarizer_segments", new=fake_fetch):
        engine = TranscriptionEngine()
        out = asyncio.run(engine.transcribe_async(audio_bytes, with_speaker=True, apply_hotword=False))

    assert out["sentences"][0]["speaker"] == "说话人甲"
    assert out.get("speaker_turns")
    assert out.get("transcript")

    assert mock_model_manager.backend.transcribe.call_count == 1
    kwargs = mock_model_manager.backend.transcribe.call_args.kwargs
    assert kwargs.get("with_speaker") is True


def test_external_diarizer_failure_ignores_speaker_when_backend_unsupported(mock_model_manager, monkeypatch):
    from src.core.engine import TranscriptionEngine

    mock_model_manager.backend.supports_speaker = False
    mock_model_manager.backend.transcribe.return_value = {
        "text": "普通文本",
        "sentence_info": [{"text": "普通文本", "start": 0, "end": 1000}],
    }

    # Even when strict behavior is configured, external diarizer failures should
    # not hard-fail the request.
    monkeypatch.setattr(engine_mod.settings, "speaker_unsupported_behavior", "error", raising=False)

    monkeypatch.setattr(engine_mod.settings, "speaker_external_diarizer_enable", True, raising=False)
    monkeypatch.setattr(
        engine_mod.settings, "speaker_external_diarizer_base_url", "http://diar:8000", raising=False
    )

    async def fake_fetch(*args, **kwargs):
        raise RuntimeError("diarizer down")

    audio_bytes = b"\x00" * (2 * 16000 * 2)

    with patch.object(engine_mod, "fetch_diarizer_segments", new=fake_fetch):
        engine = TranscriptionEngine()
        out = asyncio.run(engine.transcribe_async(audio_bytes, with_speaker=True, apply_hotword=False))

    assert out["text"] == "普通文本"
    assert out["sentences"] == [{"text": "普通文本", "start": 0, "end": 1000}]
    assert "speaker_turns" not in out
    assert "transcript" not in out

    assert mock_model_manager.backend.transcribe.call_count == 1
    kwargs = mock_model_manager.backend.transcribe.call_args.kwargs
    assert kwargs.get("with_speaker") is False

