import importlib.util
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

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
        backend.supports_speaker = False
        mock_mm.backend = backend
        yield mock_mm


def test_transcribe_sync_with_external_diarizer_uses_async_helper(mock_model_manager, monkeypatch):
    from src.core.engine import TranscriptionEngine

    monkeypatch.setattr(engine_mod.settings, "speaker_external_diarizer_enable", True, raising=False)
    monkeypatch.setattr(
        engine_mod.settings, "speaker_external_diarizer_base_url", "http://diar:8000", raising=False
    )
    monkeypatch.setattr(engine_mod.settings, "speaker_unsupported_behavior", "error", raising=False)

    engine = TranscriptionEngine()

    expected = {
        "text": "speaker text",
        "text_accu": None,
        "sentences": [],
        "speaker_turns": [],
        "transcript": "t",
        "raw_text": "speaker text",
    }

    engine._transcribe_with_external_diarizer = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    audio_bytes = b"\x00" * (2 * 16000 * 2)
    out = engine.transcribe(audio_bytes, with_speaker=True, apply_hotword=False, apply_llm=False)

    assert out == expected
    mock_model_manager.backend.transcribe.assert_not_called()

