import asyncio
import importlib.util
import sys
import types
from unittest.mock import MagicMock, patch


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
from src.utils.service_metrics import metrics


def test_metrics_counts_external_diarizer_requests(monkeypatch):
    metrics.reset()

    with patch.object(engine_mod, "model_manager") as mock_mm:
        backend = MagicMock()
        backend.get_info.return_value = {"name": "MockBackend", "type": "mock"}
        backend.supports_speaker = False
        backend.transcribe.return_value = {"text": "第一段", "sentence_info": []}
        mock_mm.backend = backend

        monkeypatch.setattr(engine_mod.settings, "speaker_unsupported_behavior", "ignore", raising=False)
        monkeypatch.setattr(engine_mod.settings, "speaker_external_diarizer_enable", True, raising=False)
        monkeypatch.setattr(
            engine_mod.settings, "speaker_external_diarizer_base_url", "http://diar:8000", raising=False
        )

        async def fake_fetch(*args, **kwargs):
            return [{"spk": 0, "start": 0, "end": 1000}]

        audio_bytes = b"\x00" * (2 * 16000 * 2)

        with patch.object(engine_mod, "fetch_diarizer_segments", new=fake_fetch):
            engine = engine_mod.TranscriptionEngine()
            _out = asyncio.run(
                engine.transcribe_async(audio_bytes, with_speaker=True, apply_hotword=False, apply_llm=False)
            )

    stats = metrics.get_stats()
    assert stats["diarizer_requests_total"] == 1
    assert stats["diarizer_failures_total"] == 0
    assert stats["diarizer_latency_seconds_count"] == 1
    assert stats["diarizer_latency_seconds_sum"] >= 0.0

