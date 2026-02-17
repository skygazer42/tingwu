from unittest.mock import Mock
import sys

import numpy as np


def test_whisper_backend_transcribe_pcm_bytes(monkeypatch):
    fake_whisper = Mock()
    fake_model = Mock()
    fake_model.transcribe.return_value = {
        "text": "hello",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
        ],
    }
    fake_whisper.load_model.return_value = fake_model

    # Ensure the backend imports our stub instead of a real dependency.
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    from src.models.backends.whisper import WhisperBackend

    backend = WhisperBackend(model="small", device="cuda", language="zh")
    backend.load()

    pcm = (np.zeros(16000, dtype=np.int16)).tobytes()
    out = backend.transcribe(pcm)

    assert out["text"] == "hello"
    assert out["sentence_info"] == [{"text": "hello", "start": 0, "end": 1000}]

