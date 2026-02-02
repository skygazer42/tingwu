from typing import Any, Dict, Optional


def test_router_backend_routes_by_duration_and_speaker_flag():
    from src.models.backends.base import ASRBackend
    from src.models.backends.router import RouterBackend

    class StubBackend(ASRBackend):
        def __init__(self, name: str, supports_speaker: bool):
            self.name = name
            self._supports_speaker = supports_speaker
            self.calls = []

        def load(self) -> None:
            return None

        @property
        def supports_speaker(self) -> bool:
            return self._supports_speaker

        def transcribe(self, audio_input, hotwords: Optional[str] = None, **kwargs) -> Dict[str, Any]:
            self.calls.append({"hotwords": hotwords, **kwargs})
            return {"text": self.name, "sentence_info": []}

    short = StubBackend("short", supports_speaker=False)
    long = StubBackend("long", supports_speaker=True)

    router = RouterBackend(
        short_backend=short,
        long_backend=long,
        long_audio_threshold_s=2.0,
        force_vibevoice_when_with_speaker=True,
    )

    short_audio = b"\x00\x00" * 16000  # 1s @ 16kHz PCM16LE mono
    long_audio = b"\x00\x00" * 16000 * 3  # 3s

    out1 = router.transcribe(short_audio, with_speaker=False)
    assert out1["text"] == "short"

    out2 = router.transcribe(long_audio, with_speaker=False)
    assert out2["text"] == "long"

    out3 = router.transcribe(short_audio, with_speaker=True)
    assert out3["text"] == "long"
    assert router.supports_speaker is True

