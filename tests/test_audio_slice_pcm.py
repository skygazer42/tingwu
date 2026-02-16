import wave
import io


def _pcm_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    samples = int(duration_ms * sample_rate / 1000)
    return b"\x00\x00" * samples


def _pcm_to_wav_bytes(pcm16le: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16le)
    return buf.getvalue()


def test_ensure_pcm16le_converts_wav_bytes_to_pcm():
    from src.core.audio.slice import ensure_pcm16le_16k_mono_bytes

    pcm = _pcm_silence(1000)
    wav = _pcm_to_wav_bytes(pcm)

    out = ensure_pcm16le_16k_mono_bytes(wav)
    assert isinstance(out, (bytes, bytearray))
    assert len(out) == len(pcm)


def test_slice_pcm16le_uses_ms_and_clamps():
    from src.core.audio.slice import slice_pcm16le

    pcm = _pcm_silence(1000)

    # 200ms -> 500ms = 300ms => 4800 samples => 9600 bytes
    out = slice_pcm16le(pcm, start_ms=200, end_ms=500)
    assert len(out) == 9600

    # Clamp negative / overflow.
    out2 = slice_pcm16le(pcm, start_ms=-100, end_ms=2000)
    assert out2 == pcm

