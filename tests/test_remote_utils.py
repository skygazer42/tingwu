def test_pcm16k_to_wav_bytes_roundtrip_has_riff_header():
    from src.models.backends.remote_utils import pcm16le_to_wav_bytes

    # 1s of silence (16kHz, 16-bit mono)
    wav = pcm16le_to_wav_bytes(b"\x00\x00" * 16000)
    assert wav[:4] == b"RIFF"

