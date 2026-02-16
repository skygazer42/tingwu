import numpy as np

from src.core.audio.preprocessor import AudioPreprocessor


def test_adaptive_skips_denoise_on_high_snr(monkeypatch):
    pre = AudioPreprocessor(
        normalize_enable=False,
        trim_silence_enable=False,
        denoise_enable=True,
        adaptive_enable=True,
        remove_dc_offset=False,
        highpass_enable=False,
        soft_limit_enable=False,
    )

    monkeypatch.setattr(pre, "estimate_snr", lambda audio, sample_rate=16000: 40.0)

    called = {"denoise": 0}

    def _fake_denoise(audio, sample_rate=16000):
        called["denoise"] += 1
        return audio

    monkeypatch.setattr(pre, "denoise", _fake_denoise)

    audio = np.random.default_rng(0).normal(0.0, 0.1, size=(16000,)).astype(np.float32)
    _ = pre.process(audio, sample_rate=16000, validate=False)

    assert called["denoise"] == 0


def test_adaptive_enables_denoise_on_low_snr(monkeypatch):
    pre = AudioPreprocessor(
        normalize_enable=False,
        trim_silence_enable=False,
        denoise_enable=False,
        adaptive_enable=True,
        remove_dc_offset=False,
        highpass_enable=False,
        soft_limit_enable=False,
    )

    monkeypatch.setattr(pre, "estimate_snr", lambda audio, sample_rate=16000: 0.0)

    called = {"denoise": 0}

    def _fake_denoise(audio, sample_rate=16000):
        called["denoise"] += 1
        return audio * 0.5

    monkeypatch.setattr(pre, "denoise", _fake_denoise)

    audio = np.random.default_rng(0).normal(0.0, 0.1, size=(16000,)).astype(np.float32)
    out = pre.process(audio, sample_rate=16000, validate=False)

    assert called["denoise"] == 1
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(audio)))

