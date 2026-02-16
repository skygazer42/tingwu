import numpy as np

from src.core.audio.preprocessor import AudioPreprocessor


def _sine(freq_hz: float, duration_s: float, sr: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr)), dtype=np.float32) / float(sr)
    return (amp * np.sin(2.0 * np.pi * float(freq_hz) * t)).astype(np.float32)


def test_robust_normalization_targets_active_segment_rms():
    sr = 16000
    silence = np.zeros((sr * 9,), dtype=np.float32)
    speech = _sine(440.0, 1.0, sr, amp=0.1)
    audio = np.concatenate([silence, speech])

    base = dict(
        target_db=-20.0,
        normalize_enable=True,
        trim_silence_enable=False,
        denoise_enable=False,
        adaptive_enable=False,
        remove_dc_offset=False,
        highpass_enable=False,
        soft_limit_enable=False,
    )

    pre_global = AudioPreprocessor(
        **base,
        normalize_robust_rms_enable=False,
    )
    pre_robust = AudioPreprocessor(
        **base,
        normalize_robust_rms_enable=True,
        normalize_robust_rms_percentile=95.0,
    )

    out_global = pre_global.process(audio, sample_rate=sr, validate=False)
    out_robust = pre_robust.process(audio, sample_rate=sr, validate=False)

    target_rms = float(pre_global.target_rms)
    speech_global_rms = float(np.sqrt(np.mean(out_global[-sr:] ** 2)))
    speech_robust_rms = float(np.sqrt(np.mean(out_robust[-sr:] ** 2)))

    assert abs(speech_robust_rms - target_rms) < abs(speech_global_rms - target_rms)
    assert speech_global_rms > target_rms * 1.5


def test_robust_normalization_amplifies_background_noise_less():
    sr = 16000
    rng = np.random.default_rng(0)

    silence_with_noise = rng.normal(0.0, 0.001, size=(sr * 9,)).astype(np.float32)
    speech = _sine(440.0, 1.0, sr, amp=0.05) + rng.normal(0.0, 0.001, size=(sr,)).astype(np.float32)
    audio = np.concatenate([silence_with_noise, speech]).astype(np.float32)

    base = dict(
        target_db=-20.0,
        normalize_enable=True,
        trim_silence_enable=False,
        denoise_enable=False,
        adaptive_enable=False,
        remove_dc_offset=False,
        highpass_enable=False,
        soft_limit_enable=False,
    )

    pre_global = AudioPreprocessor(
        **base,
        normalize_robust_rms_enable=False,
    )
    pre_robust = AudioPreprocessor(
        **base,
        normalize_robust_rms_enable=True,
        normalize_robust_rms_percentile=95.0,
    )

    out_global = pre_global.process(audio, sample_rate=sr, validate=False)
    out_robust = pre_robust.process(audio, sample_rate=sr, validate=False)

    noise_global_rms = float(np.sqrt(np.mean(out_global[: sr * 9] ** 2)))
    noise_robust_rms = float(np.sqrt(np.mean(out_robust[: sr * 9] ** 2)))

    assert noise_robust_rms < noise_global_rms * 0.7

