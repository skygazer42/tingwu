"""集成测试"""
import pytest
from unittest.mock import patch, Mock, MagicMock


def test_full_pipeline():
    """测试完整转写流程"""
    with patch('src.core.engine.model_manager') as mock_mm:
        mock_loader = MagicMock()
        mock_mm.loader = mock_loader
        mock_loader.transcribe.return_value = {
            "text": "买当劳很好吃",
            "sentence_info": [
                {"text": "买当劳很好吃", "start": 0, "end": 1500, "spk": 0}
            ]
        }

        from src.core.engine import TranscriptionEngine

        engine = TranscriptionEngine()
        engine.update_hotwords(["麦当劳", "肯德基"])

        result = engine.transcribe(
            b"fake_audio",
            with_speaker=True,
            apply_hotword=True
        )

        # 验证热词纠错生效
        assert "麦当劳" in result["text"]
        # 验证说话人标注
        assert result["sentences"][0]["speaker"] == "说话人甲"


def test_multi_speaker_pipeline():
    """测试多说话人流程"""
    with patch('src.core.engine.model_manager') as mock_mm:
        mock_loader = MagicMock()
        mock_mm.loader = mock_loader
        mock_loader.transcribe.return_value = {
            "text": "你好世界你也好",
            "sentence_info": [
                {"text": "你好世界", "start": 0, "end": 1000, "spk": 0},
                {"text": "你也好", "start": 1200, "end": 2000, "spk": 1},
            ]
        }

        from src.core.engine import TranscriptionEngine

        engine = TranscriptionEngine()
        result = engine.transcribe(b"fake_audio", with_speaker=True)

        assert len(result["sentences"]) == 2
        assert result["sentences"][0]["speaker"] == "说话人甲"
        assert result["sentences"][1]["speaker"] == "说话人乙"
        assert "transcript" in result


def test_hotword_correction_accuracy():
    """测试热词纠错准确率"""
    from src.core.hotword import PhonemeCorrector

    corrector = PhonemeCorrector(threshold=0.8)
    corrector.update_hotwords("Claude\nBilibili\n麦当劳\n肯德基\nFunASR")

    test_cases = [
        ("Hello klaude", "Claude"),
        ("我要去买当劳", "麦当劳"),
        ("肯得鸡真好吃", "肯德基"),
    ]

    for input_text, expected_word in test_cases:
        result = corrector.correct(input_text)
        assert expected_word in result.text, f"Failed: '{input_text}' -> '{result.text}', expected '{expected_word}'"


def test_hotword_no_false_positives():
    """测试不误纠正"""
    from src.core.hotword import PhonemeCorrector

    corrector = PhonemeCorrector(threshold=0.8)
    corrector.update_hotwords("Claude\n麦当劳")

    safe_texts = [
        "今天天气不错",
        "我要去上班",
        "Python是一种编程语言",
    ]

    for text in safe_texts:
        result = corrector.correct(text)
        assert result.text == text, f"False positive: '{text}' -> '{result.text}'"
