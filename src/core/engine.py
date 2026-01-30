"""核心转写引擎"""
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from src.config import settings
from src.models.model_manager import model_manager
from src.core.hotword import PhonemeCorrector
from src.core.speaker import SpeakerLabeler

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """转写引擎 - 整合 ASR + 热词纠错 + 说话人识别"""

    def __init__(self):
        self.corrector = PhonemeCorrector(
            threshold=settings.hotwords_threshold,
            similar_threshold=settings.hotwords_threshold - 0.2
        )
        self.speaker_labeler = SpeakerLabeler()
        self._hotwords_loaded = False

    def load_hotwords(self, path: Optional[str] = None):
        """加载热词"""
        if path is None:
            path = str(settings.hotwords_dir / settings.hotwords_file)

        if Path(path).exists():
            count = self.corrector.load_hotwords_file(path)
            logger.info(f"Loaded {count} hotwords from {path}")
            self._hotwords_loaded = True
        else:
            logger.warning(f"Hotwords file not found: {path}")

    def update_hotwords(self, hotwords: Union[str, List[str]]):
        """更新热词"""
        if isinstance(hotwords, list):
            hotwords = "\n".join(hotwords)

        count = self.corrector.update_hotwords(hotwords)
        logger.info(f"Updated {count} hotwords")
        self._hotwords_loaded = True

    def transcribe(
        self,
        audio_input: Union[bytes, str, Path],
        with_speaker: bool = False,
        apply_hotword: bool = True,
        hotwords: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行转写"""
        # 执行 ASR
        try:
            raw_result = model_manager.loader.transcribe(
                audio_input,
                hotwords=hotwords,
                with_speaker=with_speaker,
                **kwargs
            )
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            raise

        # 提取文本和句子信息
        text = raw_result.get("text", "")
        sentence_info = raw_result.get("sentence_info", [])

        # 热词纠错
        if apply_hotword and self._hotwords_loaded and text:
            correction = self.corrector.correct(text)
            text = correction.text

            # 同时纠错每个句子的文本
            for sent in sentence_info:
                sent_correction = self.corrector.correct(sent.get("text", ""))
                sent["text"] = sent_correction.text

        # 说话人标注
        if with_speaker and sentence_info:
            sentence_info = self.speaker_labeler.label_speakers(sentence_info)

        # 构建返回结果
        result = {
            "text": text,
            "sentences": [
                {
                    "text": s.get("text", ""),
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    **({"speaker": s.get("speaker"), "speaker_id": s.get("speaker_id")}
                       if with_speaker else {})
                }
                for s in sentence_info
            ],
            "raw_text": raw_result.get("text", ""),
        }

        # 生成格式化转写稿
        if with_speaker:
            result["transcript"] = self.speaker_labeler.format_transcript(
                result["sentences"],
                include_timestamp=True
            )

        return result

    def transcribe_streaming(
        self,
        audio_chunk: bytes,
        cache: Dict[str, Any],
        is_final: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """流式转写 (单个音频块)"""
        online_model = model_manager.loader.asr_model_online

        result = online_model.generate(
            input=audio_chunk,
            cache=cache.get("asr_cache", {}),
            is_final=is_final,
            **kwargs
        )

        if result:
            cache["asr_cache"] = result[0].get("cache", {})
            text = result[0].get("text", "")

            # 应用热词纠错
            if self._hotwords_loaded and text:
                correction = self.corrector.correct(text)
                text = correction.text

            return {"text": text, "is_final": is_final}

        return {"text": "", "is_final": is_final}


# 全局引擎实例
transcription_engine = TranscriptionEngine()
