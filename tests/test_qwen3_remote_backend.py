from unittest.mock import patch


def test_qwen3_remote_backend_calls_chat_completions_and_parses_asr_text_tag():
    from src.models.backends.qwen3_remote import Qwen3RemoteBackend

    backend = Qwen3RemoteBackend(
        base_url="http://fake",
        model="Qwen/Qwen3-ASR-0.6B",
        api_key="EMPTY",
        timeout_s=1.0,
    )

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "language Chinese<asr_text>ok"}}]}

        def raise_for_status(self):
            return None

    with patch("httpx.Client.post", return_value=Resp()) as post:
        out = backend.transcribe(b"\x00\x00" * 16000, hotwords="foo bar")
        assert out["text"] == "ok"
        assert "sentence_info" in out

        assert post.called
        called_url = post.call_args.args[0]
        assert called_url == "http://fake/v1/chat/completions"
        called_json = post.call_args.kwargs["json"]
        assert called_json["model"] == "Qwen/Qwen3-ASR-0.6B"


def test_qwen3_remote_backend_formats_hotwords_as_a_context_hint():
    from src.models.backends.qwen3_remote import Qwen3RemoteBackend

    backend = Qwen3RemoteBackend(
        base_url="http://fake",
        model="Qwen/Qwen3-ASR-0.6B",
        api_key="EMPTY",
        timeout_s=1.0,
    )

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "language Chinese<asr_text>ok"}}]}

        def raise_for_status(self):
            return None

    with patch("httpx.Client.post", return_value=Resp()) as post:
        _ = backend.transcribe(b"\x00\x00" * 16000, hotwords="OpenAI\nTingWu")
        called_json = post.call_args.kwargs["json"]
        content = called_json["messages"][0]["content"]
        text_blocks = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        merged = "\n".join(text_blocks)
        assert "专有名词" in merged
        assert "OpenAI" in merged
        assert "TingWu" in merged
