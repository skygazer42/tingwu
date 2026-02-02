from unittest.mock import patch


def test_vibevoice_remote_backend_parses_json_segments():
    from src.models.backends.vibevoice_remote import VibeVoiceRemoteBackend

    backend = VibeVoiceRemoteBackend(
        base_url="http://fake",
        model="vibevoice",
        api_key="EMPTY",
        timeout_s=1.0,
        use_chat_completions_fallback=True,
    )

    content = """```json
[
  {"Start time": 0.0, "End time": 1.5, "Speaker ID": 0, "Content": "hello"},
  {"Start time": 1.5, "End time": 2.0, "Speaker ID": 1, "Content": "world"}
]
```"""

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

        def raise_for_status(self):
            return None

    with patch("httpx.Client.post", return_value=Resp()) as post:
        out = backend.transcribe(b"\x00\x00" * 16000)
        assert out["text"] == "hello world"
        assert len(out["sentence_info"]) == 2

        s0 = out["sentence_info"][0]
        assert s0["text"] == "hello"
        assert s0["start"] == 0
        assert s0["end"] == 1500
        assert s0["spk"] == 0

        assert post.call_args.args[0] == "http://fake/v1/chat/completions"

