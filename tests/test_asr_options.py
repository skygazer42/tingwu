import pytest

from src.api.asr_options import parse_asr_options


def test_asr_options_chunking_strategy_normalizes():
    opts = parse_asr_options('{"chunking":{"strategy":" TIME "}}')
    assert opts is not None
    assert opts["chunking"]["strategy"] == "time"


def test_asr_options_chunking_strategy_rejects_unknown():
    with pytest.raises(ValueError, match="chunking\\.strategy"):
        parse_asr_options('{"chunking":{"strategy":"vad"}}')

