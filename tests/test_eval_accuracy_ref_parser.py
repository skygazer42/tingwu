import json
from pathlib import Path

from scripts.eval_accuracy import get_reference_for_audio, load_reference_map


def test_load_reference_map_tsv(tmp_path: Path) -> None:
    p = tmp_path / "ref.txt"
    p.write_text(
        "# comment\n"
        "\n"
        "audio1.wav\thello world\n"
        "audio2.wav\t你好 世界\n",
        encoding="utf-8",
    )

    ref_map = load_reference_map(p)
    assert ref_map["audio1.wav"] == "hello world"
    assert ref_map["audio2.wav"] == "你好 世界"


def test_load_reference_map_json(tmp_path: Path) -> None:
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"a.wav": "A", "b.wav": "B"}, ensure_ascii=False), encoding="utf-8")

    ref_map = load_reference_map(p)
    assert ref_map["a.wav"] == "A"
    assert ref_map["b.wav"] == "B"


def test_get_reference_for_audio_matches_by_basename(tmp_path: Path) -> None:
    p = tmp_path / "ref.txt"
    p.write_text("a.wav\tREF A\n", encoding="utf-8")
    ref_map = load_reference_map(p)

    assert get_reference_for_audio(ref_map, tmp_path / "a.wav") == "REF A"

