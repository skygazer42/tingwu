from src.core.text_processor.post_processor import TextPostProcessor, PostProcessorSettings


def _processor(**kwargs) -> TextPostProcessor:
    # Keep defaults off so tests only validate the new behavior.
    settings = PostProcessorSettings(
        filler_remove_enable=False,
        filler_aggressive=False,
        qj2bj_enable=False,
        itn_enable=False,
        itn_erhua_remove=False,
        spacing_cjk_ascii_enable=False,
        spoken_punc_enable=kwargs.get("spoken_punc_enable", False),
        acronym_merge_enable=kwargs.get("acronym_merge_enable", False),
        zh_convert_enable=False,
        zh_convert_locale="zh-hans",
        punc_convert_enable=False,
        punc_add_space=True,
        punc_restore_enable=False,
        punc_restore_model="ct-punc-c",
        punc_restore_device="cpu",
        punc_merge_enable=False,
        trash_punc_enable=False,
        trash_punc_chars="，。,.",
    )
    return TextPostProcessor(settings)


def test_spoken_punctuation_commands_prefix_and_suffix_only():
    pp = _processor(spoken_punc_enable=True)

    assert pp.process("你好逗号") == "你好，"
    assert pp.process("句号你好") == "。你好"

    # Conservative: do not replace in the middle.
    assert pp.process("你好逗号世界") == "你好逗号世界"


def test_spoken_punctuation_commands_multiple_commands():
    pp = _processor(spoken_punc_enable=True)

    assert pp.process("回车回车你好") == "\n\n你好"
    assert pp.process("你好句号回车") == "你好。\n"


def test_acronym_merge_basic():
    pp = _processor(acronym_merge_enable=True)

    assert pp.process("A I 技术") == "AI 技术"
    assert pp.process("V S Code") == "VS Code"
    assert pp.process("U S A") == "USA"


def test_acronym_merge_is_opt_in():
    pp = _processor(acronym_merge_enable=False)
    assert pp.process("A I 技术") == "A I 技术"

