# Meeting/Recall Speaker Output — Implementation Plan (20 Tasks)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship meeting/recall-friendly transcription output with **speaker 1/2/3** labels and **speaker turns** (merged paragraphs), while keeping long-audio accuracy work (`text_accu`, context hotwords, preprocessing) compatible and controllable per request.

**Architecture:** Upload → FFmpeg decode to 16kHz PCM16LE → optional preprocessing → ASR backend → (if no speaker + long) chunk + merge (`text_accu`) → correction pipeline → optional LLM polish → API response.  
When `with_speaker=true`: prefer **single-pass diarization** (no chunking by default) → label → build turns → format transcript.

**Tech Stack:** Python, FastAPI, numpy, FFmpeg, FunASR backends, remote OpenAI-compatible ASR backends (Qwen3/VibeVoice), pytest.

---

## Status snapshot (already implemented)

These are already present on `main` and should be treated as the baseline:

- Context hotwords file + hot reload: `data/hotwords/hotwords-context.txt`, `HOTWORDS_CONTEXT_FILE`
- Long-audio precision merge output: `text_accu`
- Robust chunk merge + boundary reconcile (non-speaker mode)
- Per-request tuning via `asr_options` for preprocess/chunking/backend/postprocess
- Multi-model per-port containers with on-demand profiles (`docker-compose.models.yml`)

This plan focuses on **speaker-friendly meeting output** plus **evaluation tooling** for accuracy iteration.

---

## Task 01: Add `asr_options.speaker` section (allowlist + types)

**Files:**
- Modify: `src/api/asr_options.py`
- Test: `tests/test_asr_options.py`

**Step 1: Write failing tests**

Add tests:

```python
def test_asr_options_speaker_keys_allowed():
    opts = parse_asr_options('{"speaker":{"label_style":"numeric","turn_merge_enable":true,"turn_merge_gap_ms":800}}')
    assert opts["speaker"]["label_style"] == "numeric"

def test_asr_options_speaker_rejects_unknown_keys():
    with pytest.raises(ValueError, match="asr_options\\.speaker"):
        parse_asr_options('{"speaker":{"wat":1}}')
```

**Step 2: Run tests (expect FAIL)**

Run: `pytest -q tests/test_asr_options.py`  
Expected: FAIL because `speaker` is not an allowed section.

**Step 3: Implement minimal parser support**

- Add `"speaker"` to `_TOP_LEVEL_KEYS`
- Add `_SPEAKER_KEYS`, `_SPEAKER_TYPES`
- Add validation + normalization for `label_style` ∈ `{zh,numeric}`

**Step 4: Run tests (expect PASS)**

Run: `pytest -q tests/test_asr_options.py`

**Step 5: Commit**

```bash
git add src/api/asr_options.py tests/test_asr_options.py
git commit -m "feat: allow asr_options speaker section"
git push origin main
```

---

## Task 02: Add speaker defaults to `Settings` + `.env.example`

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`
- Optional docs: `README.md`

**Step 1: Add settings (no behavior change yet)**

Add:
- `speaker_label_style: Literal["zh","numeric"]="zh"`
- `speaker_turn_merge_enable: bool = True`
- `speaker_turn_merge_gap_ms: int = 800`
- `speaker_turn_merge_min_chars: int = 1`

**Step 2: Compile check**

Run: `python3 -m compileall -q src`

**Step 3: Commit**

```bash
git add src/config.py .env.example README.md
git commit -m "config: add speaker output defaults"
git push origin main
```

---

## Task 03: Implement numeric speaker labels in `SpeakerLabeler`

**Files:**
- Modify: `src/core/speaker/diarization.py`
- Test: `tests/test_speaker.py`

**Step 1: Write failing test**

```python
def test_speaker_labeling_numeric_style():
    labeler = SpeakerLabeler(label_style="numeric")
    out = labeler.label_speakers([{"text":"A","spk":0},{"text":"B","spk":1}])
    assert out[0]["speaker"] == "说话人1"
    assert out[1]["speaker"] == "说话人2"
```

**Step 2: Run test (expect FAIL)**

Run: `pytest -q tests/test_speaker.py`

**Step 3: Implement**

- Add `label_style: str = "zh"` param
- If `numeric`, label is `f"{prefix}{mapped_id + 1}"`
- Preserve existing `zh` default behavior

**Step 4: Run tests (expect PASS)**

Run: `pytest -q tests/test_speaker.py`

**Step 5: Commit**

```bash
git add src/core/speaker/diarization.py tests/test_speaker.py
git commit -m "feat: support numeric speaker labels"
git push origin main
```

---

## Task 04: Add speaker turn builder (pure python)

**Files:**
- Create: `src/core/speaker/turns.py`
- Modify: `src/core/speaker/__init__.py`
- Test: `tests/test_speaker_turns.py`

**Step 1: Write failing tests**

Cases:
- merge consecutive sentences by same speaker
- do not merge across speaker change
- do not merge if gap > threshold

**Step 2: Implement minimal turn builder**

Add `build_speaker_turns(sentences, gap_ms=800, min_chars=1)` returning list of dict:
`{"speaker","speaker_id","start","end","text","sentence_count"}`

**Step 3: Run tests**

Run: `pytest -q tests/test_speaker_turns.py`

**Step 4: Commit**

```bash
git add src/core/speaker/turns.py src/core/speaker/__init__.py tests/test_speaker_turns.py
git commit -m "feat: add speaker turn builder"
git push origin main
```

---

## Task 05: Format meeting transcript from `speaker_turns`

**Files:**
- Modify: `src/core/speaker/diarization.py`
- Test: `tests/test_speaker.py` (add new test)

**Step 1: Add failing test**

```python
def test_format_transcript_prefers_turns():
    labeler = SpeakerLabeler(label_style="numeric")
    turns = [{"speaker":"说话人1","start":0,"end":1000,"text":"你好"}]
    assert "说话人1" in labeler.format_transcript(turns)
```

**Step 2: Implement**

Extend `format_transcript(...)` to support both:
- sentence dicts (existing)
- turn dicts (same key schema)

**Step 3: Run tests + commit**

Run: `pytest -q tests/test_speaker.py`  
Commit + push.

---

## Task 06: Parse request-scoped speaker options in engine (no heavy deps)

**Files:**
- Modify: `src/core/engine.py`

**Step 1: Implement helper**

Add `_get_request_speaker_options(asr_options)` returning:
- `label_style`
- `turn_merge_enable`
- `turn_merge_gap_ms`
- `turn_merge_min_chars`

Default to `settings.*` when unset.

**Step 2: Compile check**

Run: `python3 -m compileall -q src`

**Step 3: Commit**

Commit + push.

---

## Task 07: Add `speaker_turns` to `transcribe_async` output

**Files:**
- Modify: `src/core/engine.py`

**Steps:**
1. After `label_speakers(...)`, call `build_speaker_turns(...)`
2. Put `speaker_turns` into result dict
3. Generate `transcript` from `speaker_turns` (not raw sentences)
4. Keep `sentences` unchanged

**Validation:**
- `python3 -m compileall -q src`

**Commit:** commit + push.

---

## Task 08: Ensure `transcribe_long_audio` remains safe with `with_speaker`

**Files:**
- Modify: `src/core/engine.py`

**Steps:**
1. If `with_speaker=true` and chunked path is entered, still build `speaker_turns` from merged `sentence_info`.
2. Add a warning comment/log that chunking+speaker may be inconsistent.

**Validation:** compileall  
**Commit:** commit + push.

---

## Task 09: Make `transcribe_auto_async` skip chunk routing when `with_speaker=true`

**Files:**
- Modify: `src/core/engine.py`

**Steps:**
1. Early-return to `transcribe_async(...)` when `with_speaker` is true.
2. Keep the current long-audio heuristic for non-speaker.

**Validation:** compileall  
**Commit:** commit + push.

---

## Task 10: Add strict behavior for unsupported speaker diarization (no silent fallback)

**Files:**
- Modify: `src/config.py`
- Modify: `src/core/engine.py`
- Optional docs: `README.md`

**Design:**
- Add `speaker_strict_backend: bool = True` (recommended for per-port container correctness)
- If `with_speaker=true` and backend doesn’t support speaker:
  - if strict: raise `ValueError("backend does not support speaker diarization")`
  - else: keep existing fallback to PyTorch

**Validation:** compileall  
**Commit:** commit + push.

---

## Task 11: Add API schema models for `speaker_turns`

**Files:**
- Modify: `src/api/schemas.py`

**Steps:**
1. Add `SpeakerTurn` model with fields
2. Add `speaker_turns: Optional[List[SpeakerTurn]]` to `TranscribeResponse`

**Validation:** compileall  
**Commit:** commit + push.

---

## Task 12: Return `speaker_turns` from HTTP routes

**Files:**
- Modify: `src/api/routes/transcribe.py`
- Modify: `src/api/routes/async_transcribe.py` (if it returns dict)

**Steps:**
1. Include `speaker_turns=result.get("speaker_turns")`
2. Keep `text/text_accu/sentences/transcript` stable

**Validation:** compileall  
**Commit:** commit + push.

---

## Task 13: Update README docs (meeting/recall usage)

**Files:**
- Modify: `README.md`

**Include:**
- `with_speaker=true` example
- numeric label style via `asr_options.speaker.label_style="numeric"`
- explain `speaker_turns` vs `sentences`

**Commit:** commit + push.

---

## Task 14: Add `tests/test_speaker_turns.py` edge cases

**Files:**
- Modify: `tests/test_speaker_turns.py`

**Cases:**
- missing timestamps default to 0
- empty texts
- unknown speaker_id

**Run:** `pytest -q tests/test_speaker_turns.py`  
**Commit:** commit + push.

---

## Task 15: Hotwords workflow for meetings (context list generator)

**Files:**
- Create: `scripts/hotwords/extract_context_hotwords.py`
- Optional docs: `README.md`

**Behavior:**
- Input: a plain text file (meeting notes / domain list)
- Output: sorted unique phrases (one per line), written to `data/hotwords/hotwords-context.txt` (or stdout)

**Validation:** `python3 -m compileall -q scripts`  
**Commit:** commit + push.

---

## Task 16: Add a pure-python CER/WER metric helper (for your 5-min reference TXT)

**Files:**
- Create: `scripts/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Steps:**
1. Implement Levenshtein DP for CER (character-level)
2. Implement WER with naive whitespace tokenization (for mixed English)
3. Add simple normalizers (strip spaces/punc options)

**Run:** `pytest -q tests/test_eval_metrics.py`  
**Commit:** commit + push.

---

## Task 17: Add A/B port comparison runner (multi-container workflow)

**Files:**
- Create: `scripts/eval/ab_compare_ports.py`
- Optional docs: `README.md`

**Behavior:**
- For a given audio file, call multiple ports:
  - `http://localhost:8101/api/v1/transcribe`
  - `http://localhost:8102/api/v1/transcribe`
  - ...
- Save responses + compute CER/WER if reference is provided

**Validation:** compileall  
**Commit:** commit + push.

---

## Task 18: Add “duplication ratio” metric (catch overlap repeats)

**Files:**
- Modify: `scripts/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Metric idea:**
- n-gram repeat ratio (e.g. repeated 4-grams / total)
- “longest repeated substring” heuristic (optional)

**Run:** `pytest -q tests/test_eval_metrics.py`  
**Commit:** commit + push.

---

## Task 19: Meeting-safe LLM role (optional, accuracy-first)

**Files:**
- Create: `src/core/llm/roles/meeting.py`
- Modify: `src/core/llm/roles/__init__.py` (if needed)
- Test: `tests/test_roles.py` (if lightweight), otherwise compileall only

**Prompt goals:**
- Keep content exactly (no summarization)
- Only fix obvious ASR errors and punctuation
- Respect hotwords/context hotwords

**Commit:** commit + push.

---

## Task 20: Add a `scripts/transcribe_meeting.sh` “best accuracy” recipe

**Files:**
- Create: `scripts/transcribe_meeting.sh`
- Modify: `README.md` (reference it)

**Behavior:**
- One command example that sets:
  - `with_speaker=true`
  - `asr_options.speaker.label_style=numeric`
  - `asr_options.speaker.turn_merge_gap_ms=800`
  - optionally `postprocess.punc_restore_enable=true`

**Commit:** commit + push.

---

## Execution

Once this plan is saved, execute tasks in order.  
Suggested minimal verification after each commit:

```bash
python3 -m compileall -q src tests scripts
pytest -q tests/test_asr_options.py tests/test_speaker.py tests/test_speaker_turns.py
```

