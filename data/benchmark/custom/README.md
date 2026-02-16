# Custom Benchmark Data (not committed audio by default)

Put your own “golden” accuracy samples here (e.g. a ~5-minute meeting clip) to do before/after comparisons.

## Suggested layout

- `audio.wav` / `audio.mp3` / `meeting.wav` … (your audio file)
- `ref.txt` or `ref.json` (reference transcripts mapping)

### `ref.txt` format (TSV)

One file per line:

```
meeting.wav\t这里放参考转写文本（尽量是最终稿）
```

Lines starting with `#` are ignored.

### `ref.json` format

```
{
  "meeting.wav": "这里放参考转写文本"
}
```

## Run eval (HTTP ports / multi-container)

Example:

```bash
python scripts/eval_accuracy.py \
  --audio data/benchmark/custom \
  --ref data/benchmark/custom/ref.txt \
  --ports 8101 8102 8200 \
  --with-speaker \
  --asr-options '{"speaker":{"label_style":"numeric","turn_merge_gap_ms":800}}'
```

