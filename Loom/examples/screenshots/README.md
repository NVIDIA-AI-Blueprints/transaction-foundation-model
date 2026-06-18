# Demo screenshots

Reference screenshots for [`../demo-tabformer.md`](../demo-tabformer.md) — the real output
of each `loom` command on the TabFormer sample, rendered to a clean terminal image.

| File | Step | Shows |
|---|---|---|
| `01-raw.png` | 1 | the raw TabFormer CSV (15 columns, `$`-amounts, 19-digit merchant IDs) |
| `02-ingest.png` | 2 | `loom ingest` + the leakage/identity scan |
| `03-propose.png` | 3 | `loom propose` — a strategy for every field, with reasons |
| `04-tokenize.png` | 4 | `loom tokenize` — a 1,055-token Corpus, contracts PASS |
| `05-refused.png` | 5 | the C1 contract refusing a colliding spec (no Corpus written) |
| `06-preset.png` | 6 | `loom tokenize --preset financial` — the 6,251-token production tokenizer |

These are rendered headlessly (no GUI) from the **actual** command output — they are not
mock-ups. When you record the video you'll capture your own live terminal; these exist so the
doc reads completely on its own.

## Regenerate

```bash
S=/tmp/loom-shots; mkdir -p $S/work; cd $S/work
python <repo>/Loom/examples/tabformer_sample.py
loom ingest   --in tf.csv --entity User --event transaction --target "Is Fraud?" --name tabformer -q
loom propose  --in IngestDataset/1 -q
cp <repo>/Loom/examples/collision.yaml .
# capture each command's output through a pty (preserves layout), then render to PNG:
script -q $S/03-propose.ansi  loom propose  --in IngestDataset/1
# ... (one per step) ...
python <repo>/Loom/examples/render_shot.py $S/03-propose.col out.png "loom propose"
```

`render_shot.py` parses ANSI, word-wraps to ~100 cols, and draws a dark terminal window
(Menlo, traffic-light chrome). Requires Pillow (`pip install Pillow`) and macOS system fonts.
