# Loom examples

Reproducible artifacts for the tokenization walkthrough in
[`../TOKENIZATION.md`](../TOKENIZATION.md). Run them with the venv active
(`loom` = the engine CLI).

| File | What it is |
|---|---|
| `listens.py` | Generates `listens.csv` — a synthetic music-listening event log (the lead example: exercises mapping / hash / amount / calendar / timedelta / entity-exclusion). |
| `dna.py` | Generates `dna.csv` — 60 DNA reads over `{A,C,G,T}` (the "any domain" example). |
| `dna_kmer.yaml` | A k-mer field-map that tokenizes DNA into 3-mer codons → a 69-token vocab. |
| `collision.yaml` | A deliberately broken spec (two fields share a token prefix) that trips the C1 contract → `REFUSED`, no Corpus written. |

## The ≈4-minute demo, end to end

```bash
# 1. a music-listening tokenizer, in three commands
python examples/listens.py
loom ingest   --in listens.csv --entity user_id --event listen --name music-listens
loom propose  --in IngestDataset/1
loom tokenize --in IngestDataset/1 --spec TokenizerSpec/1          # → Corpus, vocab 593

# 2. you're in control: edit a field, re-tokenize (see TOKENIZATION.md §6)
#    (e.g. change `track` to {strategy: hash, buckets: 64} → vocab 456)

# 3. the safety net: a deliberate collision is refused, nothing written
loom tokenize --in IngestDataset/1 --spec examples/collision.yaml  # → REFUSED_CONTRACT (C1)

# 4. any domain: the same verbs tokenize DNA
python examples/dna.py
loom ingest   --in dna.csv --entity seq_id --event read --name dna-reads
loom tokenize --in IngestDataset/2 --spec examples/dna_kmer.yaml   # → Corpus, vocab 69
```
