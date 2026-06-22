"""Prepare a laptop-sized, demo-ready sample of TabFormer for the Loom tokenization demo.

TabFormer is IBM's synthetic credit-card dataset (~24.4M transactions). Download it
per docs/04-data/01-tabformer.md (notebook 01 automates it, or the IBM Box link) to
  data/TabFormer/card_transaction.v1.csv

This script:
  1. reads the first N rows — a representative slice. The tokenizer is config-only
     (deterministic), so a SAMPLE is all you need to DESIGN and validate it; the
     vocabulary is bit-identical on all 24M rows.
  2. fills the documented online-transaction defaults: a blank `Zip` -> "00000" and a
     blank `Merchant State` -> "XX" (online rows have no ZIP/state). This keeps the
     financial preset's materialization clean.

    python examples/tabformer_prep.py [PATH_TO_CSV] [N_ROWS]
    # defaults: data/TabFormer/card_transaction.v1.csv, 200000 rows -> tabformer_sample.csv
"""
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/TabFormer/card_transaction.v1.csv"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000

df = pd.read_csv(src, nrows=n, dtype=str)
df = df.fillna({"Zip": "00000", "Merchant State": "XX"})
df.to_csv("tabformer_sample.csv", index=False)
print(f"wrote tabformer_sample.csv ({len(df)} rows, {len(df.columns)} columns, from {src})")
