"""Generate a tiny DNA dataset — the "any domain" example in TOKENIZATION.md.

60 reads of length 40 over the {A,C,G,T} alphabet. Tokenized into 3-mers (codons)
by examples/dna_kmer.yaml — a completely different modality through the same verbs.

    python examples/dna.py                # writes dna.csv
    loom ingest   --in dna.csv --entity seq_id --event read --name dna-reads
    loom tokenize --in IngestDataset/<n> --spec examples/dna_kmer.yaml   # → vocab 69
"""
import csv, random

random.seed(1)
rows = [["seq_id", "sequence"]]
for i in range(60):
    rows.append([f"s{i:03d}", "".join(random.choice("ACGT") for _ in range(40))])
csv.writer(open("dna.csv", "w", newline="")).writerows(rows)
print(f"wrote dna.csv ({len(rows) - 1} reads)")
