"""Generate a faithful sample of the TabFormer credit-card transactions schema.

This is for the demo in `demo-tabformer.md`. Use it when the real ~2.2 GB
`card_transaction.v1.csv` isn't on hand — it reproduces TabFormer's exact 15
columns and documented quirks (a `$`-prefixed Amount string with thousands
commas, a 19-digit integer-string Merchant Name, the Year/Month/Day/Time split,
the rare `Is Fraud?` label, online rows defaulting to ONLINE/XX/000). Seeded, so
the numbers in the demo doc (propose vocab ≈ 1055; preset financial = 6251) are
exactly reproducible.

    python examples/tabformer_sample.py        # writes tf.csv (20,000 rows)
    loom ingest --in tf.csv --entity User --event transaction --target "Is Fraud?" --name tabformer

To run the demo on the REAL data instead, sample your downloaded CSV:
    awk 'NR==1 || rand()<0.008' data/TabFormer/card_transaction.v1.csv > tf.csv
"""
import csv, random

random.seed(42)
mccs = [5411, 4111, 5912, 5812, 5541, 4814, 7011, 5311, 5999, 4900] + [3000 + i for i in range(100)]
chips = ["Swipe Transaction", "Chip Transaction", "Online Transaction"]
states = ["CA", "NY", "TX", "FL", "IL", "WA", "MA", "GA", "PA", "OH", "NC", "MI", "AZ", "CO"]
cities = [f"City{i:04d}" for i in range(2000)]
header = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant Name",
          "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"]

rows = [header]
for _ in range(20000):
    online = random.random() < 0.10
    amt = round(random.lognormvariate(3.2, 1.1), 2)
    rows.append([
        random.randint(0, 1999), random.randint(0, 9), random.randint(2002, 2019),
        random.randint(1, 12), random.randint(1, 28),
        f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
        f"${amt:,.2f}",                                   # "$1,234.56" — the currency-string quirk
        random.choice(chips),
        str(random.randint(10 ** 18, 10 ** 19 - 1)),      # 19-digit synthetic merchant id
        "ONLINE" if online else random.choice(cities),
        "XX" if online else random.choice(states),
        "000" if online else f"{random.randint(10000, 99999)}",
        random.choice(mccs),
        random.choice(["", "", "", "", "Insufficient Balance", "Bad PIN"]),
        "Yes" if random.random() < 0.01 else "No",
    ])
csv.writer(open("tf.csv", "w", newline="")).writerows(rows)
print(f"wrote tf.csv ({len(rows) - 1} rows, TabFormer schema)")
