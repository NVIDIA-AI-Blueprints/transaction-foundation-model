"""Generate a synthetic music-listening event log — the lead example in TOKENIZATION.md.

3,000 "user played a track" events with the column shapes you meet everywhere:
an entity (user_id), categoricals of varying cardinality (track/artist/genre/city),
a timestamp (ts), and a continuous quantity (minutes_played). Reproducible (seeded).

    python examples/listens.py            # writes listens.csv
    loom ingest --in listens.csv --entity user_id --event listen --name music-listens
"""
import csv, random, datetime as dt

random.seed(7)
genres = ["pop", "rock", "jazz", "hiphop", "classical", "electronic"]
artists = [f"artist_{i:02d}" for i in range(40)]
tracks = [f"trk_{i:04d}" for i in range(200)]
cities = [f"city_{i:03d}" for i in range(800)]
base = dt.datetime(2026, 3, 1, 8, 0, 0)

rows = [["user_id", "track", "artist", "genre", "city", "ts", "minutes_played"]]
for _ in range(3000):
    rows.append([
        f"u{random.randint(0, 49):03d}",
        random.choice(tracks), random.choice(artists),
        random.choice(genres), random.choice(cities),
        (base + dt.timedelta(minutes=random.randint(0, 7 * 24 * 60))).isoformat(),
        round(random.uniform(0.3, 5.0), 2),
    ])
csv.writer(open("listens.csv", "w", newline="")).writerows(rows)
print(f"wrote listens.csv ({len(rows) - 1} rows)")
