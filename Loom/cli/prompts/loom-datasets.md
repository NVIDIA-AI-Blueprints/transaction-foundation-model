---
description: List ingested Metaflow data objects (the catalog) via the Client API.
argument-hint: <no args — lists the ingested data objects>
---

# /loom-datasets — list ingested data objects (read-only)

Surface the catalog of data objects already registered into Loom's MLOps
interface. To *register* a new source, use `/loom-ingest`. To *profile* what is
inside one, use `/loom-eda`. If there is no datastore yet, use
`/loom-setup-metaflow`.

## 1. Plan — read-only
A pure Client-API read; it never prompts. State: "I'll list the ingested data
objects via the Client API."

## 2. Run — call the `loom_datasets` tool
Call `loom_datasets`. It prints one row per data object: `pathspec · name ·
nrows/schema`. Speak only the interface; never touch Metaflow/raw S3.

## 3. Deliver — summarize the catalog
Summarize how many data objects exist, their names, and row counts. Hand back the
pathspecs so the user can pass one to `/loom-eda` / `/loom-run` / `/loom-validate`.
If the list is empty on a fresh datastore, say so and offer `/loom-ingest`.

If the tool reports a setup failure (exit 2 / datastore unreachable), run
`/loom-doctor` and, if the datastore is down, point at `/loom-setup-metaflow`.
