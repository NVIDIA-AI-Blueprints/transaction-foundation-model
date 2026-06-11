---
description: Launch an interactive GPU notebook (JupyterLab in the NeMo container) on an on-demand cloud GPU and forward it to the laptop. EXPENSIVE — spends real GPU while running; confirm before the live launch (--dry-run previews for free).
argument-hint: "[--dry-run] [--gpu modal | modal://<app>] [--no-datastore]"
---

# /loom-notebook — interactive GPU notebook (EXPENSIVE — gate on cost)

Give the user a GPU-backed **JupyterLab**, running in the team's standard NeMo
container on an **on-demand cloud GPU (Modal)**, forwarded to their laptop — with
their governed Loom datastore connected so the Metaflow Client API works in the
notebook. Loom **orchestrates** the remote notebook; it is not a notebook IDE/host.
The GPU burst is **ephemeral** — it bills only while running and disappears when the
session ends.

Notebook request: $@

## 1. Intake — decide dry-run vs live launch
- **`--dry-run`** (and `--json`) **previews for free** — it prints exactly what would
  launch (app, GPU=H100, the NeMo image, the forwarded port, datastore) and spends
  nothing. Use it first if the user is just checking, or if anything is unclear.
- The **live launch spends real GPU** for the whole session. Only the `modal` /
  `modal://<app>` target is supported; any other target refuses cleanly.

## 2. Plan — EXPENSIVE, confirm before the live launch (never auto-fire)
A live `loom notebook` provisions a real cloud GPU and bills until stopped, so it is
the **expensive** tier: **propose, then stop for explicit confirmation** before the
spending launch. Surface:
- **What runs** — JupyterLab in the NeMo container on an on-demand GPU, forwarded to
  the laptop; the datastore env forwarded unless `--no-datastore`.
- **Cost shape** — a real GPU bills for the whole session; the **first launch also
  pulls the team's container** (a few minutes). It stops when the user stops it.
- **Free preview** — offer `--dry-run` first.

Do not start the live launch until the user confirms after seeing the plan + cost.

## 3. Run — launch the notebook
Run the `notebook` verb with the user's flags (default to `--dry-run` when only a
preview was asked for; pass `--gpu` / `--no-datastore` through). On a live launch it
prints a notebook URL once the container is up — **surface that URL** and tell the
user to keep the session terminal open, that the datastore/Client API is available
inside, and that **stopping the session stops the GPU billing**.

## 4. Verify
- **Dry run** → confirm the printed plan (app / GPU / image / port / datastore) and
  remind them to drop `--dry-run` (with a Modal token set) to actually launch.
- **Live** → confirm a notebook URL was produced; if a non-Modal target was given it
  refuses with an actionable message; if `modal` isn't installed/authenticated it
  refuses with install/auth steps.

## 5. Deliver — narrate, hand back the URL
- Lead with whether this was a **free dry-run preview** or a **live GPU launch**, then
  the notebook URL (live) or the plan (dry-run).
- Inside the notebook, the governed datastore is reachable via the Metaflow Client
  API; exploration can flow back into the tracked lifecycle (`/loom-eda`,
  `/loom-features`, `/loom-validate`, `/loom-train`).
- Remind: the GPU is ephemeral — **stop the session when done** to stop billing.
