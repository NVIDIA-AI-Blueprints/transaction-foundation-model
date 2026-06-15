# @loom/cli — the `loom` agentic CLI

`loom` is the standalone **agentic data-science operator**: natural language in,
the agent picks the right **Loom verb**, runs it, reads the structured `VERDICT` /
summary, and composes the next step. The agent loop, tool-calling, multi-provider
LLM, sessions, and TUI are hosted on the [Pi](https://github.com/earendil-works/pi)
coding-agent harness. Loom's Python engine (AIDE search +
Metaflow flows + providers + telemetry) is **unchanged** — this package is a thin,
swappable interface layer: a launcher, one extension that maps the 15 verbs to Pi
tools, and a persona/branding bundle.

## Install

```bash
# 1) The Python engine (unchanged) must be importable as `python -m loom`.
#    On this box it lives in the Loom venv:
#      /Users/anub/Work/Loom/.venv/bin/python -m loom verbs --json   # the 15-verb manifest

# 2) Build the agentic CLI:
cd /Users/anub/Work/Loom/cli
npm install
npm run build
```

Requires **Node ≥ 22.19**. The Pi dependency is pinned exactly
(`@earendil-works/pi-coding-agent@0.79.0` + `@earendil-works/pi-ai@0.79.0`); the
launcher tolerates the legacy `@mariozechner` scope as a fallback.

## Usage

```bash
loom                          # open the Loom agent (interactive TUI)
loom "explore my data"        # start with a one-shot goal in natural language
loom eda --dataset IngestDataset/123     # jump straight to a verb workflow (/loom-eda)
loom --help
loom --version
```

Run from the repo without a global install via `node dist/index.js …` after
`npm run build`. Inside the REPL, every verb has a `/loom-<verb>` slash-command
(e.g. `/loom-eda --dataset X`) that runs that verb's full workflow
(intake → plan/tier → run → verify → deliver), and the agent can also pick a verb
from a natural-language request.

### Picking the agent's model

The agent's own reasoning uses a pi-ai provider, chosen with `/model` in the REPL
or `--model <provider/id>`, with env-key fallback (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) or Pi's `/login` OAuth. This is a **separate
channel** from the engine's optimization providers (the moat): `loom_run` drives
the Python engine's own `--code-provider` / `--feedback-provider`, which the agent
never sees or proxies.

## `LOOM_PYTHON` — the engine interpreter

Each verb tool shells out to **`python -m loom <verb> --json`** using a resolved
interpreter. By default this is the Loom venv on this box:

```
/Users/anub/Work/Loom/.venv/bin/python
```

Override it by exporting `LOOM_PYTHON`:

```bash
export LOOM_PYTHON=/path/to/your/loom/venv/bin/python
loom
```

Invoking the engine as `python -m loom` (rather than a `loom` console script)
avoids any PATH collision with the `loom` agent command and needs no console-script
install. The 0/1/2 exit contract is preserved end-to-end (0 ok, 1 runtime failure
with a well-formed result, 2 setup-or-bad-args → run `/loom-doctor`).

## The 15 verbs

| Lifecycle | Verbs | Tier |
|---|---|---|
| Understand | `doctor` `datasets` `eda` `validate` `viz` `report` `ops` | read-only |
| Build | `ingest` `features` `pipeline` | workspace-write |
| Operate | `run` | expensive (notify) |
| Gated | `deploy` `train` `collab` `skillopt` | irreversible — explicit confirm, never auto-fired |

The four gated verbs are **registered but kept out of the model-offered set** (no
`promptSnippet`), so the model only reaches them via an explicit `/loom-<verb>`
command, and the per-call approval gate requires an interactive confirmation before
the irreversible action runs. Read-only/workspace-write verbs run freely; the
expensive `run` notifies before firing.

## Layout

```
cli/
  bin/loom.js              # Node-version gate -> dist/index.js
  src/
    index.ts               # arg dispatch + verb->/loom-<verb> routing + launch
    manifest.ts            # `python -m loom verbs --json` -> TypeBox tools + tiers + gate
    pi/{runtime,launch,pi-cli-wrapper,settings}.ts   # spawn Pi with our flags + env
    branding/{logo,header}.ts                        # the LOOM banner (session_start setHeader)
    bootstrap/sync.ts      # hash-tracked asset sync into the Pi home
  extensions/loom-tools.ts # default-export factory: registerTool per verb + gate + header
  prompts/loom-*.md        # the /loom-<verb> workflow templates (--prompt-template)
  home/                    # the branded Pi agent dir (PI_CODING_AGENT_DIR)
    SYSTEM.md              # the Loom persona + verb catalog + conventions (--system-prompt)
    themes/loom.json       # the LOOM TUI theme (--theme)
    settings.json          # forced theme + quietStartup + collapseChangelog
  assets/home/themes/loom.json   # canonical bundled theme (synced into home/)
```
