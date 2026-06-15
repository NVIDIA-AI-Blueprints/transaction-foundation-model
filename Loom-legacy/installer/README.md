# @zkailabs.com/loom

The installer for **Loom** — an agentic CLI for data science.

```bash
npm install -g @zkailabs.com/loom
loom
```

This package is a tiny bootstrapper. Loom's product (the CLI, the Python engine,
the local datastore) lives in the **private** `ZKAI-Network/Loom` repo. On first
run, `loom`:

1. clones that repo (with **your own git credentials**) into `~/.loom/repo`,
2. runs its installer (`install.sh`) — the Python engine, the Node CLI build, and
   the local Metaflow datastore,
3. delegates every command to the installed CLI from then on.

**Access is the gate.** Loom is private, so installing it requires git access to
`ZKAI-Network/Loom` (an SSH key authorized for the org). Without access the clone
fails with a clear message — nothing here exposes the product. Need access? Reach
out to ZKAI Labs.

### Overrides

| Env var | Default | Meaning |
|---|---|---|
| `LOOM_INSTALL_DIR` | `~/.loom` | where the repo is cloned (`<dir>/repo`) |
| `LOOM_REPO_URL` | `git@github.com:ZKAI-Network/Loom.git` | clone source |

Update the product later with `loom update`; update this installer with
`npm i -g @zkailabs.com/loom@latest`.
