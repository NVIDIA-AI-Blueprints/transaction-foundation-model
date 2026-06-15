#!/usr/bin/env node
"use strict";

/**
 * Loom bootstrapper — the `loom` command installed by `npm i -g @zkailabs.com/loom`.
 *
 * Loom's product (the Node CLI + the Python engine + the datastore) lives in the
 * (private) ZKAI-Network/Loom repo, not in this npm package. This launcher is tiny
 * and dependency-free: on first run it CLONES that repo with the user's own git
 * credentials and runs its `install.sh`, then DELEGATES every invocation to the
 * installed CLI. Access to the private repo IS the license gate (the Gruntwork
 * model) — anyone without git access gets a clear message, not a broken install.
 *
 * Overrides:
 *   LOOM_INSTALL_DIR  where to clone (default: ~/.loom)        -> repo at <dir>/repo
 *   LOOM_REPO_URL     clone source (default: the SSH URL)
 */

const { spawnSync } = require("node:child_process");
const { existsSync } = require("node:fs");
const { join } = require("node:path");
const os = require("node:os");

const INSTALL_DIR = process.env.LOOM_INSTALL_DIR || join(os.homedir(), ".loom");
const REPO_DIR = join(INSTALL_DIR, "repo");
const REPO_URL = process.env.LOOM_REPO_URL || "git@github.com:ZKAI-Network/Loom.git";

const CLI_ENTRY = join(REPO_DIR, "cli", "bin", "loom.js");
const CLI_BUILT = join(REPO_DIR, "cli", "dist", "index.js"); // only present after a build
const VENV_PY = join(REPO_DIR, ".venv", "bin", "python");

function run(cmd, args, opts) {
  const res = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (res.error) {
    console.error(`loom: could not run \`${cmd}\`: ${res.error.message}`);
    if (cmd === "git") console.error("loom: install git (or the Xcode Command Line Tools) and try again.");
    process.exit(1);
  }
  return res.status == null ? 1 : res.status;
}

function isInstalled() {
  return existsSync(CLI_BUILT) && existsSync(VENV_PY);
}

function bootstrap() {
  console.error("loom: first run — setting up Loom (clone the repo + run its installer). One time.\n");

  if (!existsSync(REPO_DIR)) {
    console.error(`loom: cloning ${REPO_URL}\n      -> ${REPO_DIR}`);
    if (run("git", ["clone", REPO_URL, REPO_DIR]) !== 0) {
      console.error(
        "\nloom: clone failed. Loom is a PRIVATE repo (ZKAI-Network/Loom) — installing it\n" +
        "requires git access (an SSH key authorized for the ZKAI-Network org). Ask for\n" +
        "access, or set LOOM_REPO_URL to a source you can reach, then run `loom` again.",
      );
      process.exit(1);
    }
  }

  // Run the repo's agentic-first installer. LOOM_BOOTSTRAP tells it this launcher
  // owns the global `loom` (so it must NOT `npm link` a competing one).
  const code = run("bash", [join(REPO_DIR, "install.sh")], {
    cwd: REPO_DIR,
    env: { ...process.env, LOOM_BOOTSTRAP: "1" },
  });
  if (code !== 0 || !isInstalled()) {
    console.error(
      `\nloom: setup did not finish. Re-run \`loom\`, or install by hand — see ${join(REPO_DIR, "INSTALL.md")}.`,
    );
    process.exit(code || 1);
  }
  console.error("\nloom: setup complete.\n");
}

if (!isInstalled()) bootstrap();

// Delegate to the installed CLI, pointing it at the cloned engine venv unless the
// user already set LOOM_PYTHON.
const env = { ...process.env };
if (!env.LOOM_PYTHON && existsSync(VENV_PY)) env.LOOM_PYTHON = VENV_PY;
process.exit(run(process.execPath, [CLI_ENTRY, ...process.argv.slice(2)], { env }));
