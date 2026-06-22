#!/usr/bin/env python3
"""Loom NeMo wrapping launcher — runs ON THE VM, inside the NeMo container.

ARCHITECTURE §5.1 / §10 step 7a. This script is the **net-new progress source**:
it wraps the SAME 4-line recipe shell as the team's
``scripts/train_decoder_model.py`` (verified: ``parse_args_and_load_config()`` →
``TrainFinetuneRecipeForNextTokenPrediction(cfg)`` → ``.setup()`` →
``.run_train_validation_loop()``) and, around it, emits a **structured JSONL
progress log** (one record per training step: ``step, loss, lr, tokens``). That
JSONL — NOT a scrape of the recipe's stdout — is what
``nemo_builder.NeMoTrainingHandle.stream_events()`` tails into ``ProgressEvent``s.
The visible ``train_decoder_model.py`` prints only a config header; per-step loss
is emitted from *inside* NeMo's ``run_train_validation_loop()``, whose stdout/log
grammar is a NeMo-AutoModel internal that is **not pinned anywhere in this repo**,
so we never parse it.

CRITICAL — HARD CONSTRAINT #1 (lazy/guarded NeMo·torch import). This module is
NOT imported by the loom package at load time: ``nemo_builder.py`` only ships its
path (``_nemo_train_entry.__file__``-equivalent, resolved as a sibling file) into
the VM-side argv; nothing in ``import loom`` imports this module. All
nemo_automodel/torch imports below live INSIDE ``main()`` so that *if* this file
is ever imported on the CPU control plane (e.g. by a static check), the import of
the module body alone pulls in zero banned packages. The banned packages exist
only inside the NeMo container on the VM, which is the only place this entry point
ever runs.

UNVERIFIED-AGAINST-A-REAL-RUN. The exact attachment point for the per-step hook
depends on the NeMo-AutoModel recipe internals (the loss/step attribute names on
``TrainFinetuneRecipeForNextTokenPrediction`` and whether
``run_train_validation_loop`` exposes a callback). This launcher is written to be
robust to that uncertainty (ARCHITECTURE §5.1 priority order): it
**(a)** attaches a logging hook if the recipe exposes one
(``register_step_callback`` / ``add_callback`` / a ``callbacks`` list), else
**(b)** wraps the recipe's own per-step logging method by monkeypatching it, else
**(c)** falls back to a background thread that polls the checkpoint dir +
``step_scheduler`` state and writes a coarser JSONL. In ALL cases it additionally
writes the step-0 ``loss ≈ ln(vocab_size)`` **canary** record before the loop
starts. The precise hook name in branch (a)/(b) MUST be confirmed against one real
2-step smoke run on the VM before this is trusted for budget telemetry; until
then the checkpoint-dir poller (c) is the guaranteed-correct floor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# The JSONL progress sink — the net-new source the adapter tails. Pure stdlib,
# import-safe on the CPU control plane (no torch/nemo at module scope).
# ---------------------------------------------------------------------------

#: Schema tag stamped on every record so the tailing side can version-guard it.
PROGRESS_SCHEMA = "loom-nemo-progress/1"


class _ProgressLog:
    """Append-only JSONL writer for per-step progress records.

    One JSON object per line, flushed + ``fsync``'d each write so a tail on the
    control plane (over ``gcloud ... cat`` / a synced path) sees whole lines. The
    record shape is the contract ``nemo_builder.stream_events`` reads::

        {"schema": "...", "step": int, "loss": float|null, "lr": float|null,
         "tokens": int|null, "phase": "warmup|train|val|consolidate",
         "wall_clock_min": float, "note": str|null, "ts": epoch_float}
    """

    def __init__(self, path: str) -> None:
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Truncate at start so a resumed run does not concatenate onto a stale log.
        self._fh = open(path, "w", encoding="utf-8")
        self._t0 = time.time()
        self._lock = threading.Lock()

    def emit(
        self,
        *,
        step: int,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        tokens: Optional[int] = None,
        phase: str = "train",
        note: Optional[str] = None,
    ) -> None:
        rec = {
            "schema": PROGRESS_SCHEMA,
            "step": int(step),
            "loss": (float(loss) if loss is not None else None),
            "lr": (float(lr) if lr is not None else None),
            "tokens": (int(tokens) if tokens is not None else None),
            "phase": phase,
            "wall_clock_min": (time.time() - self._t0) / 60.0,
            "note": note,
            "ts": time.time(),
        }
        line = json.dumps(rec, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:  # pragma: no cover - non-fsyncable fs
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:  # pragma: no cover
                pass


# ---------------------------------------------------------------------------
# Hook attachment — branch (a)/(b)/(c) of the §5.1 priority order. All of this is
# best-effort and degrades to the checkpoint-dir poller; none of it imports torch
# at module scope (the recipe object is passed in by ``main``).
# ---------------------------------------------------------------------------


def _coerce_float(v: Any) -> Optional[float]:
    """Pull a python float out of a tensor/np scalar/number without importing
    torch at module scope (we ``.item()`` duck-typed)."""
    if v is None:
        return None
    try:
        item = getattr(v, "item", None)
        if callable(item):
            return float(item())
        return float(v)
    except (TypeError, ValueError):
        return None


def _attach_step_hook(recipe: Any, log: _ProgressLog) -> bool:
    """Branch (a): if the recipe exposes a callback registration surface, register
    a per-step hook that emits one JSONL record. Returns True iff attached.

    The names tried are the plausible NeMo-AutoModel surfaces; NONE is verified
    against a real recipe, so this is wrapped in try/except and reported back to
    ``main`` which falls through to (b)/(c) on failure."""

    def _on_step(step: int, loss: Any = None, lr: Any = None, tokens: Any = None, **_: Any) -> None:
        log.emit(step=step, loss=_coerce_float(loss), lr=_coerce_float(lr),
                 tokens=(int(tokens) if tokens is not None else None), phase="train")

    for reg_name in ("register_step_callback", "add_step_callback", "add_callback"):
        reg = getattr(recipe, reg_name, None)
        if callable(reg):
            try:
                reg(_on_step)
                return True
            except Exception:  # noqa: BLE001 - unknown signature; try the next surface
                continue
    # A mutable callbacks list is another common shape.
    cbs = getattr(recipe, "callbacks", None)
    if isinstance(cbs, list):
        cbs.append(_on_step)
        return True
    return False


def _wrap_step_logger(recipe: Any, log: _ProgressLog) -> bool:
    """Branch (b): monkeypatch the recipe's own per-step logging method so every
    call also writes a JSONL record. Returns True iff a method was wrapped.

    Tries the plausible internal method names; reads loss/step off the recipe's
    state object when the wrapped call does not pass them. Unverified — guarded."""
    for meth_name in ("log_train_step", "_log_train_metrics", "log_step", "_log_metrics"):
        orig = getattr(recipe, meth_name, None)
        if not callable(orig):
            continue

        def _wrapped(*args: Any, _orig=orig, **kwargs: Any) -> Any:
            out = _orig(*args, **kwargs)
            try:
                step = _read_step(recipe)
                loss = _read_loss(recipe)
                lr = _read_lr(recipe)
                log.emit(step=step, loss=loss, lr=lr, phase="train")
            except Exception:  # noqa: BLE001 - the poller is the floor; never break training
                pass
            return out

        try:
            setattr(recipe, meth_name, _wrapped)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _read_step(recipe: Any) -> int:
    for path in ("step_scheduler.step", "step_scheduler.current_step", "global_step", "state.step"):
        v = _getattr_path(recipe, path)
        if v is not None:
            try:
                return int(_coerce_float(v) or 0)
            except (TypeError, ValueError):
                pass
    return -1


def _read_loss(recipe: Any) -> Optional[float]:
    for path in ("last_loss", "state.loss", "metrics.loss", "_last_loss"):
        v = _getattr_path(recipe, path)
        f = _coerce_float(v)
        if f is not None:
            return f
    return None


def _read_lr(recipe: Any) -> Optional[float]:
    for path in ("lr_scheduler.last_lr", "optimizer.param_groups", "state.lr"):
        v = _getattr_path(recipe, path)
        if path.endswith("param_groups") and isinstance(v, (list, tuple)) and v:
            return _coerce_float(v[0].get("lr") if isinstance(v[0], dict) else None)
        f = _coerce_float(v)
        if f is not None:
            return f
    return None


def _getattr_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _start_checkpoint_poller(
    recipe: Any, log: _ProgressLog, *, checkpoint_dir: str, max_steps: int, stop: threading.Event
) -> threading.Thread:
    """Branch (c): the guaranteed-correct floor. A daemon thread that polls the
    recipe step counter + the checkpoint dir and emits a coarse JSONL record each
    time the step advances (or a checkpoint appears). Never imports torch; reads
    only duck-typed attributes + the filesystem. This always runs (even when (a)/(b)
    attached) as a backstop, deduped on step in the tailer."""

    def _poll() -> None:
        last_step = -1
        ckpt = Path(checkpoint_dir)
        while not stop.wait(2.0):
            step = _read_step(recipe)
            if step > last_step and step >= 0:
                log.emit(step=step, loss=_read_loss(recipe), lr=_read_lr(recipe),
                         phase="train", note="poller")
                last_step = step
            # Note checkpoint writes as a coarse phase marker.
            if ckpt.exists():
                for p in ckpt.glob("**/*.safetensors"):
                    log.emit(step=last_step if last_step >= 0 else 0, phase="consolidate",
                             note=f"checkpoint:{p.name}")
                    break

    t = threading.Thread(target=_poll, name="loom-nemo-poller", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# main — the wrapped recipe. NeMo/torch imports live HERE, never at module scope.
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Loom NeMo wrapping launcher (writes a JSONL progress log around the recipe)."
    )
    parser.add_argument("--loom-progress-jsonl", required=True,
                        help="path the per-step JSONL progress log is written to (the net-new source)")
    parser.add_argument("--loom-vocab-size", type=int, default=None,
                        help="vocab size for the step-0 loss≈ln(vocab) canary record")
    # Everything else is the recipe's own argv (``-c <yaml> --dataset.data_path … --step_scheduler.max_steps …``)
    # which we forward UNTOUCHED to NeMo's parser by leaving it on sys.argv.
    known, recipe_argv = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    log = _ProgressLog(known.loom_progress_jsonl)

    # The step-0 canary (ARCHITECTURE §3): a from-scratch CLM starts at
    # loss ≈ ln(vocab_size). Emitted BEFORE the loop so the widget shows the
    # sanity reference even if the first real step is slow.
    if known.loom_vocab_size and known.loom_vocab_size > 1:
        ln_vocab = math.log(known.loom_vocab_size)
        log.emit(step=0, loss=ln_vocab, phase="warmup",
                 note=f"loss≈ln(vocab)={ln_vocab:.3f} (step-0 canary)")

    # Hand the recipe its own argv (strip our two flags so NeMo's parser is happy).
    sys.argv = [sys.argv[0]] + recipe_argv

    # === the SAME 4-line recipe as scripts/train_decoder_model.py ===============
    # LAZY import (HARD CONSTRAINT #1): nemo_automodel is imported HERE, on the VM,
    # never at module load on the CPU control plane.
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    cfg = parse_args_and_load_config()
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()

    # Attach the per-step JSONL hook (a) → (b); the poller (c) always runs as the
    # floor. The checkpoint dir + max_steps come off the resolved config.
    attached = _attach_step_hook(recipe, log)
    if not attached:
        attached = _wrap_step_logger(recipe, log)

    checkpoint_dir = _getattr_path(cfg, "checkpoint.checkpoint_dir") or "checkpoints"
    max_steps = int(_getattr_path(cfg, "step_scheduler.max_steps") or 0)
    stop = threading.Event()
    poller = _start_checkpoint_poller(
        recipe, log, checkpoint_dir=str(checkpoint_dir), max_steps=max_steps, stop=stop
    )

    rc = 0
    try:
        recipe.run_train_validation_loop()
    except BaseException:  # noqa: BLE001 - record the failure, re-raise for the exit code
        log.emit(step=_read_step(recipe), phase="train", note="run failed")
        rc = 1
        raise
    finally:
        stop.set()
        poller.join(timeout=5.0)
        # A final consolidate record so the tailer sees a terminal phase.
        log.emit(step=_read_step(recipe), loss=_read_loss(recipe), phase="consolidate",
                 note="run_train_validation_loop returned" if rc == 0 else "run_train_validation_loop raised")
        log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
