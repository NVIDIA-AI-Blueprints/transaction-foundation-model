"""Configuration for Loom.

:class:`LoomConfig` is the single source of truth for which providers to use and
how to route models and budget a search. It is loaded from (in increasing order
of precedence): an optional YAML file, a ``.env`` file / process environment.

Secrets are NEVER hardcoded or stored on the config object. API keys and
endpoints (``NVIDIA_API_KEY``, ``OPENAI_BASE_URL``, ``ANTHROPIC_API_KEY``,
``METAFLOW_*``) are read from the environment at the point of use; this module
only records *which* environment-derived values (like the model names and the
NIM base URL) influence routing, never the key material itself.

Dependency-light: ``python-dotenv`` and ``omegaconf``/``PyYAML`` are imported
lazily inside :meth:`LoomConfig.load`, so importing this module never requires
those optional packages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

# Default model routing. Model *names* are safe to default; endpoints/keys are
# always taken from the environment and never baked in here.
_DEFAULT_CODE_MODEL = "claude-sonnet-4-5"
_DEFAULT_FEEDBACK_MODEL = "claude-sonnet-4-5"


@dataclass
class BudgetConfig:
    """Search budget knobs passed through to the search provider.

    Attributes:
        steps: Number of search steps (agent iterations) to run.
        num_drafts: Number of initial draft solutions before improving.
        debug_prob: Probability of choosing to debug a buggy node.
        max_debug_depth: Maximum depth of consecutive debug attempts.
    """

    steps: int = 10
    num_drafts: int = 3
    debug_prob: float = 0.5
    max_debug_depth: int = 3


@dataclass
class LoomConfig:
    """Top-level Loom configuration.

    Attributes:
        search_provider: Name of the search ("brain") provider. Default
            ``"aide"``.
        mlops_provider: Name of the execution ("muscle") provider. Default
            ``"metaflow"`` (use ``"local"`` for a Metaflow-free dev path).
        metaflow_profile: Metaflow profile name (env ``METAFLOW_PROFILE``);
            ``None`` uses Metaflow's default profile. Lets a tenant point Loom
            at their own Metaflow endpoint (BYO perimeter).
        code_model: Model used to generate solution code.
        feedback_model: Model used to review/score executed solutions.
        code_provider: Name of the model ("LLM backend") provider for the code
            role -- which model and how it is authenticated. Default
            ``"anthropic-api"`` (native Claude), preserving historical behavior.
        feedback_provider: Name of the model provider for the feedback/judge
            role. Default ``"anthropic-api"``. The judge always uses tool calling,
            so this provider's resolved route must be judge-capable.
        nim_base_url: OpenAI-compatible base URL for model routing (env
            ``OPENAI_BASE_URL``; e.g. an NVIDIA NIM endpoint). The matching API
            key is read from the environment at call time, never stored here.
        model_base_url: OpenAI-compatible base URL for the generic
            ``openai-compat`` model provider (env ``LOOM_MODEL_BASE_URL``; e.g. a
            LiteLLM/vLLM/Ollama endpoint). The matching API key is read from the
            environment at call time, never stored here.
        budget: Search budget knobs (see :class:`BudgetConfig`).
        corpus_path: Path to the JSONL corpus the controller appends node
            records to.
        learnings_path: Path to the JSONL learnings flywheel the controller
            appends one command-level rollout record to per run. This is the
            command-level rollup (one row per ``run_loom`` call) that sits above
            the per-node ``corpus_path`` capture; both are anchored absolute at
            load time so they survive a provider ``chdir`` into an ephemeral
            workspace.
        tenant: Logical tenant this config runs under.
        owned_by: IP owner tag applied to corpus records. ``"general"`` marks
            records as usable by a cross-tenant moat model; any other value
            tags them as tenant-owned and excludes them from the general set.
    """

    search_provider: str = "aide"
    mlops_provider: str = "metaflow"
    metaflow_profile: str | None = None
    code_model: str = _DEFAULT_CODE_MODEL
    feedback_model: str = _DEFAULT_FEEDBACK_MODEL
    code_provider: str = "anthropic-api"
    feedback_provider: str = "anthropic-api"
    nim_base_url: str | None = None
    model_base_url: str | None = None
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    corpus_path: str = "corpus/nodes.jsonl"
    learnings_path: str = "learnings/rollouts.jsonl"
    tenant: str = "default"
    owned_by: str = "general"

    @classmethod
    def load(
        cls,
        yaml_path: str | None = None,
        env: Mapping[str, str] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "LoomConfig":
        """Build a :class:`LoomConfig` from YAML, ``.env``/env, and overrides.

        Precedence (lowest to highest): dataclass defaults < YAML file < the
        process environment (with ``.env`` loaded into it if present) <
        explicit ``overrides``.

        Args:
            yaml_path: Optional path to a YAML config file. Ignored if the file
                does not exist. Requires ``omegaconf`` or ``PyYAML`` (imported
                lazily) only when a readable file is supplied.
            env: Environment mapping to read from. Defaults to ``os.environ``.
                A ``.env`` file in the current directory is loaded into the
                process environment first (best-effort, via ``python-dotenv``).
            overrides: Explicit field overrides applied last.

        Returns:
            A fully populated :class:`LoomConfig`. No secret material is read
            onto the returned object.
        """
        # Best-effort: load a local .env into the process environment so that
        # env-driven settings below pick it up. Never fails the import path.
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except Exception:
                pass
            env = os.environ

        cfg = cls()

        # 1) YAML file (optional, lowest precedence above defaults).
        if yaml_path and os.path.isfile(yaml_path):
            cfg = cls._apply_mapping(cfg, cls._read_yaml(yaml_path))

        # 2) Environment / .env.
        cfg = cls._apply_mapping(cfg, cls._from_env(env))

        # 3) Explicit overrides (highest precedence).
        if overrides:
            cfg = cls._apply_mapping(cfg, overrides)

        # Anchor the corpus path to an ABSOLUTE location at load time (the launch
        # cwd), BEFORE any execution provider chdir's into an ephemeral workspace.
        # Otherwise a relative corpus_path resolves against that workspace at
        # write time and the node records are written into — and deleted with —
        # the workspace, and the post-run leaderboard read finds nothing.
        if cfg.corpus_path and not os.path.isabs(cfg.corpus_path):
            cfg = replace(cfg, corpus_path=os.path.abspath(cfg.corpus_path))

        # Anchor the learnings path absolute for the same reason as corpus_path:
        # the controller appends the command-level rollout AFTER the execution
        # provider has chdir'd into (and will tear down) its ephemeral workspace,
        # so a relative path would write the moat fuel into the doomed workspace.
        if cfg.learnings_path and not os.path.isabs(cfg.learnings_path):
            cfg = replace(cfg, learnings_path=os.path.abspath(cfg.learnings_path))

        return cfg

    @staticmethod
    def _read_yaml(yaml_path: str) -> dict[str, Any]:
        """Read a YAML config file into a plain dict (lazy YAML import)."""
        try:
            from omegaconf import OmegaConf

            return dict(OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True))  # type: ignore[arg-type]
        except Exception:
            import yaml  # PyYAML fallback

            with open(yaml_path, "r", encoding="utf-8") as fh:
                return dict(yaml.safe_load(fh) or {})

    @staticmethod
    def _from_env(env: Mapping[str, str]) -> dict[str, Any]:
        """Extract recognized settings from an environment mapping.

        Only non-secret routing/selection values are pulled. API key material
        (``NVIDIA_API_KEY``, ``ANTHROPIC_API_KEY``, ...) is intentionally NOT
        read here; it is consumed directly from the environment by the adapters
        that need it.
        """
        out: dict[str, Any] = {}
        budget: dict[str, Any] = {}

        if (v := env.get("LOOM_SEARCH_PROVIDER")) is not None:
            out["search_provider"] = v
        if (v := env.get("LOOM_MLOPS_PROVIDER")) is not None:
            out["mlops_provider"] = v
        if (v := env.get("METAFLOW_PROFILE")) is not None:
            out["metaflow_profile"] = v
        if (v := env.get("LOOM_CODE_MODEL")) is not None:
            out["code_model"] = v
        if (v := env.get("LOOM_FEEDBACK_MODEL")) is not None:
            out["feedback_model"] = v
        if (v := env.get("LOOM_CODE_PROVIDER")) is not None:
            out["code_provider"] = v
        if (v := env.get("LOOM_FEEDBACK_PROVIDER")) is not None:
            out["feedback_provider"] = v
        if (v := env.get("OPENAI_BASE_URL")) is not None:
            out["nim_base_url"] = v
        if (v := env.get("LOOM_MODEL_BASE_URL")) is not None:
            out["model_base_url"] = v
        if (v := env.get("LOOM_CORPUS_PATH")) is not None:
            out["corpus_path"] = v
        if (v := env.get("LOOM_LEARNINGS_PATH")) is not None:
            out["learnings_path"] = v
        if (v := env.get("LOOM_TENANT")) is not None:
            out["tenant"] = v
        if (v := env.get("LOOM_OWNED_BY")) is not None:
            out["owned_by"] = v

        if (v := env.get("LOOM_BUDGET_STEPS")) is not None:
            budget["steps"] = int(v)
        if (v := env.get("LOOM_BUDGET_NUM_DRAFTS")) is not None:
            budget["num_drafts"] = int(v)
        if (v := env.get("LOOM_BUDGET_DEBUG_PROB")) is not None:
            budget["debug_prob"] = float(v)
        if (v := env.get("LOOM_BUDGET_MAX_DEBUG_DEPTH")) is not None:
            budget["max_debug_depth"] = int(v)

        if budget:
            out["budget"] = budget
        return out

    @staticmethod
    def _apply_mapping(cfg: "LoomConfig", data: Mapping[str, Any]) -> "LoomConfig":
        """Return a copy of ``cfg`` updated with values from ``data``.

        Unknown keys are ignored. The ``budget`` key may be a mapping that is
        merged field-wise into the existing :class:`BudgetConfig`.
        """
        if not data:
            return cfg

        field_names = {f.name for f in cfg.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        updates: dict[str, Any] = {}

        for key, value in data.items():
            if key not in field_names:
                continue
            if key == "budget" and isinstance(value, Mapping):
                budget_fields = {
                    f.name for f in cfg.budget.__dataclass_fields__.values()  # type: ignore[attr-defined]
                }
                budget_updates = {
                    k: v for k, v in value.items() if k in budget_fields
                }
                updates["budget"] = replace(cfg.budget, **budget_updates)
            else:
                updates[key] = value

        return replace(cfg, **updates)


__all__ = ["LoomConfig", "BudgetConfig"]
