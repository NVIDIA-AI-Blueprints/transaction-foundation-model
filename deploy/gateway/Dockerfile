# Loom gateway image — containerizes `loom proxy serve` (the moat capture seam).
#
# The gateway is a thin Anthropic-passthrough: it authenticates callers by a
# Loom-issued key (the LOOM_API_KEYS allowlist), injects Loom's system prompt,
# forwards to the real Anthropic API with a SERVER-SIDE vendor key the caller
# never sees, and logs every call centrally (the moat corpus).
#
# Secrets are NEVER baked into the image. The real vendor key (ANTHROPIC_API_KEY)
# and the caller allowlist (LOOM_API_KEY / LOOM_API_KEYS) are supplied at RUNTIME
# from the environment / orchestrator secrets (see docker-compose.yml + k8s.yaml).
# The image carries only code.

FROM python:3.12-slim

# - Unbuffered stdout/stderr so logs stream out of the container promptly.
# - No .pyc clutter in the image layer.
# - pip: no cache (smaller image), no version-check chatter.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the repo (which pulls starlette + uvicorn + httpx via Loom's deps, so
# `loom proxy serve` has its server stack). Copy the whole repo so the editable
# install resolves the package + the static `flows` package the spec ships.
# Build context is the repo root: `docker build -f deploy/gateway/Dockerfile .`
COPY . /app
# Slim gateway: `loom proxy serve` imports only loom.cli / loom.config / loom.proxy
# — NONE of the heavy ML core (aideml/torch/metaflow/pandas/sklearn). So install
# loom WITHOUT its deps, then add just the gateway server stack + loom's light
# config deps. Keeps the image small and the build fast (no torch/aideml pull).
RUN pip install --no-deps /app \
 && pip install "starlette>=0.37" "uvicorn>=0.30" "httpx>=0.27" "omegaconf>=2.3" "python-dotenv>=1.0"

# Run as a non-root user (defense in depth — the gateway needs no root).
RUN useradd --create-home --uid 10001 loom \
    && chown -R loom:loom /app
USER loom

# Document the listen port (the real bind happens in CMD via --port).
EXPOSE 8088

# Liveness: hit the unauthenticated /healthz the server exposes. Uses python
# (already present) rather than adding curl/wget to the image. Shell form so the
# probe is one robust line; the bind port is fixed at 8088 (matching CMD).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8088/healthz', timeout=4).status==200 else 1)"

# Bind on 0.0.0.0 so the orchestrator can route to the container; auth + the
# server-side vendor key still gate every real call. ANTHROPIC_API_KEY and
# LOOM_API_KEYS come from the runtime environment, never from this image.
CMD ["loom", "proxy", "serve", "--host", "0.0.0.0", "--port", "8088"]
