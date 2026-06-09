# Hosting the Loom gateway

This directory containerizes `loom proxy serve` — the **Loom gateway**, Loom's
primary moat-capture seam. The gateway is a thin **Anthropic-passthrough** that:

1. authenticates callers by a **Loom-issued key** (the `LOOM_API_KEYS` allowlist),
2. injects Loom's system prompt,
3. forwards to the real Anthropic Messages API using a **server-side vendor key**
   (`ANTHROPIC_API_KEY`) the caller never sees, and
4. logs every call centrally to one JSONL corpus (the moat).

Secrets are **never** baked into the image: the real vendor key and the caller
allowlist are supplied at **runtime** from the environment / orchestrator
secrets. The image carries only code.

Once the gateway is hosted and clients point `LOOM_API_BASE` at it, `loom-proxy`
becomes the **default** model provider (see [DEFAULT-ONCE-HOSTED](#default-once-hosted)).

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Containerizes `loom proxy serve` (slim Python, `/healthz` HEALTHCHECK). |
| `docker-compose.yml` | Single-service, env-file-driven local/VM hosting. |
| `.env.example` | The runtime env to copy to `.env` (real keys, gitignored). |
| `k8s.yaml` | Deployment + Service + readiness probe; creds via a referenced Secret. |

## 1. Build the image

The build context is the **repo root** (the Dockerfile `COPY . /app` needs the
whole package, including the static `flows` package):

```bash
docker build -f deploy/gateway/Dockerfile -t loom-gateway:latest .
```

## 2. Run with Docker Compose

```bash
cp deploy/gateway/.env.example deploy/gateway/.env   # then fill in REAL keys
docker compose -f deploy/gateway/docker-compose.yml up --build
```

Set in `.env` (never commit it):

- `ANTHROPIC_API_KEY` — the **real** vendor key the gateway forwards with (stays
  on the server).
- `LOOM_API_KEYS` — comma-separated allowlist of Loom keys callers must present.
- `LOOM_PROXY_LOG_PATH` — where the JSONL moat corpus is appended (point it at the
  mounted volume so it survives restarts).

Verify it is up:

```bash
curl -s http://localhost:8088/healthz     # -> {"status":"ok","service":"loom-proxy"}
```

## 3. Deploy to Kubernetes

Create the secret **out-of-band** (so no key material lands in git or the
manifest), then apply:

```bash
kubectl create secret generic loom-gateway-secrets \
  --namespace loom \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --from-literal=LOOM_API_KEYS=loom-key-1,loom-key-2

kubectl apply -f deploy/gateway/k8s.yaml
```

`k8s.yaml` pulls those values via `envFrom: secretRef` — the manifest references
the Secret by name and never inlines the keys. The readiness/liveness probes hit
`/healthz`. Push the image to a registry your cluster can pull and update the
`image:` field before applying.

## 4. Point clients at the gateway

On any client, set the gateway URL and the caller's Loom key (one of the
allowlisted keys); the client never holds the real vendor key:

```bash
export LOOM_API_BASE="https://gateway.example.com"   # your hosted URL
export LOOM_API_KEY="loom-key-1"                      # one allowlisted key
loom run --data ./task --goal "..." --metric "..."    # loom-proxy is now the default
```

## DEFAULT-ONCE-HOSTED

When a **hosted** gateway is detected — `LOOM_API_BASE` set to a non-loopback URL,
**or** the `LOOM_PROXY_DEFAULT` env truthy — and the user has **not** explicitly
chosen a model provider, Loom defaults `code_provider` / `feedback_provider` to
`loom-proxy` (so a hosted deployment captures the moat corpus by default). An
explicit choice always wins:

```
explicit --code/feedback/model-provider or LOOM_CODE/FEEDBACK_PROVIDER
   →  hosted gateway (loom-proxy)
   →  anthropic-api (the historical default)
```

A loopback `LOOM_API_BASE` (`127.0.0.1` / `localhost` / `::1` / `0.0.0.0`, the
default included) is **local dev, not hosting** — it does not flip the default.
A tenant that does not want its traces collected sets an explicit BYO-key
provider (e.g. `--model-provider anthropic-api`); its bulk data is protected the
same way either way.
