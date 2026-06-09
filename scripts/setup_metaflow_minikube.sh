#!/usr/bin/env bash
#
# setup_metaflow_minikube.sh -- stand up a LOCAL-DEV Metaflow datastore on
# minikube, backed by an in-cluster minio (S3-compatible) object store, so the
# `loom` CLI's `metaflow` provider has a datastore to read/write through the
# Metaflow Client API.
#
# This is the verified local recipe (macOS, docker driver via colima), made
# IDEMPOTENT and DEFENSIVE: every prerequisite is detected before it is
# installed, every cluster object is applied with `kubectl apply` (re-runnable),
# and the bucket is created only if absent. Re-running the script is safe.
#
# It DOES NOT push anything off-box and DOES NOT deploy. The only credentials it
# writes are the LOCAL-DEV minio creds that already live in the manifest
# (minioadmin / minioadmin123) -- those are NOT secrets; they exist solely inside
# this laptop minikube cluster. The whole cluster is reversible with
# `minikube delete`.
#
# Verify after running:   source .env.metaflow && loom doctor
#
# This script is GATED behind the /loom-setup skill (EXPENSIVE/MUTATE tier): it
# installs software + starts a local cluster. Read that skill before running.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants -- the verified recipe's fixed values. The minio creds are
# local-dev only (they match skills/loom-setup-metaflow/manifests/minio.yaml).
# ---------------------------------------------------------------------------
NAMESPACE="loom"
MINIO_USER="minioadmin"
MINIO_PASSWORD="minioadmin123"
# The Metaflow datastore root. The bucket is the first path component
# ("metaflow"); the sysroot points one level deeper ("metaflow/metaflow").
DATASTORE_BUCKET="metaflow"
DATASTORE_SYSROOT="s3://metaflow/metaflow"
S3_ENDPOINT_URL="http://localhost:9000"

# Resolve repo paths relative to this script (so it runs from anywhere).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MINIO_MANIFEST="${REPO_ROOT}/skills/loom-setup-metaflow/manifests/minio.yaml"
ENV_FILE="${REPO_ROOT}/.env.metaflow"

# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Install a brew formula/cask only if the command is missing (macOS).
ensure_tool() {
  # $1 = command to look for, $2 = brew install spec (formula or "--cask name").
  local cmd="$1"; shift
  local brew_spec="$*"
  if have "${cmd}"; then
    info "found ${cmd} ($(command -v "${cmd}"))"
    return 0
  fi
  info "missing ${cmd}"
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: ${cmd} is not installed and auto-install is macOS/brew-only." >&2
    echo "       install ${cmd} with your package manager and re-run." >&2
    exit 1
  fi
  if ! have brew; then
    echo "error: Homebrew is required to install ${cmd} but 'brew' was not found." >&2
    echo "       install Homebrew (https://brew.sh) and re-run." >&2
    exit 1
  fi
  info "installing ${cmd} via: brew install ${brew_spec}"
  # shellcheck disable=SC2086 -- brew_spec may carry a --cask flag word.
  brew install ${brew_spec}
}

# ---------------------------------------------------------------------------
# 0) Sanity: the minio manifest must be present.
# ---------------------------------------------------------------------------
step "Checking the minio manifest"
if [[ ! -f "${MINIO_MANIFEST}" ]]; then
  echo "error: minio manifest not found at ${MINIO_MANIFEST}" >&2
  exit 1
fi
info "manifest: ${MINIO_MANIFEST}"

# ---------------------------------------------------------------------------
# 1) Prerequisites -- detect before install (macOS: brew). On a Mac the docker
#    driver runs via colima; we ensure colima, minikube, kubectl, and aws-cli.
# ---------------------------------------------------------------------------
step "Ensuring prerequisites (detect-before-install)"
if [[ "$(uname -s)" == "Darwin" ]]; then
  ensure_tool colima colima
  ensure_tool docker docker
fi
ensure_tool minikube minikube
ensure_tool kubectl kubectl
ensure_tool aws awscli

# Start colima (the docker runtime) on macOS if it is installed but not running.
if [[ "$(uname -s)" == "Darwin" ]] && have colima; then
  if colima status >/dev/null 2>&1; then
    info "colima already running"
  else
    info "starting colima (docker runtime)"
    colima start
  fi
fi

# ---------------------------------------------------------------------------
# 2) minikube -- start it only if the cluster is not already running.
# ---------------------------------------------------------------------------
step "Starting minikube (docker driver)"
if minikube status >/dev/null 2>&1; then
  info "minikube already running"
else
  info "minikube start --driver=docker"
  minikube start --driver=docker
fi

# ---------------------------------------------------------------------------
# 3) Namespace + 4) minio -- both applied idempotently. The manifest already
#    declares the `loom` namespace, so `kubectl apply -f` is fully re-runnable.
# ---------------------------------------------------------------------------
step "Creating namespace '${NAMESPACE}' (idempotent)"
kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 \
  && info "namespace '${NAMESPACE}' already exists" \
  || kubectl create namespace "${NAMESPACE}"

step "Applying the minio manifest (idempotent)"
kubectl apply -n "${NAMESPACE}" -f "${MINIO_MANIFEST}"

# ---------------------------------------------------------------------------
# 5) Wait for minio to be ready.
# ---------------------------------------------------------------------------
step "Waiting for minio to become ready"
if kubectl wait -n "${NAMESPACE}" --for=condition=available \
    --timeout=180s deployment/minio; then
  info "minio deployment is available"
else
  echo "error: minio did not become ready within the timeout." >&2
  echo "       inspect with: kubectl -n ${NAMESPACE} get pods,events" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 6) Create the datastore bucket once (idempotent). We run the minio client
#    INSIDE the cluster (a throwaway pod talking to the in-cluster minio
#    service) so the host needs no extra S3 tooling. `mc mb -p` is a no-op when
#    the bucket already exists, making this safe to re-run.
# ---------------------------------------------------------------------------
step "Ensuring the datastore bucket '${DATASTORE_BUCKET}' exists (idempotent)"
if kubectl run minio-mc-setup \
    --namespace "${NAMESPACE}" \
    --rm -i --restart=Never \
    --image=minio/mc:latest \
    --command -- /bin/sh -c "
      set -e
      mc alias set local http://minio:9000 '${MINIO_USER}' '${MINIO_PASSWORD}' >/dev/null
      mc mb -p local/${DATASTORE_BUCKET} || true
      mc ls local/${DATASTORE_BUCKET} >/dev/null
    " >/dev/null 2>&1; then
  info "bucket '${DATASTORE_BUCKET}' present (created if it was absent)"
else
  info "in-cluster bucket setup did not confirm; you can create it after"
  info "starting the port-forward (step below) with the host aws-cli:"
  info "  AWS_ACCESS_KEY_ID=${MINIO_USER} AWS_SECRET_ACCESS_KEY=${MINIO_PASSWORD} \\"
  info "    aws --endpoint-url ${S3_ENDPOINT_URL} s3 mb s3://${DATASTORE_BUCKET}"
fi

# ---------------------------------------------------------------------------
# 7) Write the sourceable env file with the 7 exports (gitignored). Overwriting
#    is intentional and safe: the values are fixed local-dev settings.
# ---------------------------------------------------------------------------
step "Writing the datastore env file: ${ENV_FILE}"
cat > "${ENV_FILE}" <<EOF
# Loom local-dev Metaflow datastore env -- source this before any datastore verb:
#   source .env.metaflow && loom doctor
#
# Generated by scripts/setup_metaflow_minikube.sh. The minio creds are LOCAL-DEV
# ONLY (they live in skills/loom-setup-metaflow/manifests/minio.yaml) -- NOT
# secrets. This file is gitignored. Requires the port-forward below to be live.
export METAFLOW_DEFAULT_DATASTORE=s3
export METAFLOW_DATASTORE_SYSROOT_S3=${DATASTORE_SYSROOT}
export METAFLOW_S3_ENDPOINT_URL=${S3_ENDPOINT_URL}
export AWS_ACCESS_KEY_ID=${MINIO_USER}
export AWS_SECRET_ACCESS_KEY=${MINIO_PASSWORD}
export METAFLOW_DEFAULT_METADATA=local
export METAFLOW_USER=\$(whoami)
EOF
info "wrote 7 exports to ${ENV_FILE}"

# ---------------------------------------------------------------------------
# Done -- print the port-forward command + the verify step.
# ---------------------------------------------------------------------------
step "Setup complete -- next steps"
cat <<EOF
1) Start the minio port-forward in a SEPARATE terminal (keep it running):

     kubectl port-forward -n ${NAMESPACE} svc/minio 9000:9000 9001:9001

   (minio API on :9000, console on :9001 -- ${MINIO_USER} / ${MINIO_PASSWORD})

2) Source the env and verify with the read-only doctor (must end PASS):

     source ${ENV_FILE} && loom doctor

3) Smoke the datastore through the Metaflow Client API:

     loom datasets

To tear the whole local cluster down (reversible): minikube delete
EOF
