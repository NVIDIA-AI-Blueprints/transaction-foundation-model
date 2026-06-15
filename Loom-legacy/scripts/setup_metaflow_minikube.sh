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
METAFLOW_MANIFEST="${REPO_ROOT}/skills/loom-setup-metaflow/manifests/metaflow.yaml"
METAFLOW_MD_IMAGE="netflixoss/metaflow_metadata_service:latest"
ENV_FILE="${REPO_ROOT}/.env.metaflow"

# The visual Metaflow UI (metaflow-ui SPA). It has NO published image, so we build
# it from source — natively for arm64 — with the Loom Dockerfile, then load it. The
# version pins the upstream tag we build + the image tag the manifest references.
METAFLOW_UI_VERSION="1.3.14"
METAFLOW_UI_IMAGE="loom/metaflow-ui:${METAFLOW_UI_VERSION}"
METAFLOW_UI_REPO="https://github.com/Netflix/metaflow-ui.git"
METAFLOW_UI_REF="v${METAFLOW_UI_VERSION}"
METAFLOW_UI_DOCKERFILE="${REPO_ROOT}/skills/loom-setup-metaflow/manifests/metaflow-ui.Dockerfile"

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
# 6b) Deploy the Metaflow metadata SERVICE + its Postgres, so Loom registers runs
#     to a real service (METAFLOW_DEFAULT_METADATA=service) instead of local
#     ~/.metaflow files. The netflixoss image is amd64-only; minikube's containerd
#     won't PULL it on Apple Silicon (no arm64 manifest), but it RUNS once loaded
#     (colima emulates amd64). So: host-pull (emulated) -> minikube image load.
# ---------------------------------------------------------------------------
step "Loading + deploying the Metaflow metadata service + Postgres"
if docker pull --platform linux/amd64 "${METAFLOW_MD_IMAGE}" >/dev/null 2>&1 \
   && minikube image load "${METAFLOW_MD_IMAGE}" >/dev/null 2>&1; then
  info "metadata-service image loaded into the node"
else
  info "could not preload ${METAFLOW_MD_IMAGE}; if the metadata pod ImagePullBackOffs,"
  info "  run: docker pull --platform linux/amd64 ${METAFLOW_MD_IMAGE} && minikube image load ${METAFLOW_MD_IMAGE}"
fi
# Build + load the visual UI (metaflow-ui SPA) from source. No published image
# exists, so we build it natively for arm64 with the Loom Dockerfile. This is the
# only heavy step; it is NON-FATAL — if git/docker are missing or the build fails,
# the metadata service (the load-bearing part) still comes up and the dashboard pod
# simply waits for the image (build it later with the hint below).
UI_READY=0
if minikube image ls 2>/dev/null | grep -q "${METAFLOW_UI_IMAGE}"; then
  info "UI image ${METAFLOW_UI_IMAGE} already loaded"
  UI_READY=1
elif have git && have docker; then
  UI_SRC="$(mktemp -d)"
  info "building the metaflow-ui dashboard from source (${METAFLOW_UI_REF}) -- one-time native build"
  if git clone --depth 1 --branch "${METAFLOW_UI_REF}" "${METAFLOW_UI_REPO}" "${UI_SRC}" >/dev/null 2>&1 \
     && DOCKER_BUILDKIT=0 docker build -f "${METAFLOW_UI_DOCKERFILE}" \
          --build-arg BUILD_RELEASE_VERSION="${METAFLOW_UI_VERSION}" \
          -t "${METAFLOW_UI_IMAGE}" "${UI_SRC}" >/dev/null 2>&1 \
     && minikube image load "${METAFLOW_UI_IMAGE}" >/dev/null 2>&1; then
    info "metaflow-ui built + loaded (${METAFLOW_UI_IMAGE})"
    UI_READY=1
  else
    info "metaflow-ui build/load skipped or failed; the dashboard pod will wait for the image."
    info "  build it later (then re-run this script): git clone --depth 1 --branch ${METAFLOW_UI_REF} \\"
    info "    ${METAFLOW_UI_REPO} /tmp/metaflow-ui && DOCKER_BUILDKIT=0 docker build -f \\"
    info "    ${METAFLOW_UI_DOCKERFILE} -t ${METAFLOW_UI_IMAGE} /tmp/metaflow-ui && \\"
    info "    minikube image load ${METAFLOW_UI_IMAGE}"
  fi
  rm -rf "${UI_SRC}" 2>/dev/null || true
else
  info "git/docker missing; skipping the visual UI build (the metadata service still works)."
fi

kubectl apply -n "${NAMESPACE}" -f "${METAFLOW_MANIFEST}"
# Always wait on Postgres + the metadata service + the UI backend (same already-loaded
# image); only wait on the UI frontend when its image actually loaded.
WAIT_DEPLOYS="deployment/metaflow-db deployment/metaflow-metadata deployment/metaflow-ui-backend"
[[ "${UI_READY}" == "1" ]] && WAIT_DEPLOYS="${WAIT_DEPLOYS} deployment/metaflow-ui"
# shellcheck disable=SC2086 -- WAIT_DEPLOYS is a deliberately word-split list of targets.
if kubectl wait -n "${NAMESPACE}" --for=condition=available --timeout=240s ${WAIT_DEPLOYS}; then
  info "Metaflow stack is available (Postgres + metadata service$([[ "${UI_READY}" == "1" ]] && echo ' + UI'))"
else
  info "some Metaflow component is not ready yet; inspect: kubectl -n ${NAMESPACE} get pods,events"
fi

# ---------------------------------------------------------------------------
# 7) Write the sourceable env file with the exports (gitignored). Overwriting
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
export METAFLOW_DEFAULT_METADATA=service
export METAFLOW_SERVICE_URL=http://localhost:8080/
export METAFLOW_USER=\$(whoami)
EOF
info "wrote the datastore + metadata-service exports to ${ENV_FILE}"

# ---------------------------------------------------------------------------
# Done -- print the port-forward command + the verify step.
# ---------------------------------------------------------------------------
step "Setup complete -- next steps"
cat <<EOF
1) Start the port-forwards in a SEPARATE terminal (keep them running):

     kubectl port-forward -n ${NAMESPACE} svc/minio 9000:9000 9001:9001 &
     kubectl port-forward -n ${NAMESPACE} svc/metaflow-metadata 8080:8080 &
     kubectl port-forward -n ${NAMESPACE} svc/metaflow-ui 3000:3000 &

   Local dashboards & endpoints:
     - Metaflow UI (the visual dashboard: runs,         http://localhost:3000
       DAGs, timelines, artifacts -- the /api proxy
       reaches the UI backend in-cluster, so this
       single port-forward is all the browser needs)
     - minio console (browse the datastore)             http://localhost:9001
         login: ${MINIO_USER} / ${MINIO_PASSWORD}
     - Metaflow metadata service (Loom registers        http://localhost:8080
       runs here; the Client API reads it -- NOT ~/.metaflow files)
     - Loom's own run views (no service needed)          loom report / loom datasets / loom viz

2) Source the env and verify with the read-only doctor (must end PASS):

     source ${ENV_FILE} && loom doctor

3) Smoke the datastore through the Metaflow Client API:

     loom datasets

To tear the whole local cluster down (reversible): minikube delete
EOF
