# Loom's build of the metaflow-ui SPA — a single-arch simplification of the
# upstream Netflix/metaflow-ui Dockerfile.
#
# Why our own: the upstream Dockerfile selects its base via `FROM ${TARGETARCH}-build`,
# which only resolves under BuildKit/buildx (auto platform ARGs). colima's docker ships
# the LEGACY builder with no buildx plugin, where that indirection fails ("invalid
# reference format"). Plain `FROM node:20-alpine` / `FROM nginx` need none of that: on
# an Apple-Silicon host the legacy builder pulls the arm64 variant from each multi-arch
# manifest, so this builds NATIVELY for arm64 (fast, no emulation).
#
# Build (context = a checkout of Netflix/metaflow-ui):
#   docker build -f metaflow-ui.Dockerfile -t loom/metaflow-ui:<version> /path/to/metaflow-ui
#   minikube image load loom/metaflow-ui:<version>
#
# The /api -> ui_backend proxy is NOT baked here; it is layered at deploy time by the
# ConfigMap in metaflow.yaml (so this stays the stock SPA build).

FROM node:20-alpine AS build
ARG BUILD_RELEASE_VERSION=""
ENV REACT_APP_RELEASE_VERSION=$BUILD_RELEASE_VERSION
WORKDIR /app
COPY package.json yarn.lock ./
# --network-timeout is the documented fix the upstream Dockerfile applies for slow installs.
RUN yarn --frozen-lockfile --network-timeout 1000000
COPY . ./
RUN yarn build

FROM nginx
# Defaults mirror upstream; the deployment overrides METAFLOW_SERVICE etc. The nginx
# template (baked here, overridden by the ConfigMap at runtime) reads these via envsubst.
ENV PORT=3000 \
    METAFLOW_SERVICE=/api \
    METAFLOW_HEAD='' \
    METAFLOW_BODY_BEFORE='' \
    METAFLOW_BODY_AFTER='' \
    MF_DEFAULT_TIME_FILTER_DAYS=''
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 3000
