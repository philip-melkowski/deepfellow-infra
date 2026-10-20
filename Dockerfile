# Which web UI this image serves. `legacy` builds `webui/`, the UI this service has always shipped.
# `dashboard` builds `webui2/`, the DeepFellow Dashboard in single-instance mode. See webui2/README.md.
#
# The default keeps the legacy UI, and its image content is what it always was.
# BuildKit is now REQUIRED, including for the default build: the dashboard stage uses a secret mount, which the
# classic builder rejects, and the classic builder builds every stage. Docker 23 and newer use BuildKit by default.
# BuildKit builds only the stages the target depends on, so the stage this argument does NOT select never runs.
ARG UI_VARIANT=legacy

FROM hub.simplito.com/public/python-docker:3.13.11-docker29.1.2@sha256:11f127bf40f09b49f3b21ada7174242c391c81185811f8d413bd165680fa0d0b AS base

FROM base AS builder
COPY --from=hub.simplito.com/public/uv:0.11.8@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 /uv /uvx /bin/

WORKDIR /app
COPY . .
# Tests are not run here. The `pytest` CI job is the only coverage gate, so the result never depends on which
# Python patch version this image happens to carry.
RUN uv sync --frozen --no-dev
RUN rm -rf .venv/lib/python*/site-packages/*/test
RUN rm -rf .venv/lib/python*/site-packages/*/tests
RUN rm -rf webui2 .ruff_cache

FROM hub.simplito.com/public/node:22.20.0-bookworm AS ui-legacy
WORKDIR /app/webui
COPY webui .
RUN npm install
RUN npm run build
RUN cp -R /app/webui/dist /ui

FROM hub.simplito.com/public/node:22.20.0-bookworm AS ui-dashboard
WORKDIR /app/webui2
COPY webui2 .
# The Dashboard artifact lives in a private package registry, so the install needs a read token.
# The secret is mounted as a FILE and never becomes a layer or a build argument. It is read into the variable
# that webui2/.npmrc expands, inside the same RUN, because a mount sets no environment variable by itself.
RUN --mount=type=secret,id=gitlab_npm_token \
    GITLAB_NPM_TOKEN="$(cat /run/secrets/gitlab_npm_token)" npm ci
RUN npm run build
RUN cp -R /app/webui2/dist /ui

# The stage `UI_VARIANT` names. Nothing else in this file mentions either UI again.
FROM ui-${UI_VARIANT} AS ui

FROM base AS runner

WORKDIR /app
COPY --from=builder /app /app
COPY --from=ui /ui /app/static

ARG DF_BUILD_COMMIT=""
ENV DF_BUILD_COMMIT=$DF_BUILD_COMMIT

CMD ["./.venv/bin/uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8086", "--log-config", "server/utils/logging_config.yaml", "--ws-ping-interval", "5", "--ws-ping-timeout", "10"]
