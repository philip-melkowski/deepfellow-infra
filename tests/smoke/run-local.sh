#!/usr/bin/env bash
#
# Run the Bruno smoke tests against a local DeepFellow Infra instance.
#
# Starts the dev server the same way the README does (`just dev`, uvicorn on
# localhost:8086), waits until it is healthy, runs the Bruno collection against
# the `local` environment, then shuts the server down again.
#
# If a server is already listening on :8086 it is reused and left running.
#
# Usage:
#   tests/smoke/run-local.sh                 # run the whole collection
#   tests/smoke/run-local.sh smoke           # run only the smoke/ folder
#   tests/smoke/run-local.sh --reporter-junit results.xml
#
# Any extra arguments are forwarded to `bru run`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOST="http://localhost:8086"
HEALTH_URL="$HOST/health"
LOG_FILE="$(mktemp -t df-infra-bruno.XXXXXX.log)"

# --- preconditions -----------------------------------------------------------

if ! command -v bru >/dev/null 2>&1; then
  echo "error: Bruno CLI not found. Install it with: npm install -g @usebruno/cli@4.0.0" >&2
  exit 1
fi

# --- cleanup ------------------------------------------------------------------
#
# Registered up front so a failure anywhere below still removes the generated
# token file and stops the server.

SERVER_PID=""
GENERATED_ENV=""

cleanup() {
  if [[ -n "$GENERATED_ENV" ]]; then
    rm -f "$GENERATED_ENV"
  fi
  if [[ -n "$SERVER_PID" ]]; then
    echo "Stopping dev server (pgid $SERVER_PID)..."
    # `just dev` runs uvicorn --reload, which spawns child processes; setsid put
    # them all in one process group so we can take the whole group down.
    kill -- "-$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# --- auth tokens: match whatever the server actually boots with ---------------
#
# The two keys come from different places, and getting this wrong just yields
# confusing 401s:
#
#   * DF_INFRA_ADMIN_API_KEY is a bootstrap setting — always read from .env.
#   * DF_INFRA_API_KEY is dynamic config. Once storage/config.json exists it is
#     the authority and the .env value is no longer read (see
#     server/dynamic_config.py), so prefer it and fall back to .env.

env_value() { # env_value KEY FILE
  grep -E "^$1=" "$2" 2>/dev/null | tail -n1 | cut -d= -f2- || true
}

API_TOKEN=""
ADMIN_TOKEN=""

if [[ -f "$ROOT/.env" ]]; then
  API_TOKEN="$(env_value DF_INFRA_API_KEY "$ROOT/.env")"
  ADMIN_TOKEN="$(env_value DF_INFRA_ADMIN_API_KEY "$ROOT/.env")"
else
  echo "warning: $ROOT/.env not found — relying on API_TOKEN/ADMIN_TOKEN from the environment" >&2
fi

CONFIG_JSON="${DF_STORAGE_DIR:-$ROOT/storage}/config.json"
if [[ -f "$CONFIG_JSON" ]]; then
  FROM_CONFIG="$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1]))['settings']['infra_api_key'])
except Exception:
    pass
" "$CONFIG_JSON")"
  if [[ -n "$FROM_CONFIG" ]]; then
    API_TOKEN="$FROM_CONFIG"
    echo "Using server key from $CONFIG_JSON (it overrides .env once it exists)"
  fi
fi

# The version /info must report — the smoke test skips that assertion if unset.
EXPECTED_VERSION="$(cd "$ROOT" && python3 -c "
import tomllib
print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])
")"

# Bruno resolves {{process.env.X}} from the collection's own .env file only — it
# does NOT inherit exported shell variables. Without this file the requests go
# out with an empty bearer token, which looks like a plain 401 failure (and, for
# the auth negatives, would pass for the wrong reason).
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  echo "Using existing $SCRIPT_DIR/.env (delete it to let this script regenerate one)"
else
  umask 077
  cat >"$SCRIPT_DIR/.env" <<EOF
API_TOKEN=$API_TOKEN
ADMIN_TOKEN=$ADMIN_TOKEN
EXPECTED_VERSION=$EXPECTED_VERSION
EOF
  GENERATED_ENV="$SCRIPT_DIR/.env"
fi

# --- start the server (unless one is already up) -----------------------------

if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
  echo "Server already healthy at $HOST — reusing it."
else
  echo "Starting dev server (just dev), logging to $LOG_FILE ..."
  ( cd "$ROOT" && exec setsid just dev ) >"$LOG_FILE" 2>&1 &
  SERVER_PID=$!

  echo -n "Waiting for $HEALTH_URL "
  for _ in $(seq 1 60); do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      echo " up."
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo
      echo "error: dev server exited before becoming healthy. Last log lines:" >&2
      tail -n 30 "$LOG_FILE" >&2
      exit 1
    fi
    echo -n "."
    sleep 1
  done

  if ! curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    echo
    echo "error: server did not become healthy within 60s. Last log lines:" >&2
    tail -n 30 "$LOG_FILE" >&2
    exit 1
  fi
fi

# --- run the tests -----------------------------------------------------------

# `bru run` must be invoked from the collection root ("You can run only at the
# root of a collection"), and needs -r to descend into smoke/ and auth/.
TARGET="."
# A bare first argument naming a subfolder runs just that folder.
if [[ $# -gt 0 && -d "$SCRIPT_DIR/$1" ]]; then
  TARGET="$1"
  shift
fi

echo "Running: bru run $TARGET -r --env local $*"
cd "$SCRIPT_DIR"
bru run "$TARGET" -r --env local "$@"
