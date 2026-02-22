#!/bin/bash

set -e -o pipefail

# Setting environment variables that are needed for testing. These won't be used but must be set.
export GEOCODE_INDEX_HOST=localhost
export GEOCODE_INDEX_PORT=9200
export GEOCODE_INDEX_REGION=us-east-1

WATCH_MODE=false

EXTRA_ARGS=()

for arg in "$@"; do
  case $arg in
    --watch)
      WATCH_MODE=true
      shift
      ;;
    *)
      EXTRA_ARGS+=("$arg")
  esac
done


if [[ "$WATCH_MODE" = true ]]; then
  pytest-watch -- -vv -rA --log-cli-level=INFO "${EXTRA_ARGS[@]}"
else
  # -vv for verbose output
  # -rA to capture stdout output
  # --log-cli-level=INFO to capture INFO level logs
  pytest -vv -rA --log-cli-level=INFO "${EXTRA_ARGS[@]}"
fi
