#!/bin/bash

set -e -o pipefail

if [[ -e ".env" ]]; then
  set -a
  source .env
  set +a
fi

EXTRA_ARGS=("${@:1}")

# -n 4 to run with 4 parallel workers
# -vv for verbose output
# -rA to capture stdout output
# --log-cli-level=DEBUG to capture DEBUG level logs
pytest -m integration \
  --ignore tests/experiment \
  -n 4 \
  --log-cli-format="%(asctime)s %(levelname)s %(name)s:%(lineno)s %(message)s" \
  -vv \
  -rA \
  --log-cli-level=DEBUG \
  "${EXTRA_ARGS[@]}"
