#!/bin/bash

set -e -o pipefail

EXTRA_ARGS=("${@:1}")

# -vv for verbose output
# -rA to capture stdout output
# --log-cli-level=INFO to capture INFO level logs
pytest -vv -rA --log-cli-level=INFO "${EXTRA_ARGS[@]}"
