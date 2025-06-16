#!/bin/bash

####################################################################################################
# Performs code linting, type checks, and formatting checks. Fails if errors are found.
####################################################################################################

set -e -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Running shellcheck"
"$SCRIPT_DIR/shellcheck.sh"

echo "Running Ruff"
ruff check src/ tests/ evals/
echo "---"

echo "Running pyright"
pyright src/ tests/ evals/
echo "---"

echo "Checking if code has been formatted with Ruff"
if ! ruff format --check; then
    echo "ERROR: Code has not been formatted with Ruff. Please run: ruff format."
    exit 1
fi
echo "---"

echo "All linting checks passed!"
