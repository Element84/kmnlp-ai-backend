#!/bin/bash

####################################################################################################
# Performs code linting and type checks. Fails if errors are found
####################################################################################################

set -eo pipefail

apt-get update && apt-get install -y shellcheck

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$SCRIPT_DIR/lint.sh"
