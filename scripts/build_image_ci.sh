#!/bin/sh

# Pipefail doesn't exist in POSIX sh
# shellcheck disable=SC3040
set -e -o pipefail

SCRIPT_DIR="$(dirname "$0")"
. "$SCRIPT_DIR/image_name.sh"

docker build -t "${IMAGE_NAME:?}" .
