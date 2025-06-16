#!/bin/sh

# #######################################################
# Pull an image frmo ECR
#
# This script is used in CI to pass images between stages
# #######################################################

# Pipefail doesn't exist in POSIX sh
# shellcheck disable=SC3040
set -eo pipefail

# Setup vars
SCRIPT_DIR="$(dirname "$0")"
. "$SCRIPT_DIR/image_name.sh"

"$SCRIPT_DIR/setup_ecr.sh"

# Tag and push image
docker pull "${ECR_TAG:?}"
