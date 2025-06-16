#!/bin/sh

# #######################################################
# Push a container to ECR
#
# This script is used in CI to pass images between stages
# #######################################################

# Pipefail doesn't exist in POSIX sh
# shellcheck disable=SC3040
set -e -o pipefail

# Setup vars
SCRIPT_DIR="$(dirname "$0")"
. "$SCRIPT_DIR/image_name.sh"

"$SCRIPT_DIR/setup_ecr.sh"

# Tag and push image
docker tag "${IMAGE_NAME:?}" "${ECR_TAG:?}"
docker push "$ECR_TAG"
