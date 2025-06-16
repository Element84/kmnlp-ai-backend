#!/bin/sh

###################################
# Setup creds and login for AWS ECR
###################################

# Pipefail doesn't exist in POSIX sh
# shellcheck disable=SC3040
set -e -o pipefail

SCRIPT_DIR="$(dirname "$0")"
. "$SCRIPT_DIR/image_name.sh"

# Authenticate Docker with ECR
if [ -n "$AWS_PROFILE" ]; then
  echo "Authenticating with ECR using profile $AWS_PROFILE"
  aws ecr get-login-password --profile "$AWS_PROFILE" |
    docker login --username AWS --password-stdin "${ECR_URL:?}"
else
  echo "Authenticating with AWS with no profile. Account id is: ${AWS_ACCOUNT_ID:?}. Access key is: ${AWS_ACCESS_KEY_ID:?}"
  aws ecr get-login-password |
    docker login --username AWS --password-stdin "$ECR_URL"
fi
