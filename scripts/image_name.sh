#!/bin/sh

# Pipefail doesn't exist in POSIX sh
# shellcheck disable=SC3040
set -e -o pipefail

export IMAGE_NAME="kmnlp/llm-agent:ci-${CI_COMMIT_SHORT_SHA:?}"
#
# Ensure required AWS details were retrieved
if [ -z "$AWS_REGION" ] || [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Failed to retrieve AWS account ID or region."
    exit 1
fi

# Construct ECR URL and tag
export ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export ECR_TAG="${ECR_URL}/${IMAGE_NAME:?}"
