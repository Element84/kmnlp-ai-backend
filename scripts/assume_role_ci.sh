#!/bin/sh

##################################################
# Assume AWS role and export related env variables
##################################################

# Pipefail doesn't exist in POSIX sh
# shellcheck disable=SC3040
set -e -o pipefail

CREDS=$(aws sts assume-role --role-arn "${CI_INTEGRATION_TEST_ROLE_ARN:?}" --role-session-name "git-lab-integration-tests")
AWS_ACCESS_KEY_ID=$(echo "$CREDS" | jq -r '.Credentials.AccessKeyId')
AWS_SECRET_ACCESS_KEY=$(echo "$CREDS" | jq -r '.Credentials.SecretAccessKey')
AWS_SESSION_TOKEN=$(echo "$CREDS" | jq -r '.Credentials.SessionToken')
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN

echo "Access key id: $AWS_ACCESS_KEY_ID"
