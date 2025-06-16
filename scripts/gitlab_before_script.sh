#!/bin/sh

####################################################################################################
# Sets up the gitlab runner docker image to install additional things needed for build steps.
####################################################################################################

# Pipefail doesn't exist in POSIX sh
# shellcheck disable=SC3040
set -e -o pipefail

# Install necessary dependencies, incl. aws-cli.
apk update && \
  apk add \
  shellcheck \
  jq \
  aws-cli
