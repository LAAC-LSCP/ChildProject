#!/bin/sh

set -eu

# Detect the shell from which the script was called
parent=$(ps -o comm $PPID |tail -1)
parent=${parent#-}  # remove the leading dash that login shells have

# Computing artifact location
case "$(uname)" in
  Linux)
    PLATFORM="linux" ;;
  Darwin)
    PLATFORM="macos" ;;
  *NT*)
    PLATFORM="windows" ;;
esac

RELEASE_URL="https://raw.githubusercontent.com/LAAC-LSCP/ChildProject/master/env_${PLATFORM}.yml"

# find environment manager
if hash micromamba >/dev/null 2>&1; then
  ENV_MANAGER="micromamba"
elif hash conda >/dev/null 2>&1; then
  ENV_MANAGER="conda"
else
  echo "Neither micromamba nor conda was found, get micromamba at https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html" >&2
  exit 1
fi

# Downloading artifact
mkdir -p "${BIN_FOLDER}"
if hash curl >/dev/null 2>&1; then
  HTTP_CMD="curl -o"
elif hash wget >/dev/null 2>&1; then
  HTTP_CMD="wget -O"
else
  echo "Neither curl nor wget was found" >&2
  exit 1
fi

eval "${HTTP_CMD} /tmp/env.yml ${RELEASE_URL} && ${ENV_MANAGER} env create -f /tmp/env.yml"

if [ $? -ne 0 ]; then
  echo "Installation of environment failed with code $?"
  exit $?

case "$PLATFORM" in
  macos)
    brew install git-annex ;;
  windows)
    eval "${HTTP_CMD} /tmp/git-annex-installer.exe https://downloads.kitenet.net/git-annex/windows/current/git-annex-installer.exe && ./tmp/git-annex-installer.exe" ;;
esac

echo "Installation Completed for environment childproject, activate it with: \n${ENV_MANAGER} activate childproject"
