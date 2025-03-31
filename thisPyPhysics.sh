#!/bin/sh

# Determine the shell and set SCRIPT_DIR accordingly
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%N}" )" &> /dev/null && pwd )
else
    echo "Unsupported shell. Use Bash or Zsh."
    return 1
fi

# Export PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:"${SCRIPT_DIR}/src/"
