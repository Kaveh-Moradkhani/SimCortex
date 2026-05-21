#!/usr/bin/env bash
set -e

umask "${UMASK:-002}"

export HOME="${HOME:-/tmp}"
mkdir -p "$HOME" 2>/dev/null || export HOME=/tmp

exec "$@"
