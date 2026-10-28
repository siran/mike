#!/usr/bin/env bash
# Build MIK-e's computer, the `mike` image. Extra arguments go to `docker build`.
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker build -t "${MIKE_IMAGE:-mike}" "$@" .
