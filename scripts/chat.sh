#!/usr/bin/env bash
# Start a fresh container from the `mike` image and run `mike ARGS...` in it (default: `mike chat`).
#   scripts/chat.sh
#   scripts/chat.sh converse --script tests/conversations/remember-add.txt
#   scripts/chat.sh converse --script - < my-script.txt
# His home (/home/mike) and the model cache are named volumes and persist; anything else he changes,
# including the kernel in /opt/mike, ends with the container.
# Env: MIKE_IMAGE, MIKE_MODEL, MIKE_HOME_VOLUME, MIKE_HF_VOLUME, MIKE_MEMORY, MIKE_CPUS,
#      MIKE_DEV=1 (mount this repo's src read-only over the image's kernel).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cpus=${MIKE_CPUS:-8}
args=(--rm -i --init --hostname mike
      --memory "${MIKE_MEMORY:-10g}" --cpus "$cpus" --pids-limit 1024
      -e MIKE_THREADS="${cpus%.*}"
      -v "${MIKE_HOME_VOLUME:-mike-home}:/home/mike"
      -v "${MIKE_HF_VOLUME:-mike-hf}:/var/cache/huggingface")
if [ -t 0 ] && [ -t 1 ]; then args+=(-t); fi
if [ -n "${MIKE_MODEL:-}" ]; then args+=(-e MIKE_MODEL="$MIKE_MODEL"); fi
if [ -n "${MIKE_NETWORK:-}" ]; then args+=(--network "$MIKE_NETWORK"); fi
for v in MIKE_SERVER MIKE_SCAFFOLD MIKE_THINKING MIKE_MAX_NEW_TOKENS MIKE_TOOLS; do
  if [ -n "${!v:-}" ]; then args+=(-e "$v=${!v}"); fi
done
if [ "${MIKE_DEV:-}" = 1 ]; then args+=(-v "$repo/src:/opt/mike/src:ro"); fi
if [ $# -eq 0 ]; then set -- chat; fi
exec docker run "${args[@]}" "${MIKE_IMAGE:-mike}" mike "$@"
