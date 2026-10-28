#!/usr/bin/env bash
# Play scripted conversations, each on a fresh home volume, and keep the logs plus his home as it was left
# (without library/) under traces/conversations/<model>/<script>/.
#   scripts/test-conversations.sh [MODEL [SCRIPT...]]
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
image=${MIKE_IMAGE:-mike}
model=${1:-HuggingFaceTB/SmolLM2-360M-Instruct}
shift || true
scripts=("$@")
[ ${#scripts[@]} -gt 0 ] || scripts=("$repo"/tests/conversations/*.txt)
out="$repo/traces/conversations/$(basename "$model")"
mkdir -p "$out"
for script in "${scripts[@]}"; do
  name=$(basename "$script" .txt)
  volume="mike-test-$name"
  docker volume rm -f "$volume" >/dev/null
  # A desk with something on it.
  docker run --rm -v "$volume:/home/mike" "$image" \
    su mike -c 'mkdir -p ~/work/old && echo draft > ~/work/draft.txt && echo old > ~/work/old/x.txt'
  MIKE_MODEL=$model MIKE_HOME_VOLUME=$volume "$repo/scripts/chat.sh" converse --script - \
    < "$script" > "$out/$name.log" 2>&1
  echo "$name: exit $? ($(tail -n 1 "$out/$name.log"))"
  rm -rf "${out:?}/$name" && mkdir -p "$out/$name"
  docker run --rm -v "$volume:/home/mike:ro" "$image" tar -C /home/mike -cf - --exclude=./library . \
    | tar -C "$out/$name" -xf -
done
