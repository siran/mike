#!/usr/bin/env bash
# Talk to a candidate kernel yourself, as Mike: the exam's plain prompt, no examples, a home with the exam's files.
#   exam/talk.sh                                   Phi-4-mini
#   exam/talk.sh HF_REPO:QUANT|/models/FILE.gguf [HOME_VOLUME]
# The home volume (default mike-home-talk) is prepared like the exam's the first time and kept after.
set -euo pipefail
hf=${1:-unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M} vol=${2:-mike-home-talk}
repo=$(cd "$(dirname "$0")/.." && pwd)
net=mike-exam server=mike-llm
if [[ $hf == /* ]]; then model_args=(-m "$hf"); else model_args=(-hf "$hf"); fi
docker volume inspect "$vol" >/dev/null 2>&1 || "$repo/exam/setup.sh" "$vol"
docker network create "$net" >/dev/null 2>&1 || true
docker rm -f "$server" >/dev/null 2>&1 || true
docker run -d --name "$server" --network "$net" -e LLAMA_CACHE=/models -v mike-gguf:/models \
  --cpus "${MIKE_CPUS:-12}" --memory "${MIKE_LLM_MEMORY:-4g}" ghcr.io/ggml-org/llama.cpp:server \
  "${model_args[@]}" ${MIKE_CHAT_TEMPLATE:+--chat-template-file "$MIKE_CHAT_TEMPLATE"} --jinja --reasoning-format deepseek -c 8192 --parallel 1 -t "${MIKE_CPUS:-12}" \
  --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f "$server" >/dev/null 2>&1' EXIT
echo "starting $hf ..."
until docker run --rm --network "$net" "${MIKE_IMAGE:-mike}" curl -sf "http://$server:8080/health" >/dev/null 2>&1; do
  docker inspect -f '{{.State.Running}}' "$server" 2>/dev/null | grep -q true || { docker logs --tail 20 "$server"; exit 1; }
  sleep 3
done
MIKE_DEV=1 MIKE_NETWORK="$net" MIKE_SERVER="http://$server:8080" MIKE_SCAFFOLD=none \
  MIKE_THINKING="${MIKE_THINKING:-off}" MIKE_MEMORY=2g MIKE_HOME_VOLUME="$vol" "$repo/scripts/chat.sh" chat
