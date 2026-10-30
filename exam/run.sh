#!/usr/bin/env bash
# Give one candidate kernel the exam, unscripted, and save the transcript.
#   exam/run.sh NAME HF_REPO:QUANT [off|on]      e.g. exam/run.sh qwen3-1.7b ggml-org/Qwen3-1.7B-GGUF:Q4_K_M off
# The model is served by llama.cpp (llama-server) in its own container; MIK-e talks to it over a private network.
set -euo pipefail
name=${1:?name} hf=${2:?HF_REPO:QUANT} think=${3:-off}
repo=$(cd "$(dirname "$0")/.." && pwd)
net=mike-exam server=mike-llm vol="mike-home-exam-$name-$think"
questions=${MIKE_QUESTIONS:-$repo/exam/questions.txt}  # or a replay, e.g. exam/replays/what-dont-you-know.txt
tag=$(basename "$questions" .txt); [ "$tag" = questions ] && tag= || tag="-$tag"
out="$repo/exam/results/$name-thinking-$think$tag.txt"
mkdir -p "$repo/exam/results"
if [[ $hf == /* ]]; then model_args=(-m "$hf"); else model_args=(-hf "$hf"); fi  # a path inside the mike-gguf volume, or HF_REPO:QUANT
docker network create "$net" >/dev/null 2>&1 || true
docker rm -f "$server" >/dev/null 2>&1 || true
docker run -d --name "$server" --network "$net" -e LLAMA_CACHE=/models -v mike-gguf:/models \
  --cpus "${MIKE_CPUS:-12}" --memory "${MIKE_LLM_MEMORY:-10g}" ghcr.io/ggml-org/llama.cpp:server \
  "${model_args[@]}" --jinja --reasoning-format deepseek -c 16384 -t "${MIKE_CPUS:-12}" --host 0.0.0.0 --port 8080 >/dev/null
echo "waiting for $hf (the first time it downloads) ..."
for i in $(seq 1 180); do
  docker run --rm --network "$net" "${MIKE_IMAGE:-mike}" curl -sf "http://$server:8080/health" >/dev/null 2>&1 && break
  docker inspect -f '{{.State.Running}}' "$server" 2>/dev/null | grep -q true || { docker logs --tail 20 "$server"; exit 1; }
  sleep 10
done
"$repo/exam/setup.sh" "$vol"
echo "exam: $name, thinking $think -> $out"
{ echo "# exam: $name ($hf), thinking $think, $(date -u +%Y-%m-%dT%H:%MZ)"
  MIKE_DEV=1 MIKE_NETWORK="$net" MIKE_SERVER="http://$server:8080" MIKE_SCAFFOLD=none MIKE_THINKING="$think" \
  MIKE_MAX_NEW_TOKENS="${MIKE_MAX_NEW_TOKENS:-2048}" MIKE_HOME_VOLUME="$vol" \
    "$repo/scripts/chat.sh" converse --script - < "$questions" 2>&1 || true
} | tee "$out" | grep -E "^(user|mike)>|^\s+\$ " || true
docker rm -f "$server" >/dev/null
echo "saved $out"
