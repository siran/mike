#!/usr/bin/env bash
# Turn a model folder in the mike-kernels volume into a 4-bit GGUF in mike-gguf, which exam/run.sh can serve.
#   kernel/gguf.sh FOLDER NAME      e.g. kernel/gguf.sh phi-4-mini-instruct phi-4-mini-original
#   then: exam/run.sh NAME /models/mike/NAME-Q4_K_M.gguf off
set -euo pipefail
src=${1:?model folder in mike-kernels} name=${2:?output name}
docker run --rm --memory "${MIKE_TOOL_MEMORY:-4g}" -v mike-kernels:/kernels -v mike-gguf:/models \
  --entrypoint bash ghcr.io/ggml-org/llama.cpp:full -c "
set -e
mkdir -p /models/mike
python3 /app/convert_hf_to_gguf.py /kernels/$src --outtype bf16 --outfile /kernels/$name-bf16.gguf
/app/llama-quantize /kernels/$name-bf16.gguf /models/mike/$name-Q4_K_M.gguf Q4_K_M
rm /kernels/$name-bf16.gguf
ls -la /models/mike/$name-Q4_K_M.gguf"
