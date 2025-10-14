MODEL_PATH="../models/gpt-oss-120b"
conda activate srscientist
# For machines without internet access, set TIKTOKEN_RS_CACHE_DIR=CACHE_FILE by following the instructions in this issue: https://huggingface.co/openai/gpt-oss-120b/discussions/39.
python3 -m sglang.launch_server --model-path  $MODEL_PATH \
    --tp 8 \
    --tool-call-parser gpt-oss \
    --reasoning-parser gpt-oss \
    --mem-fraction-static 0.85

    