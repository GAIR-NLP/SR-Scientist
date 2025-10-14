MODEL_PATH="../models/Qwen3-Coder-480B-A35B-Instruct-FP8"
conda activate srscientist
python3 -m sglang.launch_server --model-path $MODEL_PATH \
    --tp 4 \
    --dp 2 \
    --tool-call-parser qwen3_coder \
    --mem-fraction-static 0.85


    