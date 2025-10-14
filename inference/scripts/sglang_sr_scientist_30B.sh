MODEL_PATH="../models/Qwen3-Coder-30B-A3B-Instruct"
conda activate srscientist
python3 -m sglang.launch_server --model-path $MODEL_PATH \
    --tp 8 \
    --tool-call-parser qwen3_coder \
    --mem-fraction-static 0.85


    