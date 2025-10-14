MODEL_PATH="../models/GLM-4.5-FP8"
conda activate srscientist
python3 -m sglang.launch_server --model-path $MODEL_PATH \
    --tp 8 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85

    