#!/bin/bash
set -x

cd inference/
# --- Initial Variable Definitions ---
PARQUET_FILE_PATH="../data/inference/llmsrbench.parquet"
OUTPUT_JSON_PATH="./output/memory_default/qwen3-coder-480b_tool2_0_001_s40_max_25_top3_N1.json"
MODEL_PATH="../models/Qwen3-Coder-480B-A35B-Instruct-FP8"
TOOL_SCHEMA="tool2"
MAPE_THRESHOLD=0.001
NUM_TURNS=40
INFERENCE_MODE="memory_default"
TOP_K=3
MAX_ASSISTANT_TURNS=25
SAMPLING_MODE="top_k"

# --- MODIFICATION 1: Define LOG_FILE path early so it can be used for all logging ---
TIMESTAMP=$(TZ='UTC-8' date +'%Y%m%d_%H%M%S')
BASENAME=$(basename "$OUTPUT_JSON_PATH" .json)
OUTPUT_DIR=$(dirname "$OUTPUT_JSON_PATH")
LOG_DIR="${OUTPUT_DIR/output/log}"
LOG_FILE="${LOG_DIR}/${BASENAME}_${TIMESTAMP}.log"

# Create the log and output directories if they don't exist to prevent errors
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

cd ../SandboxFusion
conda activate sandbox-runtime
make run-online PORT=9010 &
make run-online PORT=9020 &
make run-online PORT=9030 &
make run-online PORT=9040 &
make run-online PORT=9050 &
make run-online PORT=9060 &
make run-online PORT=9070 &
make run-online PORT=9080 &



SANDBOX_URLS=(
    "http://127.0.0.1:9010/run_code"
    "http://127.0.0.1:9020/run_code"
    "http://127.0.0.1:9030/run_code"
    "http://127.0.0.1:9040/run_code"
    "http://127.0.0.1:9050/run_code"
    "http://127.0.0.1:9060/run_code"
    "http://127.0.0.1:9070/run_code"
    "http://127.0.0.1:9080/run_code"
)
SOURCE=(
    "lsr_synth/bio_pop_growth"
    "lsr_synth/chem_react"
    "lsr_synth/matsci"
    "lsr_synth/phys_osc"
    )
MODEL_URL="http://localhost:30000/v1"


cd ../inference
conda activate srscientist

# Store the full command in a variable for clarity
SGLANG_COMMAND="python3 -m sglang.launch_server --model-path \"$MODEL_PATH\" \
    --tp 4 \
    --dp 2 \
    --tool-call-parser qwen3_coder \
    --mem-fraction-static 0.85 "

# 1. Explicitly write the command text into the main log file.
echo "--- EXECUTING COMMAND ---" >> "$LOG_FILE"
echo "$SGLANG_COMMAND" >> "$LOG_FILE"
echo "------------------------------" >> "$LOG_FILE"

eval $SGLANG_COMMAND &

echo "Waiting for server to be ready at ${MODEL_URL}..."

API_MODEL_NAME="$MODEL_PATH"
CHAT_API_URL="${MODEL_URL}/chat/completions"

echo "Waiting for server to be fully ready by sending test requests to ${CHAT_API_URL}..."
echo "Using model identifier: ${API_MODEL_NAME}"

while true; do
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{
              "model": "'"$API_MODEL_NAME"'",
              "messages": [{"role": "user", "content": "Hello"}],
              "max_tokens": 5,
              "temperature": 0
            }' \
        --connect-timeout 10 --max-time 20 \
        "$CHAT_API_URL")

    if [ "$STATUS_CODE" -eq 200 ]; then
        echo "Server is responsive and ready for inference! API Status: 200."
        break
    else
        echo "Server not fully ready (API Status: $STATUS_CODE). Retrying in 15 seconds..."
        sleep 15
    fi
done


python main.py \
    --model-name "$MODEL_PATH" \
    --model-url "$MODEL_URL" \
    --sandbox-urls "${SANDBOX_URLS[@]}" \
    --parquet-file-path "$PARQUET_FILE_PATH" \
    --mape-threshold $MAPE_THRESHOLD \
    --num-turns $NUM_TURNS \
    --max-assistant-turns $MAX_ASSISTANT_TURNS \
    --top-k $TOP_K \
    --source "${SOURCE[@]}" \
    --output-json-path "$OUTPUT_JSON_PATH" 2>&1 | tee "$LOG_FILE"