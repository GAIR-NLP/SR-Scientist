import argparse
import asyncio
import json
import os
from infer.inference import run_inference_batch
from infer.analysis import analyze_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name", 
        type=str, 
        required=True, 
        help="The model name identifier used by the vLLM server."
    )
    parser.add_argument(
        "--model-url", 
        type=str, 
        required=True, 
        help="URL of the vLLM OpenAI-compatible server."
    )
    parser.add_argument(
        "--sandbox-urls", 
        nargs='+', 
        required=True, 
        help="One or more URLs for the sandbox execution environments."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="The API key."
    )
    parser.add_argument(
        "--parquet-file-path", 
        type=str, 
        required=True, 
        help="Path to the input Parquet data file."
    )
    parser.add_argument(
        "--mape-threshold", 
        type=float, 
        required=True, 
        help="MAPE threshold for evaluation."
    )
    parser.add_argument(
        "--output-json-path", 
        type=str, 
        required=True, 
        help="Path for the output JSON results file."
    )
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=100,
        help="Maximum number of tool-calling turns for the assistant within a single main turn."
    )

    parser.add_argument(
        "--source",
        nargs='+',
        default=None,
        help="The dataset source."
    )

    parser.add_argument(
        "--num-turns",
        type=int,
        required=True,
        help="Number of turns for conversation."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top equations to include in the context for the next turn."
    )

    args = parser.parse_args()

    print("Starting batched inference with the following configuration:")
    print(json.dumps(vars(args), indent=2))
    
    # Run the inference batch process
    asyncio.run(run_inference_batch(
        model_name=args.model_name,
        model_url=args.model_url,
        sandbox_urls=args.sandbox_urls,
        parquet_file_path=args.parquet_file_path,
        mape_threshold=args.mape_threshold,
        num_turns=args.num_turns,
        top_k=args.top_k,
        output_json_path=args.output_json_path,
        max_assistant_turns=args.max_assistant_turns,
        api_key=args.api_key,
        source=args.source
    ))

    # Analyze the results
    output_dir, output_filename = os.path.split(args.output_json_path)
    analyze_results(args.output_json_path)

    


