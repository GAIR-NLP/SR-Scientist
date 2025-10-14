import asyncio  
import httpx    
import json
import logging
import os
import threading
import time
import traceback
import uuid
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 2700


def convert_numpy_types(obj):
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


tools =  [
    {
      "type": "function",
      "function": {
        "name": "equation_evaluator",
        "description": "Accepts a mathematical equation as a Python function string, optimizes its parameters to fit a dataset using the BFGS method, and returns performance metrics (MSE, NMSE, MAPE) to evaluate its goodness of fit.",
        "parameters": {
          "type": "object",
          "properties": {
            "equation": {
              "type": "string",
              "description": "The equation to evaluate, provided as a complete Python function string."
            }
          },
          "required": [
            "equation"
          ]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "data_analyzer",
        "description": "Executes Python code for data analysis and exploration on a given dataset to inspect for relationships or anomalies. This tool does not support data visualization or plotting libraries like Matplotlib.",
        "parameters": {
          "type": "object",
          "properties": {
            "code": {
              "type": "string",
              "description": "The Python code snippet for data analysis to execute."
            }
          },
          "required": [
            "code"
          ]
        }
      }
    }
  ]







def correct_stderr_line_numbers(stderr_text: str, offset: int) -> str:
    if not stderr_text:
        return ""

    traceback_pattern = re.compile(r'(File ".*?", line )(\d+)')
    warning_pattern = re.compile(r'([^"\s]+?\.py:)(\d+)')

    corrected_lines = []
    for line in stderr_text.splitlines():
        if traceback_pattern.search(line):
            corrected_line = re.sub(traceback_pattern, lambda m: f"{m.group(1)}{int(m.group(2)) - offset}", line)
        elif warning_pattern.search(line):
            corrected_line = re.sub(warning_pattern, lambda m: f"{m.group(1)}{int(m.group(2)) - offset}", line)
        else:
            corrected_line = line
        corrected_lines.append(corrected_line)
        
    return "\n".join(corrected_lines)



async def call_sandbox_api(sandbox_fusion_url: str, code: str, stdin: str, compile_timeout: int, run_timeout: int, memory_limit_mb: int, language: str = "python") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Asynchronously calls the remote sandbox API to execute code with retry logic.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Request ID: {request_id}] "

    payload = json.dumps({
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "code": code,
        "stdin": stdin,
        "memory_limit_MB": memory_limit_mb,
        "language": language,
        "files": {},
        "fetch_files": [],
    })
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None

    async with httpx.AsyncClient() as client:
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}")
                
                response = await client.post(
                    sandbox_fusion_url,
                    headers=headers,
                    data=payload,
                    timeout=request_timeout,
                )

                if response.status_code == 504:
                    last_error = f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt {attempt + 1}/{MAX_RETRIES}"
                    logger.warning(last_error)
                    if attempt < MAX_RETRIES - 1:
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                        logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                        await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                logger.info(f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}")
                return response.json(), None

            except httpx.RequestError as e:
                last_error = f"{log_prefix}API Request Error: {e}"
                break
            except json.JSONDecodeError as e:
                raw_response_text = response.text if "response" in locals() else "N/A"
                last_error = f"{log_prefix}API Response JSON Decode Error: {e}. Response text: {raw_response_text}"
                break
            except Exception as e:
                last_error = f"{log_prefix}Unexpected Error: {e}"
                break

    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


async def _execute_code(code: str, stdin_data: Any, sandbox_fusion_url: str,  timeout: int, memory_limit_mb: int, language: str, concurrent_semaphore: Optional[threading.Semaphore] = None, tool_name = 'data_analyzer') -> Tuple[int, Dict[str, Any]]:
    """Helper function to process a single test case."""
    api_response = None
    error_msg = None
    raw_code = code.strip()
    processed_code = raw_code

    if "```python" in raw_code:
        processed_code = raw_code.split("```python")[-1].split("```")[0].strip()
    elif "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            potential_code = parts[1]
            if "\n" in potential_code:
                first_line, rest_of_code = potential_code.split("\n", 1)
                if first_line.strip().isalpha():
                    processed_code = rest_of_code.strip()
                else:
                    processed_code = potential_code.strip()
            else:
                processed_code = potential_code.strip()
    
    header = """import os\nos.environ['OPENBLAS_NUM_THREADS'] = '1'\nos.environ['OMP_NUM_THREADS'] = '1'\nos.environ['MKL_NUM_THREADS'] = '1'\n"""
    
    code = (header + processed_code)
    current_generation_code = code
    
    stdin_data = json.dumps(convert_numpy_types(stdin_data))
    try:
        api_response, error_msg = await call_sandbox_api(
            sandbox_fusion_url=sandbox_fusion_url, 
            code=current_generation_code, 
            stdin=str(stdin_data), 
            compile_timeout=timeout, 
            run_timeout=timeout, 
            memory_limit_mb=memory_limit_mb, 
            language=language
        )
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness: {e}"
        traceback.print_exc()

    metadata = {
        "code": current_generation_code,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
        'tool_name': tool_name
    }
    result_status = -1

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1
        logger.error(f"API error occurred: {error_msg}")
        generation_to_log = code[:200] + "..." if len(code) > 200 else code
        logger.error(f"Code: {generation_to_log}")
    elif api_response:
        logger.debug(f"API Response: {api_response}")
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")

        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        api_status = metadata["api_status"]
        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1
        elif api_status == "Failed":
            is_compile_error = compile_result and (metadata["compile_status"] in ["Error", "TimeLimitExceeded"] or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0))
            if is_compile_error:
                if metadata["compile_status"] == "TimeLimitExceeded":
                    metadata["status"] = "compile_timeout"
                else:
                    metadata["status"] = "compile_error"
                result_status = -4
            elif run_result:
                is_runtime_error = metadata["run_status"] == "TimeLimitExceeded" or metadata["run_status"] == "Error" or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"
                        result_status = -3
                    else:
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    logger.warning(f"Unknown run_status '{metadata['run_status']}' or state within Failed API status.")
                    metadata["status"] = "unknown_failure"
                    result_status = -1
            else:
                logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                metadata["status"] = "unknown_failure_state"
                result_status = -1
        elif api_status == "Success":
            if run_result and metadata["run_status"] == "Finished":
                result_status = True
                metadata["status"] = "success"
            else:
                metadata["status"] = "unexpected_success_state"
                result_status = -1
        else:
            logger.warning(f"Unknown API status received: {api_status}")
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1
    else:
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(f"Unknown API state (no response and no error message).")
    return result_status, metadata


async def execute_code(code, stdin_data, sandbox_fusion_url='http://0.0.0.0:8080/run_code', memory_limit_mb = 1024, timeout=30, language="python"):
    result_status, metadata = await _execute_code(code, stdin_data, sandbox_fusion_url = sandbox_fusion_url, timeout = timeout, memory_limit_mb = memory_limit_mb, language = language, concurrent_semaphore = None)
    stdout = metadata.get("stdout", "")
    stderr = metadata.get("stderr", "")
    header_offset = 4
    stderr = correct_stderr_line_numbers(stderr, header_offset)
    
    if metadata["run_status"] == "Finished":
        if stderr == "":
            actual_output = f"STDOUT:\n{stdout}"
        else:
            actual_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        logger.debug(f"actual_output from sandbox fusion: {actual_output}")
        return actual_output, metadata
    else:
        failed_output = f"Execution Failed: {metadata.get('run_status', 'Unknown Status')}\n\n---STDOUT---\n{stdout}\n\n---STDERR---\n{stderr}"
        logger.debug(f"Execution not 'Finished'. Returning combined output: {failed_output}")
        return failed_output, metadata



async def equation_evaluator(equation, stdin_data, sandbox_fusion_url='http://0.0.0.0:8080/run_code', memory_limit_mb = 1024, timeout=30, language="python", mape_threshold=0.001):
    mape_threshold_percentage = f"{mape_threshold:.4%}"
    code_template = """
import numpy as np
import sys
import json

# Initialize parameters
MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS

# You only need to modify the equation here
{equation}

def evaluate(data: list) -> float:
    
    # Load data observations
    outputs = np.array([row[0] for row in data])
    inputs = np.array([row[1:] for row in data])
    X = inputs
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        try:
            y_pred = equation(*X.T, params)
            return np.mean((y_pred - outputs) ** 2)
        except (FloatingPointError, OverflowError):
            return np.inf

    loss_partial = lambda p: loss(p)
    result = minimize(loss_partial, [1.0] * MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    mse = result.fun

    if np.isnan(mse) or np.isinf(mse):
        return None, None, None

    var_outputs = np.var(outputs)
    if np.isclose(var_outputs, 0):
        nmse = 0.0 if np.isclose(mse, 0) else np.inf
    else:
        nmse = mse / var_outputs

    y_pred = equation(*X.T, optimized_params)
    
    zero_mask = np.isclose(outputs, 0)
    non_zero_mask = ~zero_mask
    
    mape = 0.0
    if np.any(non_zero_mask):
        relative_errors = np.abs((y_pred[non_zero_mask] - outputs[non_zero_mask]) / outputs[non_zero_mask])
        mape = np.mean(relative_errors)

    return float(mse), float(nmse), float(mape)


if __name__ == '__main__':
    input_data_str = sys.stdin.read()
    data_list = json.loads(input_data_str)
    mse, nmse, mape = evaluate(data_list)
    
    if mse is not None:
        print(f"MSE loss: {{mse:.6e}}; NMSE loss: {{nmse:.6e}}; Mean absolute percentage error: {{mape:.4%}}")
        if mape < {mape_threshold}:
          print("Success: The mean absolute percentage error is smaller than {mape_threshold_percentage}.")
        else:
          print("Failure: The mean absolute percentage error is larger than {mape_threshold_percentage}.")
"""
    raw_code = equation.strip()
    equation = raw_code

    if "```python" in raw_code:
        equation = raw_code.split("```python")[-1].split("```")[0].strip()
    elif "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            potential_code = parts[1]
            if "\n" in potential_code:
                first_line, rest_of_code = potential_code.split("\n", 1)
                if first_line.strip().isalpha():
                    equation = rest_of_code.strip()
                else:
                    equation = potential_code.strip()
            else:
                equation = potential_code.strip()
    
    code = code_template.format(equation = equation, mape_threshold = mape_threshold, mape_threshold_percentage = mape_threshold_percentage)
    result_status, metadata = await _execute_code(code, stdin_data, sandbox_fusion_url = sandbox_fusion_url, timeout = timeout, memory_limit_mb = memory_limit_mb, language = language, concurrent_semaphore = None, tool_name='equation_evaluator')
    stdout = metadata.get("stdout", "")
    stderr = metadata.get("stderr", "")
    header_offset = 4
    stderr = correct_stderr_line_numbers(stderr, header_offset)
    
    if metadata["run_status"] == "Finished":
        if stderr == "":
            actual_output = f"STDOUT:\n{stdout}"
        else:
            actual_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        logger.debug(f"actual_output from sandbox fusion: {actual_output}")
        return actual_output, metadata
    else:
        failed_output = f"Execution Failed: {metadata.get('run_status', 'Unknown Status')}\n\n---STDOUT---\n{stdout}\n\n---STDERR---\n{stderr}"
        logger.debug(f"Execution not 'Finished'. Returning combined output: {failed_output}")
        return failed_output, metadata

def get_function_by_name(name):
    if name == "data_analyzer":
        return execute_code
    elif name == "equation_evaluator":
        return equation_evaluator


