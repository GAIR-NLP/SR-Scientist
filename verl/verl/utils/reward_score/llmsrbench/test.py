import concurrent.futures  # <-- Import concurrent.futures
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

SUPPORTED_LANGUAGES = ["python", "cpp", "nodejs", "go", "go_test", "java", "php", "csharp", "bash", "typescript", "sql", "rust", "cuda", "lua", "R", "perl", "D_ut", "ruby", "scala", "julia", "pytest", "junit", "kotlin_script", "jest", "verilog", "python_gpu", "lean", "swift", "racket"]

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)


code_example =  f"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import sys
import json

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS

def equation(m: np.ndarray, h: np.ndarray, n: np.ndarray, epsilon: np.ndarray, params: np.ndarray) -> np.ndarray:
    E_n = params[0] * m + params[1] * h + params[2] * n + params[3] * epsilon
    return E_n

def evaluate(data: list) -> float:

    
    # Load data observations
    outputs = np.array([row[0] for row in data])
    inputs = np.array([row[1:] for row in data])
    X = inputs
    print(X)
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss

if __name__ == '__main__':
    input_data_str = sys.stdin.read()
    data_list = json.loads(input_data_str)
    score = evaluate(data_list)
    
    if score is not None:
        print(score)
"""

import random

def generate_random_float_list(rows, cols, min_val, max_val):
  """
  生成一个指定大小的二维列表，其中包含指定范围内的随机浮-点数。

  参数:
    rows (int): 列表中的行数。
    cols (int): 列表中的列数。
    min_val (float): 随机浮点数的最小值。
    max_val (float): 随机浮点数的最大值。

  返回:
    list: 生成的二维列表。
  """
  # 使用列表推导式高效地生成二维列表
  return [[random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

# --- 生成 80000 * 5 的列表 ---
rows = 80000
cols = 5
min_value = 0.0
max_value = 2.0

# 调用函数生成列表
# 注意：这个列表会很大，可能会占用大量内存
stdin_data = generate_random_float_list(rows, cols, min_value, max_value)

def call_sandbox_api(sandbox_fusion_url: str, code: str, stdin: str, compile_timeout: int = 10, run_timeout: int = 10, memory_limit_mb: int = 1024, language: str = "python") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:  # <-- Remove request_id parameter
    """
    Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        code: The code string to execute.
        stdin: The standard input string.
        compile_timeout: Compile timeout in seconds.
        run_timeout: Run timeout in seconds.
        language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,  # Use the passed language parameter
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None  # Store the last error encountered

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}")  # <-- Use internal log_prefix
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Check for Gateway Timeout (504) specifically for retrying
            if response.status_code == 504:
                last_error = f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt {attempt + 1}/{MAX_RETRIES}"  # <-- Use internal log_prefix
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}")  # <-- Use internal log_prefix
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"

stdin_data = json.dumps(stdin_data)
api_response, error_msg = call_sandbox_api(sandbox_fusion_url='http://0.0.0.0:8080/run_code', code=code_example, stdin=str(stdin_data))


metadata = {
        # "input": str(stdin_data),
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
    }
result_status = -1  # Default error: API request error or unknown sandbox error

if error_msg:
    metadata["status"] = "api_error"
    result_status = -1  # API request itself failed (includes timeout after retries)
    logger.error(f"API error occurred: {error_msg}")
    # Log code and input only on error for brevity
    generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
    logger.error(f"Code: {generation_to_log}")
elif api_response:
    # --- Add debug logging ---
    logger.debug(f"API Response: {api_response}")
    metadata["api_response"] = api_response
    metadata["api_status"] = api_response.get("status")
    compile_result = api_response.get("compile_result")
    run_result = api_response.get("run_result")

    # Extract compile information
    if compile_result:
        metadata["compile_status"] = compile_result.get("status")
        metadata["compile_duration"] = compile_result.get("execution_time")
        metadata["compile_stderr"] = compile_result.get("stderr")

    # Extract run information
    if run_result:
        metadata["run_status"] = run_result.get("status")
        metadata["stdout"] = run_result.get("stdout")
        metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
        metadata["exit_code"] = run_result.get("return_code")
        metadata["duration"] = run_result.get("execution_time")

    # --- Determine status based on API response ---
    api_status = metadata["api_status"]

    if api_status == "SandboxError":
        metadata["status"] = "sandbox_error"
        result_status = -1  # Internal sandbox error
    elif api_status == "Failed":
        # --- Add debug logging ---
        logger.debug(f"API returned Failed status. Response: {api_response}")
        logger.debug(f"Compile Result: {compile_result}")
        logger.debug(f"Run Result: {run_result}")
        # --- Check the logic here ---
        # Compile failed or timed out
        is_compile_error = compile_result and (metadata["compile_status"] in ["Error", "TimeLimitExceeded"] or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0))
        if is_compile_error:
            # Differentiate between compile_error and compile_timeout based on specific status
            if metadata["compile_status"] == "TimeLimitExceeded":
                metadata["status"] = "compile_timeout"
            else:  # Includes Error and Finished but return_code != 0 cases
                metadata["status"] = "compile_error"
            result_status = -4
        # Run failed or timed out
        elif run_result:
            # Modified condition: Check for TimeLimitExceeded OR (Finished with non-zero exit code) OR Error status
            is_runtime_error = metadata["run_status"] == "TimeLimitExceeded" or metadata["run_status"] == "Error" or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
            if is_runtime_error:
                if metadata["run_status"] == "TimeLimitExceeded":
                    metadata["status"] = "timeout"  # Runtime timeout
                    result_status = -3
                else:  # Includes Error and Finished with non-zero return_code
                    metadata["status"] = "runtime_error"
                    result_status = -2
            else:
                # Other Failed status with run_result, classify as unknown failure
                logger.warning(f"Unknown run_status '{metadata['run_status']}' or state within Failed API status.")
                metadata["status"] = "unknown_failure"
                result_status = -1  # Default to -1
        else:
            # Status is Failed but neither a clear compile error nor run_result exists
            logger.warning("API status Failed but cannot determine specific error type (compile/run).")
            metadata["status"] = "unknown_failure_state"
            result_status = -1  # Default to -1
    elif api_status == "Success":
        # Run completed successfully, now check the answer
        if run_result and metadata["run_status"] == "Finished":
            # Note: Output might contain trailing newlines, need normalization
            result_status = True
            metadata["status"] = "success"
        else:
            # Status is Success but run_result status is not Finished, this is unexpected
            metadata["status"] = "unexpected_success_state"
            result_status = -1  # Classify as unknown error
    else:
        # API returned an unknown top-level status
        logger.warning(f"Unknown API status received: {api_status}")
        metadata["status"] = f"unknown_api_status_{api_status}"
        result_status = -1  # Default to -1
else:  # api_response is None and no error_msg (Should not happen with current call_sandbox_api logic)
    metadata["status"] = "unknown_api_state"
    result_status = -1
    logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
    
# print(result_status)
print(metadata['stdout'])
print(metadata['stderr'])



