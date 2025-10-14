# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures  
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)

# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = ["python", "cpp", "nodejs", "go", "go_test", "java", "php", "csharp", "bash", "typescript", "sql", "rust", "cuda", "lua", "R", "perl", "D_ut", "ruby", "scala", "julia", "pytest", "junit", "kotlin_script", "jest", "verilog", "python_gpu", "lean", "swift", "racket"]


def call_sandbox_api(sandbox_fusion_url: str, code: str, stdin: str, compile_timeout: int, run_timeout: int, memory_limit_mb: int, language: str = "python") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:  # <-- Remove request_id parameter
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
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}")  
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _execute_code(code: str, stdin_data: Any, sandbox_fusion_url: str,  timeout: int, memory_limit_mb: int, language: str, concurrent_semaphore: Optional[threading.Semaphore] = None) -> Tuple[int, Dict[str, Any]]:
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
    stdin_data = json.dumps(stdin_data)
    try:
        if concurrent_semaphore:
            # logger.debug(f"Case {case_index + 1}: Attempting to acquire semaphore.")
            with concurrent_semaphore:
                # logger.debug(f"Case {case_index + 1}: Semaphore acquired. Calling API.")
                api_response, error_msg = call_sandbox_api(sandbox_fusion_url=sandbox_fusion_url, code=current_generation_code, stdin=str(stdin_data), compile_timeout=timeout, run_timeout=timeout, memory_limit_mb=memory_limit_mb, language=language)
            # logger.debug(f"Case {case_index + 1}: Semaphore released.")
        else:
            api_response, error_msg = call_sandbox_api(sandbox_fusion_url=sandbox_fusion_url, code=current_generation_code, stdin=str(stdin_data), compile_timeout=timeout, run_timeout=timeout, memory_limit_mb=memory_limit_mb, language=language)
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        logger.error(f"Case {case_index + 1}: {error_msg}")
        traceback.print_exc()

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
        generation_to_log = code[:200] + "..." if len(code) > 200 else code
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
                is_runtime_error = metadata["run_status"] == "TimeLimitExceeded" or metadata["run_status"] == "Error" or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"  # Runtime timeout
                        result_status = -3
                    else:  # Includes Error and Finished with non-zero return_code
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
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
            logger.warning(f"Unknown API status received: {api_status}")
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1  # Default to -1
    else:  
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
    return result_status, metadata


def check_correctness(code: Optional[dict], stdin_data: str, sandbox_fusion_url: str,  timeout: int = DEFAULT_TIMEOUT, memory_limit_mb: int = 1024, language: str = "python", concurrent_semaphore: Optional[threading.Semaphore] = None) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Checks the correctness of code generation using the remote sandbox API,
    processing test cases concurrently.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        in_outs: Dictionary containing "inputs" and "outputs" lists.
        generation: The generated code string.
        timeout: Timeout for each test case (compile and run share this timeout).
        language: The programming language of the code.

    Returns:
        A tuple (results, metadata_list).
        results: A list containing the test result for each input/output pair
                 (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                 Results are ordered corresponding to the inputs.
        metadata_list: A list containing metadata dictionaries for each test case,
                       ordered corresponding to the inputs.
    """
    logger.info("Starting training and evaluation")

    first_compile_error_index = -1
    try:
        result_status, metadata = _execute_code(
            code=code,
            stdin_data=stdin_data,
            sandbox_fusion_url=sandbox_fusion_url,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            language=language,
            concurrent_semaphore=concurrent_semaphore, 
        )
        if result_status == -4:
            first_compile_error_index = 0
        
    except Exception as exc:
        logger.error(f"Equation generated an exception: {exc}")
        traceback.print_exc()
        result_status = -1 
        metadata = {
            "code": str(code),
            "api_request_error": f"Internal execution error: {exc}",
            "status": "internal_error",
        }

        
 
    logger.info(f"Correctness check finished. Results: {result_status}")
    return result_status, metadata
