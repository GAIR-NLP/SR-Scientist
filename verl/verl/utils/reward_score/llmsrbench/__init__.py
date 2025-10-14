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
import json
import logging
import traceback
import math
import re
import ast
from .utils import check_correctness

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)


def compute_score(code, stdin_data, sandbox_fusion_url, ):
    """
    Computes the code score using the remote sandbox API.

    Args:
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"
        code: The code string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        continuous: Whether to compute a continuous score (based on the first N test cases).
        timeout: Timeout for each test case.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata: Execution metadata.
    """
    solution = code
    if "```python" in code:
        solution = code.split("```python")[-1].split("```")[0]
    elif "```" in code:
        # Handle cases like ```\ncode\n```
        parts = code.split("```")
        if len(parts) >= 3:
            solution = parts[-2]
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():
                    solution = rest
    else:
        return 0.0, 'Code block prases errors'


    try:
        
        # Check all test cases
        # Note: The return value of check_correctness might need adaptation here
        # Assume check_correctness returns (results_list, metadata_list)
        # results_list contains True, False, or error codes (-1, -2, -3, etc.)
        result_status, metadata = check_correctness(code=solution, stdin_data=stdin_data, sandbox_fusion_url=sandbox_fusion_url)

        # Calculate score
        if result_status != True or metadata['stdout'] == '':  # If there are no results (e.g., invalid input)
            return 0.0, metadata
        # import pprint
        # pprint.pprint(f"stdout: {metadata['stdout']}")
        score = math.exp(-float(metadata['stdout']))
        # pprint.pprint(f"score: {score}")
        

    except Exception as e:
        logger.error(f"Error during compute_score: {e}")
        # traceback.print_exc()
        score = 0.0
        # Try to return partial metadata if available, otherwise return error info
        metadata = metadata if "metadata" in locals() else "Unhandled exception: {e}"

    # Ensure float and list are returned
    return float(score), metadata
