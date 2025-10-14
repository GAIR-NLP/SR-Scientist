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

import logging
import os
import re
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar
from uuid import uuid4

import ray
import ray.actor
import ray.util.multiprocessing

from verl.tools.base_tool import BaseTool
from verl.tools.utils.execute_code_utils import _execute_code

from .schemas import OpenAIFunctionToolSchema, ToolResponse 

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")





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




class EquationEvaluatorTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
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
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # TODO: better documentation for the config
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.sandbox_fusion_urls = config.get("sandbox_fusion_urls", [])
        self.next_url_index = 0
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if not self.sandbox_fusion_urls:
            raise ValueError("sandbox_fusion_urls is not set")
        log_msg = f"Init SandboxFusionTool with config: {config}"
        logger.info(log_msg)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        mape_goal = kwargs.get("mape_goal", "") 
        self._instance_dict[instance_id] = {
            "response": "",
            "mape_goal": mape_goal,
            "reward": 0.0, 
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        stdin = kwargs.get("stdin", "") 
        code = parameters.get("equation", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result, success_state = self.execute_code(instance_id, code, stdin, timeout, language)

        tool_reward = 0.0 
        
        metrics = {
                "success": float(success_state),
            }
        
        return ToolResponse(text=result), tool_reward, metrics 

    def execute_code(self, instance_id, code, stdin, timeout=30, language="python"):

      
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
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

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
        # print('using tool Equation Evaluator')
        raw_code = code.strip()
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
        mape_threshold = self._instance_dict[instance_id]['mape_goal']
        mape_threshold_percentage = f"{mape_threshold:.4%}"
        code = code_template.format(equation = equation, mape_threshold = mape_threshold, mape_threshold_percentage=mape_threshold_percentage)
        
        current_sandbox_url = self.sandbox_fusion_urls[self.next_url_index]
        
        # Update the index for the next call
        self.next_url_index = (self.next_url_index + 1) % len(self.sandbox_fusion_urls)
        result_status, metadata = _execute_code(code, stdin, current_sandbox_url, timeout, self.memory_limit_mb, language, None)
        stdout = metadata.get("stdout", "")
        stderr = metadata.get("stderr", "")
        header_offset = 4
        stderr = correct_stderr_line_numbers(stderr, header_offset)
        success_state = False
        if metadata["run_status"] == "Finished":
            if stderr == "":
                actual_output = f"STDOUT:\n{stdout}"
                success_state = True
            else:
                actual_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            logger.debug(f"actual_output from sandbox fusion: {actual_output}")
            return actual_output, success_state
        else:
            failed_output = f"Execution Failed: {metadata.get('run_status', 'Unknown Status')}\n\n---STDOUT---\n{stdout}\n\n---STDERR---\n{stderr}"
            logger.debug(f"Execution not 'Finished'. Returning combined output: {failed_output}")
            return failed_output, success_state

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        reward = self._instance_dict[instance_id].get("reward", 0.0) 
        return float(reward)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
