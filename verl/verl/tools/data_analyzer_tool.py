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




class DataAnalyzerTool(BaseTool):
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
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0, 
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]: # <<< 改动: 更新返回类型
        stdin = kwargs.get("stdin", "") 
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("default_language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result, success_state = self.execute_code(instance_id, code, stdin, timeout, language)

        tool_reward = 0.0
        
        metrics = {
                "success": float(success_state),
            }
        
        return ToolResponse(text=result), tool_reward, metrics

    def execute_code(self, instance_id, code, stdin, timeout=30, language="python"):
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