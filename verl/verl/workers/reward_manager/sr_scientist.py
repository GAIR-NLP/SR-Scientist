
import datetime 
from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register


import re
import json
import ast
from typing import List, Dict
from scipy.optimize import minimize
import numpy as np

# Global scope for the exec function
exec_globals = {
    'np': np,
    'minimize': minimize
}

# Code snippet for evaluating on a single dataset (e.g., training set)
code_snippet_single_dataset = """

{equation}

def evaluate(equation, data, thresholds):
    MAX_NPARAMS = 10
    outputs = np.array([row[0] for row in data])
    inputs = np.array([row[1:] for row in data])
    
    def loss(params):
        try:
            y_pred = equation(*inputs.T, params)
            return np.mean((y_pred - outputs) ** 2)
        except (FloatingPointError, OverflowError):
            return np.inf

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')
    optimized_params = result.x
    mse = result.fun

    if np.isnan(mse) or np.isinf(mse):
        return None, None, None, [None] * len(thresholds), optimized_params

    var_outputs = np.var(outputs)
    nmse = mse / var_outputs if not np.isclose(var_outputs, 0) else (0.0 if np.isclose(mse, 0) else np.inf)

    y_pred = equation(*inputs.T, optimized_params)
    
    zero_mask = np.isclose(outputs, 0)
    non_zero_mask = ~zero_mask
    
    mape = np.nan
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((outputs[non_zero_mask] - y_pred[non_zero_mask]) / outputs[non_zero_mask]))
    elif np.allclose(y_pred, 0):
        mape = 0.0
    else:
        mape = np.inf

    accuracies = []
    if len(outputs) == 0:
        accuracies = [0.0] * len(thresholds)
    else:
        for threshold in thresholds:
            correct_predictions = np.zeros_like(outputs, dtype=bool)
            if np.any(non_zero_mask):
                relative_error = np.abs((y_pred[non_zero_mask] - outputs[non_zero_mask]) / outputs[non_zero_mask])
                correct_predictions[non_zero_mask] = relative_error < threshold
            correct_predictions[zero_mask] = np.isclose(y_pred[zero_mask], 0, atol=1e-9)
            accuracies.append(1.0 if np.mean(correct_predictions) >= 0.95 else 0.0)
    
    return float(mse), float(nmse), float(mape), accuracies, optimized_params
"""

# Code snippet for fitting on train set and evaluating on test set (This is still needed)
code_snippet_train_test = """

{equation}

def evaluate(equation, train_data, test_data, thresholds):
    MAX_NPARAMS = 10
    train_outputs = np.array([row[0] for row in train_data])
    train_inputs = np.array([row[1:] for row in train_data])
    
    def loss(params):
        try:
            y_pred_train = equation(*train_inputs.T, params)
            return np.mean((y_pred_train - train_outputs) ** 2)
        except (FloatingPointError, OverflowError):
            return np.inf

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')
    optimized_params = result.x

    test_outputs = np.array([row[0] for row in test_data])
    test_inputs = np.array([row[1:] for row in test_data])
    
    try:
        y_pred_test = equation(*test_inputs.T, optimized_params)
    except (FloatingPointError, OverflowError):
        return None, None, None, [None] * len(thresholds), optimized_params

    mse = np.mean((y_pred_test - test_outputs) ** 2)

    if np.isnan(mse) or np.isinf(mse):
        return None, None, None, [None] * len(thresholds), optimized_params

    var_test_outputs = np.var(test_outputs)
    nmse = mse / var_test_outputs if not np.isclose(var_test_outputs, 0) else (0.0 if np.isclose(mse, 0) else np.inf)

    zero_mask = np.isclose(test_outputs, 0)
    non_zero_mask = ~zero_mask
    
    mape = np.nan
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((test_outputs[non_zero_mask] - y_pred_test[non_zero_mask]) / test_outputs[non_zero_mask]))
    elif np.allclose(y_pred_test, 0):
        mape = 0.0
    else:
        mape = np.inf

    accuracies = []
    if len(test_outputs) == 0:
        accuracies = [0.0] * len(thresholds)
    else:
        for threshold in thresholds:
            correct_predictions = np.zeros_like(test_outputs, dtype=bool)
            if np.any(non_zero_mask):
                relative_error = np.abs((y_pred_test[non_zero_mask] - test_outputs[non_zero_mask]) / test_outputs[non_zero_mask])
                correct_predictions[non_zero_mask] = relative_error < threshold
            correct_predictions[zero_mask] = np.isclose(y_pred_test[zero_mask], 0, atol=1e-9)
            accuracies.append(1.0 if np.mean(correct_predictions) > 0.95 else 0.0)
    
    return float(mse), float(nmse), float(mape), accuracies, optimized_params
"""


def _clean_code_snippet(raw_code: str) -> str:
    """
    Cleans a string of code by removing markdown fences like ```python.
    """
    processed_code = raw_code.strip()
    if "```python" in raw_code:
        processed_code = raw_code.split("```python")[-1].split("```")[0].strip()
    elif "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            potential_code = parts[1]
            if "\n" in potential_code:
                first_line, rest_of_code = potential_code.split("\n", 1)
                processed_code = rest_of_code.strip() if first_line.strip().isalpha() else potential_code.strip()
            else:
                processed_code = potential_code.strip()
    return processed_code

def parse_equations_and_mses(text: str) -> List[Dict]:
    """
    Parses the model response to find pairs of tool calls (equations)
    and their subsequent tool responses (MSE values).
    """
    parsed_results = []
    
    tool_calls = list(re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL))
    tool_responses = list(re.finditer(r'<tool_response>(.*?)</tool_response>', text, re.DOTALL))
    response_idx = 0
    for call_match in tool_calls:
        equation_match = re.search(r'<parameter=equation>(.*?)</parameter>', call_match.group(1), re.DOTALL)
        if not equation_match:
            continue
        

        raw_equation_code = equation_match.group(1)
        cleaned_equation_code = _clean_code_snippet(raw_equation_code)
        
        found_response = None
        for i in range(response_idx, len(tool_responses)):
            if tool_responses[i].start() > call_match.end():
                found_response = tool_responses[i]
                response_idx = i + 1
                break
        
        if found_response:
            response_text = found_response.group(1)
            mse_match = re.search(r"MSE loss:\s*([\d.e+-]+);\s*NMSE loss:.*?; Mean absolute percentage error:.*%", response_text)
            if mse_match:
                try:
                    mse_value = float(mse_match.group(1))
                    parsed_results.append({'equation': cleaned_equation_code, 'mse': mse_value})
                except (ValueError, TypeError):
                    continue
                    
    return parsed_results

def _execute_performance_evaluation(equations: List[str], code_snippet: str, eval_args: tuple, mape_threshold: float) -> dict:
    MSE, NMSE, MAPE, ACC_0_1, ACC_0_01, ACC_0_001, Optimized_Params, SUCCESS = [], [], [], [], [], [], [], []
    thresholds = [0.1, 0.01, 0.001]
    
    for equation_string in equations:
        success = 0
        try:
            full_code_to_exec = code_snippet.format(equation=equation_string)
            local_scope = {}
            exec(full_code_to_exec, exec_globals, local_scope)
            
            evaluate_func = local_scope['evaluate']
            equation_func = local_scope['equation']
            
            mse, nmse, mape, accuracies, optimized_params = evaluate_func(equation_func, *eval_args, thresholds)
            
            acc_0_1, acc_0_01, acc_0_001 = accuracies
            if mape is not None and mape < mape_threshold:
                success = 1
        except Exception as e:
            mse, nmse, mape, acc_0_1, acc_0_01, acc_0_001, optimized_params = None, None, None, None, None, None, [None] * 10
            success = 0
            
        MSE.append(mse)
        NMSE.append(nmse)
        MAPE.append(mape)
        ACC_0_1.append(acc_0_1)
        ACC_0_01.append(acc_0_01)
        ACC_0_001.append(acc_0_001)
        Optimized_Params.append(optimized_params.tolist() if hasattr(optimized_params, 'tolist') else optimized_params)
        SUCCESS.append(success)
        
    return {
        'equations': equations, 'mse': MSE, 'nmse': NMSE, 'mape': MAPE,
        'acc_0.1': ACC_0_1, 'acc_0.01': ACC_0_01, 'acc_0.001': ACC_0_001,
        'optimized_params': Optimized_Params, 'success': SUCCESS,
    }

def get_performance(equations: List[str], dataset: list, mape_threshold: float = 0.001) -> dict:
    return _execute_performance_evaluation(equations, code_snippet_single_dataset, (dataset,), mape_threshold)

def get_performance_on_test(equations: List[str], train_dataset: list, test_dataset: list, mape_threshold: float = 0.001) -> dict:
    return _execute_performance_evaluation(equations, code_snippet_train_test, (train_dataset, test_dataset), mape_threshold)

def get_score(response_str: str, train_dataset: list, test_dataset: list) -> float:
    parsed_results = parse_equations_and_mses(response_str)

    if not parsed_results:
        return 0.0

    score = 0.0
    
    best_result = min(parsed_results, key=lambda x: x['mse'])
    best_equation = best_result['equation']
    
    test_results_for_best_eq = get_performance_on_test([best_equation], train_dataset, test_dataset, mape_threshold=1.0)
    
    mape = None
    if test_results_for_best_eq.get('mape'):
        mape_value = test_results_for_best_eq['mape'][0]
        if mape_value is not None:
            mape = mape_value

    if mape is not None:
        # A MAPE of 0 indicates a perfect score.
        if np.isclose(mape, 0):
            score = 1.0
        else:
            score = float(np.clip(-1/3 * np.log10(mape), 0, 1))
    else:
        score = 0.0

    return score   


    
@register("sr_scientist")
class SRScientistRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # print(f"begin reward calculation !!!!  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            score = get_score(response_str, data_item.non_tensor_batch['reward_model']['stdin_train'], data_item.non_tensor_batch['reward_model']['stdin_test'])
            
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
                
            if valid_response_length > 0:
                 reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # print(f"end reward calculation !!!! {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor