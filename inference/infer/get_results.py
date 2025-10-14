import re
import json
import ast
from typing import List, Dict
from scipy.optimize import minimize
import numpy as np


exec_globals = {
    'np': np,
    'minimize': minimize
}


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

# Code snippet for fitting on train set and evaluating on test set
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

def _clean_code_block(raw_code_input: str) -> str:

    raw_code = raw_code_input.strip()
    if "```python" in raw_code:
        return raw_code.split("```python")[-1].split("```")[0].strip()
    if "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            potential_code = parts[1]
            if "\n" in potential_code:
                first_line, rest_of_code = potential_code.split("\n", 1)
                return rest_of_code.strip() if first_line.strip().isalpha() else potential_code.strip()
            else:
                return potential_code.strip()
    return raw_code


def parse_equation(messages: dict) -> List[str]:
    extracted_equations = []
    raw_code_strings = []
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            tool_calls = msg.get('tool_calls', [])
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and 'function' in tool_call:
                        function_info = tool_call['function']
                        if (function_info.get('name') == 'equation_evaluator' and 
                            'arguments' in function_info):
                            args = function_info['arguments']
                            if isinstance(args, str):
                                try:
                                    parsed_args = json.loads(args)
                                    if 'equation' in parsed_args:
                                        raw_code_strings.append(parsed_args['equation'])
                                except json.JSONDecodeError:
                                    continue
    
    for code_str in raw_code_strings:
        processed_code = _clean_code_block(code_str)
        if processed_code:
            extracted_equations.append(processed_code)
    
    return extracted_equations

def _execute_performance_evaluation(equations: List[str], code_snippet: str, eval_args: tuple, mape_threshold: float) -> dict:
    """Helper function to execute evaluation based on a given code snippet."""
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
    """
    Evaluates a list of equations on a single dataset (e.g., training set).
    """
    return _execute_performance_evaluation(equations, code_snippet_single_dataset, (dataset,), mape_threshold)

def get_performance_on_test(equations: List[str], train_dataset: list, test_dataset: list, mape_threshold: float = 0.001) -> dict:
    """
    Evaluates equations by fitting on train_dataset and scoring on test_dataset.
    """
    return _execute_performance_evaluation(equations, code_snippet_train_test, (train_dataset, test_dataset), mape_threshold)

def get_final_results(trainset_results: dict, testset_results: dict) -> dict:
    """
    Selects the best equation based on training set MSE and returns its test set performance. (No changes needed)
    """
    train_mses = trainset_results.get('mse', [])
    valid_mses = [(mse, i) for i, mse in enumerate(train_mses) if mse is not None]
    
    if not valid_mses:
        return {
            'equation': None, 'mse': None, 'nmse': None, 'mape': None,
            'acc_0.1': None, 'acc_0.01': None, 'acc_0.001': None,
            'optimized_params': None, 'success': None,
        }
        
    _, best_index = min(valid_mses, key=lambda item: item[0])
    
    final_results = {
        'equation': testset_results['equations'][best_index],
        'mse': testset_results['mse'][best_index],
        'nmse': testset_results['nmse'][best_index],
        'mape': testset_results['mape'][best_index],
        'acc_0.1': testset_results['acc_0.1'][best_index],
        'acc_0.01': testset_results['acc_0.01'][best_index],
        'acc_0.001': testset_results['acc_0.001'][best_index],
        'optimized_params': testset_results['optimized_params'][best_index],
        'success': trainset_results['success'][best_index],
    }
    
    return final_results
