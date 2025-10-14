SYSTEM_PROMPT = """You are a helpful assistant tasked with discovering mathematical functions that model scientific systems.
You will have access to a dataset of physical measurements.
Your goal is to determine the correct equation, implement it as a Python function, and optimize it until the mean absolute percentage error is less than {mape_threshold_percentage}.
You should use the `equation_evaluator` tool to evaluate the equation's goodness of fit and the `data_analyzer` tool to write code for data analysis.

For the `equation_evaluator`, it is a code interpreter that wraps your function with the following code:

```python
import numpy as np
import sys
import json

# Initialize parameters
MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS

# Example of a user-provided equation
{part2}

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
````

As shown, the `equation_evaluator` tool assesses your equation's goodness of fit. It uses SciPy's BFGS optimizer to find the optimal parameters for your equation based on the dataset. It then provides an output with performance metrics (Mean Squared Error (MSE), Normalized Mean Squared Error (NMSE), and Mean Absolute Percentage Error (MAPE)), the success status, and details of any bugs.
In utilizing the tool, you only need to pass the entire function, including the function header, to the tool.

For the `data_analyzer` tool, it is a code interpreter that can run your data exploration snippet. You can access the data as shown in the example.
However, you are forbidden from using any libraries like Matplotlib for plotting figures for analysis.

```python
import json
import sys

# load the data
input_data_str = sys.stdin.read()
data_list = json.loads(input_data_str)

# print the first 5 entries
# In each entry of data_list, the first value is the output to predict, and the rest are the inputs.
print(data_list[:5])
```
"""

USER_PROMPT = """
{part1}

Follow these steps to solve the problem:

**1. Implement the Equation in Code**

  * Based on your knowledge and analysis, identify the standard equation and implement it in the code.
  * Your equation will likely have one or more constants. Use elements from the `params` list (e.g., `params[0]`, `params[1]`, `params[2]`) to represent these constants, as the `equation_evaluator` tool is designed to optimize them. Note that the `params` list has a fixed size of 10 (`MAX_NPARAMS = 10`), so you can use up to 10 parameters in your model.

**2. Test, Analyze, and Refine**

  * Evaluate the equation's goodness of fit using the `equation_evaluator` tool.
    You need to pass the entire function, including the function header, to the tool. Here is an example:


```python
{part2}
```

You can modify the function body, but the function header must remain unchanged.

  * Your goal is to reduce the mean absolute percentage error to less than {mape_threshold_percentage}. Meeting this condition indicates that your equation is a good fit for the data.
  * If this goal is not met, refine your equation in Python and observe its performance. You can write your own data exploration snippet and use the `data_analyzer` tool to execute it, allowing you to inspect the data for potential relationships or anomalies.

**3. Submit Your Final Answer**

  * Once you are confident your equation has met the condition, or if you conclude after numerous attempts that you cannot meet it, provide the completed Python function as your answer.
"""


USER_PROMPT_WITH_MEMORY = """
{part1}

Follow these steps to solve the problem:

**1. Implement the Equation in Code**

  * Based on your knowledge and analysis, identify the standard equation and implement it in the code.
  * Your equation will likely have one or more constants. Use elements from the `params` list (e.g., `params[0]`, `params[1]`, `params[2]`) to represent these constants, as the `equation_evaluator` tool is designed to optimize them. Note that the `params` list has a fixed size of 10 (`MAX_NPARAMS = 10`), so you can use up to 10 parameters in your model.

**2. Test, Analyze, and Refine**

  * Evaluate the equation's goodness of fit using the `equation_evaluator` tool.
    You need to pass the entire function, including the function header, to the tool. Here is an example:


```python
{part2}
```

You can modify the function body, but the function header must remain unchanged.

  * Your goal is to reduce the mean absolute percentage error to less than {mape_threshold_percentage}. Meeting this condition indicates that your equation is a good fit for the data.
  * If this goal is not met, refine your equation in Python and observe its performance. You can write your own data exploration snippet and use the `data_analyzer` tool to execute it, allowing you to inspect the data for potential relationships or anomalies.

**3. Submit Your Final Answer**

  * Once you are confident your equation has met the condition, or if you conclude after numerous attempts that you cannot meet it, provide the completed Python function as your answer.

{previous_turn_context}
"""





