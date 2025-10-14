import json
import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from .prompt import SYSTEM_PROMPT, USER_PROMPT_WITH_MEMORY
from .get_results import parse_equation, get_performance, get_performance_on_test, get_final_results
from .tool import tools, get_function_by_name
import numpy as np
import os
from collections import defaultdict
import asyncio
import time
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any
import itertools
import concurrent.futures
import re
import uuid
import types
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
import signal
import traceback
import sys
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    stream=sys.stdout
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        """
        Custom JSON encoder for numpy types.
        """
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating) or isinstance(o, np.longdouble):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)



@dataclass
class Branch:
    branch_id: int
    example_index: int
    status: str  # 'pending_model', 'in_flight_model', 'in_flight_tool', 'completed'
    messages: List[Dict]
    mape_threshold: float
    tool_calls: List[Any] = field(default_factory=list)
    tool_executions: List[Dict] = field(default_factory=list)
    finish_reason: str = None
    source: str = None
    expression: str = None
    assistant_turn_count: int = 0

    final_train_results: Dict = field(default_factory=dict)
    final_test_results: Dict = field(default_factory=dict)

class BranchManager:
    def __init__(self):
        """
        Initialize the BranchManager with an empty dictionary for branches and a starting ID.
        """
        self.branches = {}
        self.next_branch_id = 0

    def create_initial_branches(self, example, num_responses, previous_turn_context, mape_threshold):
        """
        Create the initial set of branches for a given example, constructing the system and user prompts.
        """
        var_name = ["output"]
        var_desc = [example['symbol_descs'][0]]
        for v_idx, v_prop in enumerate(example['symbol_properties']):
            if v_idx == 0:
                continue
            if "V" in v_prop:
                var_name.append(example['symbols'][v_idx])
                var_desc.append(example['symbol_descs'][v_idx])
        var_name = [n.strip("$").strip("\\") for n in var_name]
        var_name = [n.replace(" ", "_").replace("text", "") for n in var_name]

        if len(var_desc) > 2:
            input_desc = ", ".join(var_desc[1:-1]) + ", and " + var_desc[-1]
        else:
            input_desc = var_desc[-1]

        part1 = f"Find the mathematical function skeleton that represents {var_desc[0]}, given data on {input_desc}.\n"
        part2 = "def equation(" + ", ".join([f"{name}: np.ndarray" for name in  var_name[1:]]) + ", params: np.ndarray) -> np.ndarray:\n" + \
            f'    """ Mathematical function for {var_desc[0]}\n\n' + \
            '    Args:\n' + \
            "\n".join([f"        {name}: A numpy array representing observations of {desc}." for name, desc in zip(var_name[1:], var_desc[1:])]) + "\n" + \
            "        params: Array of numeric constants or parameters to be optimized\n\n" + \
            "    Return:\n" + \
            f"        A numpy array representing {var_desc[0]} as the result of applying the mathematical function to the inputs.\n" + \
            '    """\n' + \
            f"    {var_name[0]} = " + " + ".join([f"params[{i}] * {name}" for i, name in enumerate(var_name[1:])]) + f" + params[{len(var_name[1:])}]\n" + \
            f"    return {var_name[0]}\n\n"

        mape_threshold_percentage = f"{mape_threshold:.4%}"
        system_prompt = SYSTEM_PROMPT.format(
            part2=part2,
            mape_threshold_percentage=mape_threshold_percentage,
            mape_threshold=mape_threshold
        )
        question = USER_PROMPT_WITH_MEMORY.format(
            part1=part1,
            part2=part2,
            previous_turn_context=previous_turn_context,
            mape_threshold_percentage=mape_threshold_percentage
        )
        initial_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

        for _ in range(num_responses):
            branch = Branch(
                branch_id=self.next_branch_id,
                example_index=example.name,
                status='pending_model',
                messages=list(initial_messages),
                mape_threshold=mape_threshold,
                source=example['dataset_identifier'],
                expression=example['expression']
            )
            self.branches[self.next_branch_id] = branch
            self.next_branch_id += 1

    def get_branches_by_status(self, status):
        """
        Return a list of branches that match the given status.
        """
        return [b for b in self.branches.values() if b.status == status]

    def active_branches_exist(self):
        """
        Check if there are any active (not completed) branches.
        """
        return any(b.status != 'completed' for b in self.branches.values())

def evaluate_branch_worker(branch, example_data):
    """
    Worker function to evaluate a single branch in a separate process.
    It parses equations from messages and calculates their performance on train and test data.
    """
    equations = list(dict.fromkeys(parse_equation(branch.messages[2:])))
    mape_threshold = branch.mape_threshold
    train_results = get_performance(equations, example_data['samples']['train'], mape_threshold)
    test_results = get_performance_on_test(equations, example_data['samples']['train'], example_data['samples']['test'], mape_threshold)

    return branch.branch_id, train_results, test_results


async def execute_tool_with_semaphore(tool_call, fn_name, fn_args, example_data, sandbox_url, semaphore, mape_threshold):
    """
    Execute a tool call asynchronously while managing concurrency with a semaphore.
    Inject the MAPE threshold into the arguments for the equation_evaluator tool.
    """
    async with semaphore:
        start_time = time.time()
        if fn_name == 'equation_evaluator':
            fn_args['mape_threshold'] = mape_threshold

        tool_func = get_function_by_name(fn_name)
        fn_res, meta_data = await tool_func(
            **fn_args,
            stdin_data=example_data['samples']['train'],
            sandbox_fusion_url=sandbox_url
        )
        duration = time.time() - start_time
        return tool_call, fn_res, meta_data, duration

async def process_branch_with_model(branch, client, model_name, tools, semaphore):
    """
    Send the message history of a branch to the model API to get a response.
    Manages concurrency using a semaphore to avoid overwhelming the API endpoint.
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                    model=model_name,
                    messages=branch.messages,
                    tools=tools,
                    n=1,
                    timeout=1200,
                    max_completion_tokens = 8192,
                    temperature = 0.7,
                    top_p = 1.0,
                )
            return branch, response
        except Exception as e:
            logging.error(f"Branch {branch.branch_id}: API call failed with exception: {e}", exc_info=True)
            return branch, e


def _create_previous_turn_context(agg_data: Dict, top_k: int) -> str:
    """
    Generate a formatted string of the top-k performing equations from the previous turn
    to be used as context for the next model prompt.
    """
    if not agg_data['train']['mape']:
        return ""

    valid_results = sorted(
        [(m, e) for m, e in zip(agg_data['train']['mape'], agg_data['train']['equations']) if m is not None],
        key=lambda x: x[0]
    )
    if not valid_results:
        return ""

    distinct_results: List[Tuple[float, str]] = []
    last_accepted_mape = float('-inf')
    min_gap = 0.005  # MAPE < 0.5%
    # Ensure different ICL examples
    for mape, eq in valid_results:
        if mape - last_accepted_mape >= min_gap:
            distinct_results.append((mape, eq))
            last_accepted_mape = mape
            if len(distinct_results) >= top_k:
                break

    if not distinct_results:
        return ""

    context_lines = [
        f"\n\nIn the previous turn, some equations have been explored.",
        "Their performance (MAPE score) is listed below, sorted from lowest to highest error.",
        "You can use these experiences as inspiration to help you design a better equation.",
        "\n--- Previously Explored Equations ---"
    ]

    for mape, eq in distinct_results:
        context_lines.append(f"\n# MAPE: {mape:.4%}\n{eq.strip()}")

    context_lines.append("\n-------------------------------------\n")
    return "\n".join(context_lines)



async def run_inference_batch(model_name, model_url, api_key, sandbox_urls, parquet_file_path, source, mape_threshold, num_turns, top_k, output_json_path, max_assistant_turns):
    """
    Orchestrate the multi-turn inference process for a batch of examples.
    This involves managing branches of exploration for each example, calling the model,
    executing tools, evaluating results, and dynamically adjusting goals between turns.
    """
    # --- Initialization ---
    sandbox_url_cycler = itertools.cycle(sandbox_urls)
    print(f"Enabling {len(sandbox_urls)} Sandbox instances for load balancing.")

    client = AsyncOpenAI(api_key=api_key, base_url=model_url)

    CONCURRENCY_LIMIT_MODEL = 512
    semaphore_model = asyncio.Semaphore(CONCURRENCY_LIMIT_MODEL)
    CONCURRENCY_LIMIT_TOOL = 64
    semaphore_tool = asyncio.Semaphore(CONCURRENCY_LIMIT_TOOL)

    df = pd.read_parquet(parquet_file_path)
    examples_to_process=[]
    print(source)
    for _, row in df.iterrows():
        if not source or row['dataset_identifier'] in source:
            examples_to_process.append(row)


    final_results_by_example = defaultdict(lambda: {
        'train': defaultdict(list),
        'test': defaultdict(list),
        'responses': [],
        'turn_success_history': []
    })


    example_mape_targets = {ex.name: mape_threshold for ex in examples_to_process}
    completed_example_indices = set()

    # --- Main Turn Loop ---
    for turn in range(num_turns):
        current_turn_examples = [ex for ex in examples_to_process if ex.name not in completed_example_indices]
        if not current_turn_examples:
            print("\nAll examples have been successfully completed. Ending turns early.")
            break

        print(f"\n{'='*25} Starting Turn {turn + 1}/{num_turns} ({len(current_turn_examples)} examples) {'='*25}")

        manager = BranchManager()

        # --- Branch Creation for Current Turn ---
        for example in current_turn_examples:
            example_index = example.name
            agg_data = final_results_by_example[example_index]
            current_mape_threshold = example_mape_targets[example_index]

            previous_turn_context = _create_previous_turn_context(agg_data, top_k) if turn > 0 else ""
            if turn > 0:
                print(f"  - Example {example_index}: Target MAPE < {current_mape_threshold:.5%}. Context provided.")

            manager.create_initial_branches(
                example,
                1,
                previous_turn_context=previous_turn_context,
                mape_threshold=current_mape_threshold
            )

        pbar = tqdm(total=len(manager.branches), desc=f"✅ Turn {turn+1} Completed", position=0)
        tool_pbar = tqdm(position=1, bar_format="{desc}")
        tool_pbar.set_description_str("🔧 Avg Tool Time: N/A")
        completed_tool_calls = 0
        total_tool_time = 0.0

        async def handle_tool_calls_for_branch(branch, df, sandbox_url_cycler, semaphore, pbar):
            """
            Handle the execution of tool calls for a single branch.
            """
            nonlocal completed_tool_calls, total_tool_time
            try:
                example_data = df.loc[branch.example_index]
                tool_tasks = [
                    asyncio.create_task(
                        execute_tool_with_semaphore(
                            tc, tc.function.name, json.loads(tc.function.arguments),
                            example_data, next(sandbox_url_cycler), semaphore, branch.mape_threshold
                        )
                    ) for tc in branch.tool_calls
                ]

                for tool_call, fn_res, meta_data, duration in await asyncio.gather(*tool_tasks):
                    completed_tool_calls += 1
                    total_tool_time += duration
                    avg_time = total_tool_time / completed_tool_calls
                    tool_pbar.set_description_str(f"🔧 Avg Tool Time: {avg_time:.3f}s ({completed_tool_calls} calls)")
                    branch.tool_executions.append(meta_data)
                    branch.messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": json.dumps(fn_res)})

                branch.tool_calls = []
                branch.status = 'pending_model'
            except Exception as e:
                logging.error(f"Branch {branch.branch_id}: Unhandled error in handle_tool_calls_for_branch: {e}", exc_info=True)
                print(f"Error handling tools for branch {branch.branch_id}: {e}")
                branch.status = 'completed'
                pbar.update(1)

        # --- Core Processing Loop for the Turn ---
        while manager.active_branches_exist():
            statuses = [b.status for b in manager.branches.values()]
            status_counts = defaultdict(int)
            for s in statuses:
                status_counts[s] += 1
            logging.info(f"Main loop check: Total active branches={len(statuses)}. Statuses={dict(status_counts)}")

            # --- Process Branches Pending Model Call ---
            model_pending_branches = manager.get_branches_by_status('pending_model')
            if model_pending_branches:
                for branch in model_pending_branches:
                    branch.assistant_turn_count += 1
                model_tasks = [asyncio.create_task(process_branch_with_model(b, client, model_name, tools, semaphore_model)) for b in model_pending_branches]
                for branch in model_pending_branches: branch.status = 'in_flight_model'

                for future in tqdm_asyncio.as_completed(model_tasks, desc=f"📞 Turn {turn+1} API Calls", position=2, leave=False):
                    branch, result = await future
                    if isinstance(result, Exception):
                        logging.error(f"Model call for branch {branch.branch_id} resulted in an exception object: {result}")
                        print(f"Model call for branch {branch.branch_id} failed: {result}")
                        branch.status = 'completed'; pbar.update(1); continue

                    response_message = result.choices[0].message
                    branch.finish_reason = result.choices[0].finish_reason
                    if response_message.content and 'to=functions.' in response_message.content and '<|call|>' in response_message.content:
                        branch.finish_reason = 'tool_calls'

                    # --- Handle Model Response: Tool Calls ---
                    if branch.finish_reason == 'tool_calls':
                        if branch.assistant_turn_count >= max_assistant_turns:
                            branch.messages.append(response_message.model_dump())
                            logging.warning(f"Branch {branch.branch_id} reached max assistant turns ({max_assistant_turns}). Forcing completion.")
                            print(f"Branch {branch.branch_id} reached max assistant turns ({max_assistant_turns}). Forcing completion for this turn.")
                            branch.status = 'completed'
                            branch.finish_reason = 'turn_limit'
                            pbar.update(1)
                        else:
                            if not response_message.tool_calls:
                                # Fix special cases for models that generate malformed tool calls in content (gpt-oss)
                                if response_message.content and 'to=functions.' in response_message.content and '<|call|>' in response_message.content:
                                    print(f"INFO: Branch {branch.branch_id} - Fixing malformed tool call from content.")
                                    pattern = re.compile(r"to=functions\.(?P<name>\w+)[\s\S]*?<\|message\|>(?P<args>.*?)\s*<\|call\|>", re.DOTALL)
                                    match = pattern.search(response_message.content)

                                    if match:
                                        tool_name = match.group('name')
                                        tool_args_str = match.group('args').strip()
                                        if tool_args_str.endswith(']}'):
                                            print(f"INFO: Branch {branch.branch_id} -  Cleaning trailing ]" + "} from arguments.")
                                            tool_args_str = tool_args_str[:-2] + '}'
                                        elif 'timeout' in tool_args_str:
                                            tool_args_str = re.sub(r'],\s*"timeout"\s*:\s*\d+', '', tool_args_str)
                                            print(f"INFO: Branch {branch.branch_id} -  Cleaning timeout from arguments.")
                                        elif tool_name == 'data_analyzer' and """{\"code\":""" not in tool_args_str:
                                            print(f"INFO: add key.")
                                            tool_args_str = json.dumps({"code": tool_args_str})
                                        elif tool_args_str.endswith('"'):
                                            print(f"INFO: add "+ "} to arguments.")
                                            tool_args_str = tool_args_str+ '}'

                                        mock_function = {
                                            "name": tool_name,
                                            "arguments": tool_args_str
                                            }
                                        mock_tool_call = {
                                            "id": f"call_{uuid.uuid4().hex}",
                                            "function": mock_function,
                                            "type": "function",
                                            "index": None
                                        }

                                        function_obj = Function(name=tool_name,arguments=tool_args_str)
                                        mock_tool_call = ChatCompletionMessageToolCall(
                                            id=f"call_{uuid.uuid4().hex}",
                                            function=function_obj,
                                            type="function"
                                        )
                                        response_message.tool_calls = [mock_tool_call]
                                        response_message.content = None
                                    else:
                                        print(f"WARNING: Branch {branch.branch_id} - Malformed tool call detected but could not be parsed.")
                            if response_message.tool_calls:
                                branch.tool_calls = response_message.tool_calls
                                branch.messages.append(response_message.model_dump())
                                branch.status = 'in_flight_tool'
                                asyncio.create_task(handle_tool_calls_for_branch(branch, df, sandbox_url_cycler, semaphore_tool, pbar))
                            else:
                                branch.status = 'completed'
                                pbar.update(1)
                    # --- Handle Model Response: Stop or Length ---
                    else:
                        branch.messages.append(response_message.model_dump())
                        branch.status = 'completed'
                        pbar.update(1)

            if not manager.get_branches_by_status('pending_model'): await asyncio.sleep(0.1)

        pbar.close(); tool_pbar.close()

        # --- Final Evaluation of Turn Results ---
        print(f"\nEvaluating final equations for Turn {turn + 1}...")
        completed_branches = [b for b in manager.branches.values() if b.status == 'completed']
        branch_map = {b.branch_id: b for b in completed_branches}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            logging.info(f"Submitting {len(completed_branches)} branches for final evaluation in process pool.")
            futures = {executor.submit(evaluate_branch_worker, b, df.loc[b.example_index]): b.branch_id for b in completed_branches}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Final Evaluation"):
                branch_id, (returned_id, train_res, test_res) = futures[future], future.result()
                original_branch = branch_map[returned_id]
                original_branch.final_train_results = train_res if train_res else {}
                original_branch.final_test_results = test_res if test_res else {}

        # --- Aggregate Turn Results and Update for Next Turn ---
        turn_success_flags = defaultdict(bool)
        best_mape_this_turn = {}

        for branch in manager.branches.values():
            if branch.status == 'completed':
                index = branch.example_index
                final_results_by_example[index]['responses'].append({
                    "turn": turn + 1,
                    "messages": branch.messages,
                    "finish_reason": branch.finish_reason,
                    "tool_executions": branch.tool_executions,
                    "trainset_results": branch.final_train_results,
                    "testset_results": branch.final_test_results,
                })
                if branch.final_train_results.get('equations'):
                    for key in branch.final_train_results:
                        final_results_by_example[index]['train'][key].extend(branch.final_train_results.get(key, []))
                        final_results_by_example[index]['test'][key].extend(branch.final_test_results.get(key, []))

                    successful_mapes = [m for m, s in zip(branch.final_train_results.get('mape', []), branch.final_train_results.get('success', [])) if s == 1 and m is not None]
                    if successful_mapes:
                        turn_success_flags[index] = True
                        min_mape_in_branch = min(successful_mapes)
                        if index not in best_mape_this_turn or min_mape_in_branch < best_mape_this_turn[index]:
                            best_mape_this_turn[index] = min_mape_in_branch

        for idx in [e.name for e in current_turn_examples]:
            final_results_by_example[idx]['turn_success_history'].append(turn_success_flags[idx])
            if turn_success_flags[idx]:
                best_mape = best_mape_this_turn[idx]
                if best_mape < 0.000001:  # MAPE < 0.0001%
                    print(f"  - Example {idx}: Completed with MAPE {best_mape:.4e}. Will not run in next turn.")
                    completed_example_indices.add(idx)
                else:
                    # Update goal for the next turn
                    next_target = 10**np.floor(np.log10(best_mape))
                    if next_target >= best_mape:
                         next_target /= 10.0
                    example_mape_targets[idx] = next_target
                    print(f"  - Example {idx}: Success! Best MAPE this turn: {best_mape:.4%}. Next target: {example_mape_targets[idx]:.4%}")
            else:
                print(f"  - Example {idx}: No improvement. Target MAPE remains {example_mape_targets[idx]:.4%}")

    # --- Final Aggregation and Saving ---
    print("\nAll turns completed. Selecting best overall result for each example...")
    saved_results = []
    for index, aggregated_data in tqdm(final_results_by_example.items(), desc="Aggregating Results"):
        submitted_results = get_final_results(aggregated_data['train'], aggregated_data['test'])
        saved_results.append({
            "index": index,
            "source": df.loc[index]['dataset_identifier'],
            "expression": df.loc[index]['expression'],
            "responses": aggregated_data['responses'],
            "submitted_results": submitted_results,
        })


    saved_results.sort(key=lambda x: x['index'])
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(saved_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    print(f"\nAll results have been successfully saved to: {output_json_path}")

