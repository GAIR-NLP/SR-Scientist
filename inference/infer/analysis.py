import json
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

def analyze_results(json_file_path: str):
    """
    Analyzes model response results from a JSON file that includes 'turns',
    provides console statistics, categorizes executed code, and generates a
    detailed log file for any tool calls that produced an error.

    Args:
        json_file_path: The path to the input JSON results file.
    """
    # --- Data Loading ---
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        print("Please ensure you have run the main script and the output file exists.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. The file might be corrupted or empty.")
        return

    # --- Initialization for Analysis ---
    total_responses = 0
    total_tool_calls = 0
    finish_reasons = []
    present_indices = set()
    dataset_stats = defaultdict(lambda: {'tool_calls': 0, 'responses': 0})
    output_distribution = {
        "Has stdout, No stderr": 0, "Has stdout, Has stderr": 0,
        "No stdout,  Has stderr": 0, "No stdout,  No stderr": 0,
    }
    analysis_code_count, evaluation_code_count = 0, 0
    analysis_with_stderr, evaluation_with_stderr = 0, 0
    execution_status_distribution = Counter()

    # --- NEW: Initialization for Turn-Based Analysis ---
    tool_calls_per_turn = Counter()
    responses_per_turn = Counter()
    errors_per_turn = Counter()
    turns_per_problem = {}

    print(f"Analyzing {len(results_data)} examples from '{json_file_path}'...")
    # --- Data Processing Loop ---
    for example in results_data:
        if 'index' in example:
            present_indices.add(example['index'])
        
        dataset_name = example.get('source', 'Unknown_Dataset')
        max_turn_for_example = 0

        if 'responses' in example and isinstance(example['responses'], list):
            dataset_stats[dataset_name]['responses'] += len(example['responses'])

            for response_branch in example['responses']:
                total_responses += 1
                
                # --- NEW: Process Turn Information ---
                turn_num = response_branch.get('turn')
                if turn_num:
                    responses_per_turn[turn_num] += 1
                    max_turn_for_example = max(max_turn_for_example, turn_num)

                if 'finish_reason' in response_branch:
                    finish_reasons.append(response_branch['finish_reason'])

                if 'tool_executions' in response_branch and isinstance(response_branch['tool_executions'], list):
                    num_tool_calls_in_branch = len(response_branch['tool_executions'])
                    total_tool_calls += num_tool_calls_in_branch
                    dataset_stats[dataset_name]['tool_calls'] += num_tool_calls_in_branch
                    if turn_num:
                        tool_calls_per_turn[turn_num] += num_tool_calls_in_branch

                    for tool_call_meta in response_branch['tool_executions']:
                        has_stderr = bool(tool_call_meta.get('stderr'))
                        status = tool_call_meta.get('status', 'Status Not Found')
                        execution_status_distribution[status] += 1
                        
                        if has_stderr and turn_num:
                            errors_per_turn[turn_num] += 1

                        if tool_call_meta['tool_name'] == 'equation_evaluator':
                            evaluation_code_count += 1
                            if has_stderr: evaluation_with_stderr += 1
                        else:
                            analysis_code_count += 1
                            if has_stderr: analysis_with_stderr += 1

                        if bool(tool_call_meta.get('stdout')) and not has_stderr: output_distribution["Has stdout, No stderr"] += 1
                        elif bool(tool_call_meta.get('stdout')) and has_stderr: output_distribution["Has stdout, Has stderr"] += 1
                        elif not bool(tool_call_meta.get('stdout')) and has_stderr: output_distribution["No stdout,  Has stderr"] += 1
                        else: output_distribution["No stdout,  No stderr"] += 1
                        
    
        if max_turn_for_example > 0:
            turns_per_problem[example.get('index')] = max_turn_for_example
    

    # --- Display Console Summary ---
    print("=" * 60)
    print("           Analysis of Model Responses")
    print("=" * 60)

    print("\n## 🛠️ Overall Tool Call Statistics\n")
    avg_tool_calls_per_response = total_tool_calls / total_responses if total_responses > 0 else 0
    print(f"Total responses analyzed: {total_responses}")
    print(f"Total tool calls made: {total_tool_calls}")
    print(f"**Average tool calls per response: {avg_tool_calls_per_response:.2f}**")
    print("-" * 60)

    # --- Average Tool Calls per Dataset Section ---
    print("\n## 📊 Average Tool Calls per Dataset\n")
    dataset_order_for_tool_calls = ['lsr_transform', 'lsr_synth/matsci', 'lsr_synth/chem_react', 'lsr_synth/bio_pop_growth', 'lsr_synth/phys_osc']
    for dataset_name in dataset_order_for_tool_calls:
        stats = dataset_stats.get(dataset_name)
        if stats and stats['responses'] > 0:
            avg_calls = stats['tool_calls'] / stats['responses']
            print(f"  - **{dataset_name}**: {avg_calls:.2f} (Total Calls: {stats['tool_calls']})")
        else:
            print(f"  - **{dataset_name}**: No tool calls recorded.")
    print("-" * 60)

    # --- NEWLY ADDED: Finish Reason Distribution ---
    print("\n## 🛑 Finish Reason Distribution\n")
    finish_reason_distribution = Counter(finish_reasons)
    if finish_reason_distribution:
        for reason, count in finish_reason_distribution.most_common():
            percentage = (count / total_responses) * 100 if total_responses > 0 else 0
            print(f"  - **{reason}**: {count} times ({percentage:.2f}%)")
    else:
        print("No finish reasons were recorded.")
    print("-" * 60)

    # --- NEWLY ADDED: Tool Call Output & Error Distribution ---
    print("\n## 📊 Tool Call Output & Error Distribution\n")
    if total_tool_calls > 0:
        dist_data = {
            "Case": list(output_distribution.keys()),
            "Count": list(output_distribution.values()),
            "Percentage": [(v / total_tool_calls) * 100 for v in output_distribution.values()]
        }
        df = pd.DataFrame(dist_data)
        df["Percentage"] = df["Percentage"].map('{:.2f}%'.format)
        print(df.to_string(index=False))
    else:
        print("No tool calls were made.")
    print("-" * 60)

    # --- Turn-Based Analysis (Unique to this script) ---
    print("\n## 🔄 Turn-Based Analysis\n")
    if responses_per_turn:
        max_turn = max(responses_per_turn.keys())
        turn_data = {
            "Turn": [t for t in range(1, max_turn + 1)],
            "Total Responses": [responses_per_turn.get(t, 0) for t in range(1, max_turn + 1)],
            "Total Tool Calls": [tool_calls_per_turn.get(t, 0) for t in range(1, max_turn + 1)],
            "Total Errors (stderr)": [errors_per_turn.get(t, 0) for t in range(1, max_turn + 1)],
        }
        turn_df = pd.DataFrame(turn_data)
        turn_df["Avg. Calls/Response"] = (turn_df["Total Tool Calls"] / turn_df["Total Responses"]).fillna(0).map('{:.2f}'.format)
        print("--- Per-Turn Statistics ---")
        print(turn_df.to_string(index=False))

        # --- Success Rate by Turns ---
        success_by_turns = defaultdict(lambda: {'success': 0, 'total': 0})
        for res in results_data:
            num_turns = turns_per_problem.get(res.get('index'))
            if num_turns:
                success_by_turns[num_turns]['total'] += 1
                if res.get('submitted_results', {}).get('success') == 1.0:
                    success_by_turns[num_turns]['success'] += 1
        
        if success_by_turns:
            success_data = {
                "Turns Taken": sorted(success_by_turns.keys()),
                "Num Problems": [success_by_turns[t]['total'] for t in sorted(success_by_turns.keys())],
                "Num Successful": [success_by_turns[t]['success'] for t in sorted(success_by_turns.keys())]
            }
            success_df = pd.DataFrame(success_data)
            success_df["Success Rate"] = ((success_df["Num Successful"] / success_df["Num Problems"]) * 100).map('{:.2f}%'.format)
            print("\n--- Success Rate by Number of Turns Taken ---")
            print(success_df.to_string(index=False))

    else:
        print("No turn-based data was found.")
    print("-" * 60)

    # --- Overall Code Execution Status (Simplified View) ---
    print("\n## 📈 Overall Code Execution Status (Simplified View)\n")
    if total_tool_calls > 0:
        status_data = {
            "Category": ["Analysis - Correct", "Analysis - Error", "Evaluation - Correct", "Evaluation - Error"],
            "Count": [analysis_code_count - analysis_with_stderr, analysis_with_stderr, evaluation_code_count - evaluation_with_stderr, evaluation_with_stderr],
            "Percentage of Total": [(c / total_tool_calls) * 100 for c in [analysis_code_count - analysis_with_stderr, analysis_with_stderr, evaluation_code_count - evaluation_with_stderr, evaluation_with_stderr]]
        }
        status_df = pd.DataFrame(status_data)
        status_df["Percentage of Total"] = status_df["Percentage of Total"].map('{:.2f}%'.format)
        print(status_df.to_string(index=False))
    else:
        print("No tool calls were made to analyze.")
    print("-" * 60)

    # --- Final Submitted Results Sections ---
    print("\n" + "="*60)
    print("     Final Submitted Results Statistics by Dataset")
    print("="*60)
    
    grouped_results = defaultdict(list)
    for res in results_data:
        if res.get('submitted_results') and isinstance(res['submitted_results']['mse'], (int,float)):
            grouped_results[res['source']].append(res['submitted_results'])
            
    dataset_order = ['lsr_transform', 'lsr_synth/matsci', 'lsr_synth/chem_react', 'lsr_synth/bio_pop_growth', 'lsr_synth/phys_osc']
    
    for dataset_name in dataset_order:
        valid_submissions = grouped_results.get(dataset_name)
        print(f"\n--- 📊 Statistics for: {dataset_name} ---")
        if not valid_submissions:
            print("No valid results found for this dataset.")
            continue
        print(f"Total number of valid results: {len(valid_submissions)}")
        avg_metrics = {}
        for key in sorted(list(set().union(*(sub.keys() for sub in valid_submissions)))):
            values = [sub[key] for sub in valid_submissions if key in sub and isinstance(sub[key], (int, float))]
            if values: avg_metrics[key] = np.mean(values)
        print("Average values for key metrics:")
        if avg_metrics:
            ordered_keys = ['acc_0.1', 'acc_0.01', 'acc_0.001', 'success']
            
            for key in ordered_keys:
                if key in avg_metrics:
                    avg_value = avg_metrics[key]
                    print(f"  - **{key}**: {avg_value:.2%}")
            
            remaining_keys = sorted([k for k in avg_metrics if k not in ordered_keys])
            for key in remaining_keys:
                avg_value = avg_metrics[key]
                if key in ['nmse', 'mse']:
                    print(f"  - **{key}**: {avg_value:.2e}")
                elif key == 'mape':
                     print(f"  - **{key}**: {avg_value:.2%}")
        else: print("  No numeric metrics found to average.")
    
    # --- NEWLY ADDED: Overall Final Statistics for the Entire Test Set ---
    print("\n" + "="*60)
    print("--- 🌍 Overall Final Statistics for the Entire Test Set ---")
    print("="*60 + "\n")
    
    all_valid_submissions = [
        res['submitted_results'] for res in results_data
        if res.get('submitted_results') and isinstance(res['submitted_results'].get('mse'), (int, float))
    ]
    
    if not all_valid_submissions:
        print("No valid results found in the entire test set.")
    else:
        num_total_valid = len(all_valid_submissions)
        print(f"Total number of valid results across all datasets: {num_total_valid}")
        
        overall_keys = set().union(*(sub.keys() for sub in all_valid_submissions))
        overall_avg_metrics = {}
        for key in sorted(list(overall_keys)):
            values = [sub[key] for sub in all_valid_submissions if key in sub and isinstance(sub[key], (int, float))]
            if values:
                overall_avg_metrics[key] = np.mean(values)
                
        print("\nAverage values for key metrics across the entire test set:")
        if overall_avg_metrics:
            ordered_keys = ['acc_0.1', 'acc_0.01', 'acc_0.001', 'success']
            
            for key in ordered_keys:
                if key in overall_avg_metrics:
                    avg_value = overall_avg_metrics[key]
                    print(f"  - **{key}**: {avg_value:.2%}")
            
            remaining_keys = sorted([k for k in overall_avg_metrics if k not in ordered_keys])
            for key in remaining_keys:
                avg_value = overall_avg_metrics[key]
                if key in ['nmse', 'mse']:
                    print(f"  - **{key}**: {avg_value:.2e}")
                elif key == 'mape':
                     print(f"  - **{key}**: {avg_value:.2%}")
        else:
            print("  No numeric metrics found to average.")

    print("\n" + "="*60)
    print("## ⚙️ Tool Call Execution Status Distribution\n")
    if total_tool_calls > 0 and execution_status_distribution:
        sorted_statuses = execution_status_distribution.most_common()
        
        status_data = {
            "Status": [status for status, count in sorted_statuses],
            "Count": [count for status, count in sorted_statuses],
            "Percentage of Total Calls": [(count / total_tool_calls) * 100 for status, count in sorted_statuses]
        }
        status_df = pd.DataFrame(status_data)
        status_df["Percentage of Total Calls"] = status_df["Percentage of Total Calls"].map('{:.2f}%'.format)
        print(status_df.to_string(index=False))
    else:
        print("No execution statuses were recorded.")
    



if __name__ == '__main__':
    # You can change the file path to point to your specific results JSON file.
    output_file = 'output_file_path'
    analyze_results(output_file)