"""Entry point for running TIMEBench evaluations against a vLLM-served model."""

import argparse
import copy
import json
import os

import numpy as np
from tqdm import tqdm

from timebench.utils.testclient import ChatCompletionClient


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM benchmarks with reproducibility, resume support, "
            "and configurable sampling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- File and run configuration ---
    parser.add_argument(
        "-i", "--input-file",
        type=str,
        default="tests/scenarios.json",
        help="Path to the input JSON file with test scenarios.",
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default=None,
        help="Path to the output JSON file. Defaults to 'Outputs/<model_name>_output.json'.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of times to run each scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Master seed for generating reproducible per-run seeds.",
    )
    parser.add_argument(
        "--redo",
        type=int,
        nargs="+",
        default=[],
        help="Space-separated list of scenario indices (0-based) to force re-running.",
    )
    parser.add_argument(
        "--no-burn-in",
        dest="burn_in",
        action="store_false",
        help="Disable the default burn-in run. Burn-in is ON by default.",
    )

    # --- Model and API configuration ---
    parser.add_argument(
        "--model",
        type=str,
        default="TIME-32B",
        help="Name of the model to benchmark.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--content-suffix",
        type=str,
        default=None,
        help=(
            "String to append to the last message's content "
            r"(e.g., '\n\nDetailed thinking:')."
        ),
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("API_BASE", "http://0.0.0.0:8000/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY", "some_random_key"),
        help="API key for the inference server.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_scenario_identifier(scenario: dict) -> str:
    """Return a stable, unique identifier for a scenario based on its messages."""
    return json.dumps(scenario.get("test_messages", []), sort_keys=True)


def load_scenarios(file_path: str) -> list:
    """Load scenarios from *file_path*, creating any missing parent directories."""
    if not os.path.exists(file_path) and "tests/scenarios.json" in file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Info: Default directory '{os.path.dirname(file_path)}' created.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found. Exiting.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Exiting.")
        exit(1)


def generate_seeds(
    num_scenarios: int,
    runs_per_scenario: int,
    master_seed: int,
) -> np.ndarray:
    """Return a deterministic array of per-run seeds derived from *master_seed*."""
    total = num_scenarios * runs_per_scenario
    print(f"Generating {total} seeds using master seed {master_seed}...")
    rng = np.random.default_rng(master_seed)
    return rng.integers(low=0, high=2**32 - 1, size=total)


def resume_from_output(output_file: str) -> tuple:
    """Load existing results and return ``(successful_results, completed_task_ids)``."""
    if not os.path.exists(output_file):
        print("Info: No existing output file found. Starting a new run.")
        return [], set()

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

        successful_results = []
        completed_tasks = set()
        for result in existing_results:
            failed = result.get("response", "").startswith("API_CALL_FAILED:")
            if "seed" in result and not failed:
                successful_results.append(result)
                completed_tasks.add((get_scenario_identifier(result), result["seed"]))

        if completed_tasks:
            failed_count = len(existing_results) - len(successful_results)
            msg = f"Info: Found {len(successful_results)} successful results"
            if failed_count > 0:
                msg += f" and {failed_count} failed entries (will be retried)"
            print(f"{msg} in '{output_file}'.")
            return successful_results, completed_tasks

        print("Info: Output file exists but contains no successful results. Starting fresh.")
        return [], set()

    except (json.JSONDecodeError, IOError) as exc:
        print(
            f"Warning: Could not read resume file '{output_file}': {exc}. "
            "Starting new run."
        )
        return [], set()


def save_incremental_results(results: list, file_path: str) -> None:
    """Write *results* to *file_path* after every task for crash-safety."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def sort_and_finalize_output(
    results: list,
    file_path: str,
    task_lookup: dict,
) -> None:
    """Sort *results* by original task order, then write the final output file."""
    print("\nAll tasks complete. Preparing final output...")

    for result in results:
        result_id = (get_scenario_identifier(result), result.get("seed"))
        result["task_index"] = task_lookup.get(result_id, float("inf"))

    results.sort(key=lambda x: x.get("task_index", float("inf")))

    for result in results:
        result.pop("task_index", None)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Running scenarios complete. Final results saved to '{file_path}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full evaluation pipeline."""
    args = get_args()

    # Resolve output file path
    if args.output_file is None:
        safe_model_name = args.model.replace("/", "_")
        if safe_model_name.endswith(".gguf"):
            safe_model_name = safe_model_name[:-5]
        args.output_file = f"Outputs/{safe_model_name}_output.json"
        print(f"Info: Output file not specified. Defaulting to '{args.output_file}'.")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not args.api_key:
        print("Error: API key not provided. Set env var or use --api-key.")
        return

    print(
        f"Sampling parameters: temperature={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}"
    )

    processed_suffix = None
    if args.content_suffix:
        processed_suffix = args.content_suffix.encode("utf-8").decode("unicode_escape")
        print(f"Applying content suffix to all prompts: '{processed_suffix}'")

    if args.burn_in:
        print("Info: Burn-in mode is enabled. Use --no-burn-in to disable.")

    scenarios = load_scenarios(args.input_file)
    all_seeds = generate_seeds(len(scenarios), args.runs, args.seed)
    results, completed_tasks = resume_from_output(args.output_file)

    # Build a lookup from (scenario_id, seed) → absolute task index for final ordering
    task_lookup: dict = {}
    for i, scenario in enumerate(scenarios):
        scenario_id = get_scenario_identifier(scenario)
        for j in range(args.runs):
            task_index = i * args.runs + j
            current_seed = int(all_seeds[task_index])
            task_lookup[(scenario_id, current_seed)] = task_index

    scenario_id_to_index = {
        get_scenario_identifier(sc): i for i, sc in enumerate(scenarios)
    }

    # Handle --redo: purge stored results for the requested scenario indices
    if args.redo:
        print(f"Info: Redo mode activated for scenario indices: {sorted(args.redo)}")
        initial_count = len(results)
        results = [
            res for res in results
            if scenario_id_to_index.get(get_scenario_identifier(res)) not in args.redo
        ]
        completed_tasks = {
            (get_scenario_identifier(res), res["seed"]) for res in results
        }
        removed = initial_count - len(results)
        if removed:
            print(f"Info: Purged {removed} results to be re-run.")

    # Determine which tasks still need to be run
    tasks_to_run = []
    for (scenario_id, seed), task_idx in task_lookup.items():
        if (scenario_id, seed) not in completed_tasks:
            scenario_index = scenario_id_to_index[scenario_id]
            tasks_to_run.append((scenarios[scenario_index], seed, task_idx))

    if not tasks_to_run:
        print("All scenarios appear complete. Verifying and finalizing output.")
        sort_and_finalize_output(results, args.output_file, task_lookup)
        return

    client = ChatCompletionClient(api_key=args.api_key, base_url=args.api_base)
    progress_bar = tqdm(
        total=len(scenarios) * args.runs,
        initial=len(completed_tasks),
        desc="Running scenarios",
        unit="task",
    )
    burned_in_scenarios: set = set()

    for scenario, seed, task_index in tasks_to_run:
        scenario_id = get_scenario_identifier(scenario)

        # Warm up the KV cache for this scenario on its first encounter
        if args.burn_in and scenario_id not in burned_in_scenarios:
            try:
                burn_in_messages = copy.deepcopy(scenario["test_messages"])
                if processed_suffix and burn_in_messages:
                    burn_in_messages[-1]["content"] += processed_suffix
                client.create_chat_completion(
                    model=args.model,
                    messages=burn_in_messages,
                    max_tokens=1,
                )
                burned_in_scenarios.add(scenario_id)
            except Exception as exc:
                scenario_num = scenario_id_to_index.get(scenario_id, "N/A")
                print(f"Warning: Burn-in failed for scenario {scenario_num}: {exc}")

        messages_for_api = copy.deepcopy(scenario["test_messages"])
        if processed_suffix:
            messages_for_api[-1]["content"] += processed_suffix

        try:
            api_params = {
                "model": args.model,
                "messages": messages_for_api,
                "seed": seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            # top_k is a vLLM extension not present in the upstream OpenAI API
            if args.api_base != "https://api.openai.com/v1":
                api_params["top_k"] = args.top_k

            response = client.create_chat_completion(**api_params)
            new_entry = {
                **scenario,
                "response": response["choices"][0]["message"]["content"],
                "output_length": response.usage.completion_tokens,
                "seed": seed,
            }
        except Exception as exc:
            print(f"\n  -> Error during API call for seed {seed}: {exc}")
            new_entry = {
                **scenario,
                "response": f"API_CALL_FAILED: {exc}",
                "output_length": 0,
                "seed": seed,
            }

        results.append(new_entry)
        save_incremental_results(results, args.output_file)
        progress_bar.update(1)

    progress_bar.close()
    sort_and_finalize_output(results, args.output_file, task_lookup)


if __name__ == "__main__":
    main()
