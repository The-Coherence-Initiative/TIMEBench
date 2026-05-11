"""Wilcoxon Signed-Rank comparison between two TIMEBench model outputs.

Accepts either ``*_output_scenario_scores.json`` (accuracy) or
``*_output_scenario_stats.json`` (structural stats) files and runs
Wilcoxon Signed-Rank tests for a fixed set of metrics that are
meaningful to compare at the scenario level.
"""

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


# Metrics tested by the Wilcoxon procedure
_ALLOWED_METRICS = {
    "mean_output_length",
    "percentage_degenerate",
    "mean_number_of_think_blocks",
    "mean_total_reasoning_tokens",
    "benchmark_score",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Perform Wilcoxon Signed-Rank tests to compare two models at the "
            "scenario level."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file-a",
        required=True,
        type=str,
        help="Path to the first JSON file (Model A).",
    )
    parser.add_argument(
        "--file-b",
        required=True,
        type=str,
        help="Path to the second JSON file (Model B).",
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default=None,
        help=(
            "Path to save the JSON output. "
            "Defaults to '<model_a>_vs_<model_b>_stat_results.json' "
            "in file-a's directory."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance level for the test.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_metrics(file_path: str) -> pd.DataFrame:
    """Load per-scenario metrics from *file_path* into a :class:`~pandas.DataFrame`."""
    with open(file_path, "r") as f:
        data = json.load(f)

    if "accuracy_scenario_scores" in data:
        df = pd.DataFrame.from_dict(data["accuracy_scenario_scores"], orient="index")
        df.columns = ["benchmark_score"]
        return df

    if "structural_scenario_stats" in data:
        return pd.DataFrame.from_dict(data["structural_scenario_stats"], orient="index")

    raise ValueError(
        "Unknown JSON structure. "
        "Expected 'accuracy_scenario_scores' or 'structural_scenario_stats'."
    )


def sanitize_name(name: str) -> str:
    """Strip pipeline-generated suffixes from a filename stem."""
    for suffix in ("_output_scenario_stats", "_output_scenario_scores"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def run_wilcoxon_tests(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    alpha: float,
) -> dict:
    """Run Wilcoxon tests on the intersection of scenarios for each allowed metric.

    Returns a dict mapping metric name → result dict.
    """
    common_indices = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_indices]
    df_b = df_b.loc[common_indices]

    results: dict = {}
    columns_to_test = _ALLOWED_METRICS.intersection(df_a.columns)

    for col in columns_to_test:
        if col not in df_b.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_a[col]):
            continue
        if not pd.api.types.is_numeric_dtype(df_b[col]):
            continue

        differences = df_a[col] - df_b[col]
        try:
            if any(differences == 0):
                stat, p = wilcoxon(df_a[col], df_b[col], mode="approx")
                mode_used = "approx"
            else:
                stat, p = wilcoxon(df_a[col], df_b[col], mode="exact")
                mode_used = "exact"

            results[col] = {
                "wilcoxon_statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < alpha),
                "mode_used": mode_used,
            }
        except (ValueError, Exception) as exc:
            results[col] = {"error": str(exc), "skipped": True}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the Wilcoxon comparison pipeline."""
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
    args = get_args()

    file_a_name = sanitize_name(Path(args.file_a).stem)
    file_b_name = sanitize_name(Path(args.file_b).stem)

    df_a = load_metrics(args.file_a)
    df_b = load_metrics(args.file_b)

    new_results = run_wilcoxon_tests(df_a, df_b, args.alpha)

    if args.output_file is None:
        output_dir = Path(args.file_a).parent
        output_filename = f"{file_a_name}_vs_{file_b_name}_stat_results.json"
        output_path = output_dir / output_filename
    else:
        output_path = Path(args.output_file)

    # Merge with any existing results in the output file
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            print(
                f"Warning: Existing file '{output_path}' is not valid JSON. "
                "It will be overwritten."
            )
            existing_results = {}
    else:
        existing_results = {}

    existing_results.update(new_results)

    with open(output_path, "w") as f:
        json.dump(existing_results, f, indent=4)

    print(f"Wilcoxon test results written to: {output_path}")


if __name__ == "__main__":
    main()
