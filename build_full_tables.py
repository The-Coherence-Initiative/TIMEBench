"""Build publication-ready statistics tables (CSV + Markdown) from model JSONs.

Column names are derived from the model JSON filenames.  The WSR p-value column
explicitly names the two models being compared (the last two JSONs on the
command line).  Output files are written to the same directory as the WSR file,
with the base name ``<wsr_file>_full.{csv,md}``.

Usage
-----
python build_full_tables.py --wsr <wsr_file.json> <model1.json> [<model2.json> ...]

Example
-------
python build_full_tables.py \\
    --wsr Outputs/Qwen3-4B_vs_TIME-4B_stat_results.json \\
    Outputs/Qwen3-4B-Non-Reasoning_results.json \\
    Outputs/Qwen3-4B_results.json \\
    Outputs/TIME-4B_results.json
"""

import argparse
import json
import pathlib
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Keys excluded from the output table
# ---------------------------------------------------------------------------

_REJECTED_KEYS = {
    "mean_non_reasoning_length",
    "mean_reasoning_token_percentage",
}

# Human-readable labels for stat keys
_STAT_LABELS = {
    "overall_accuracy":                    "Benchmark Score",
    "mean_output_length":                  "Mean Total Output Tokens per Run",
    "mean_total_reasoning_tokens":         "Mean Total Thinking Tokens per Run",
    "mean_number_of_think_blocks":         "Mean Number of Think Blocks per Run",
    "percentage_with_think_blocks":        "Percentage with Think Blocks",
    "percentage_with_heavy_markdown":      "Percentage with Heavy Markdown",
    "percentage_with_light_markdown":      "Percentage with Light Markdown",
    "percentage_degenerate":               "Percentage with Any Degeneracy",
    "percentage_infinite_repetitions":     "Percentage with Infinite Repetitions",
    "percentage_malformed":                "Percentage with Malformed Outputs",
    "percentage_with_reasoning_leakage":   "Percentage with Reasoning Leakage",
    "percentage_with_formatting_leakage":  "Percentage with Formatting Leakage",
}

# Row ordering: benchmark scores first, then reasoning, then remainder
_SCORES_ORDER = [
    "Benchmark Score",
    "Chronological Retrospection",
    "Invalid Time Detection",
    "Temporal Adaptivity",
    "Temporal Contextual Awareness",
    "Temporal Flow Anomaly Detection",
    "Time Gap Awareness",
    "Timezone Sensitivity",
]
_REASONING_ORDER = [
    "Mean Total Output Tokens per Run",
    "Mean Total Thinking Tokens per Run",
    "Mean Number of Think Blocks per Run",
    "Percentage with Think Blocks",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def format_stat(value: float, ci: tuple) -> str:
    """Return ``'mean (low–high)'`` with two decimal places."""
    return f"{value:.2f} ({ci[0]:.2f}–{ci[1]:.2f})"


def format_p_value(p: float, min_threshold: float = 1e-8) -> str:
    """Return a formatted p-value string, flooring at *min_threshold*."""
    if p < min_threshold:
        return f"<{min_threshold:.0e}"
    return f"{p:.1e}"


def pretty_model_label(path: pathlib.Path) -> str:
    """Derive a concise column header from a result JSON filename stem."""
    stem = path.stem
    stem = stem.replace("_results", "")
    stem = stem.replace("_", " ").replace("-", " ")
    return stem.strip()


def extract_stats(path: pathlib.Path) -> dict:
    """Read one ``*_results.json`` and return ``{raw_stat_key: formatted_value}``."""
    with path.open() as f:
        data = json.load(f)

    out: dict = {}

    # Overall benchmark score
    acc = data["llm_judge_summary"]["overall_accuracy"]
    out["overall_accuracy"] = format_stat(
        acc["observed_mean_accuracy_percent"],
        acc["confidence_interval_95_percent"],
    )

    # Per-category scores
    for cat, v in data["llm_judge_summary"]["per_category_accuracy"].items():
        out[cat] = format_stat(
            v["observed_mean_accuracy_percent"],
            v["confidence_interval_95_percent"],
        )

    # Bootstrapped structural stats
    for k, v in data["advanced_structural_stats"]["bootstrapped_stats"]["stats"].items():
        out[k] = format_stat(v["observed_mean"], v["confidence_interval_95_percent"])

    # Drop keys not shown in the paper tables
    return {k: v for k, v in out.items() if k not in _REJECTED_KEYS}


def nice_label(raw_name: str) -> str:
    """Return a human-readable label for a stat key."""
    return _STAT_LABELS.get(raw_name, raw_name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate full stats CSV + Markdown table."
    )
    parser.add_argument(
        "--wsr",
        required=True,
        help="JSON file with Wilcoxon Signed-Rank p-values.",
    )
    parser.add_argument(
        "model_jsons",
        nargs="+",
        help="One or more model-result JSON files.",
    )
    args = parser.parse_args(argv)

    wsr_path = pathlib.Path(args.wsr).expanduser().resolve()
    model_paths = [pathlib.Path(p).expanduser().resolve() for p in args.model_jsons]

    # Build a DataFrame with one column per model
    model_stats = {pretty_model_label(mp): extract_stats(mp) for mp in model_paths}
    df = pd.DataFrame(model_stats)
    df.index.name = "Statistic"

    # Attach WSR p-values
    with wsr_path.open() as f:
        wsr_data = json.load(f)

    # The comparison JSON may store benchmark score under a legacy key
    if "benchmark_score" in wsr_data:
        wsr_data["overall_accuracy"] = wsr_data.pop("benchmark_score")

    p_vals = {
        k: (format_p_value(wsr_data[k]["p_value"]) if k in wsr_data else "—")
        for k in df.index
    }
    df["WSR p-value"] = pd.Series(p_vals)

    # Rename stat keys to human-readable labels
    df = df.rename(index=nice_label)

    # Enforce canonical row order: scores → reasoning → remainder
    already_placed = _SCORES_ORDER + _REASONING_ORDER
    remaining = [x for x in df.index if x not in already_placed]
    df = df.loc[already_placed + remaining]

    # Write outputs alongside the WSR file
    base = wsr_path.with_suffix("")
    csv_path = base.with_name(base.name + "_full.csv")
    md_path = base.with_name(base.name + "_full.md")

    df.to_csv(csv_path)

    header = "| " + " | ".join(["Statistic", *df.columns]) + " |"
    separator = "| " + " | ".join("---" for _ in range(len(df.columns) + 1)) + " |"
    rows = [
        "| " + " | ".join([stat] + [str(v) for v in row]) + " |"
        for stat, row in df.iterrows()
    ]
    md_path.write_text("\n".join([header, separator, *rows]), encoding="utf-8")

    print("Saved:")
    print("  •", csv_path)
    print("  •", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
