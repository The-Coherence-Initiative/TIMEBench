"""Pre-comparison pipeline: judge + structural stats in one step.

Given a raw ``*_output.json`` file produced by ``run_timebench.py``, this
script runs LLM-as-judge scoring and structural analysis sequentially,
producing all artefacts required before ``compare_results.py`` can be used:

    *_output_judged.json
    *_output_counted.json
    *_output_scenario_scores.json   ← consumed by compare_results.py
    *_output_scenario_stats.json    ← consumed by compare_results.py
    *_results.json                  ← aggregated accuracy + structural CIs

Usage
-----
    python process_output.py -i Outputs/TIME-8B_output.json

    # Skip judging if a *_judged.json already exists
    python process_output.py -i Outputs/TIME-8B_output.json --skip-judge

    # Override the judge model
    python process_output.py -i Outputs/TIME-8B_output.json --judge-model gpt-4.1
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from timebench.utils.degenerate import find_endless_repetition
from timebench.utils.io import get_base_scenario_id, update_json_file


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """You are an impartial and strict AI evaluator for certain specific characteristics. Your task is to evaluate a generated response from an LLM based ONLY on a given objective. You are not given the actual messages context that led to that response to avoid bias and enable blind testing. This benchmark is specifically designed to capture meta reasoning capabilities and temporal intelligence of a target model and is not a test of knowledge or hallucination.


**Instructions:**
1.  **Assess Compliance**: Judge strictly on whether the `response` directly and effectively fulfills the provided `objective` that is related to temporal intelligence or meta reasoning, which can be unrelated to the actual content of the response.
2.  **Do NOT Fact-Check**: Do not use your own knowledge to check for factual accuracy. Your evaluation must be confined to the objective, which leads to binary results.
3.  **Be Objective**: If the response fully meets the objective, but has factual errors in other things, or clear hallucinations in the actual content, it is a still success and scored 1.
4.  **Be Strict**: If the response partially meets the objective, misunderstands it, or fails to follow constraints, it is a failure and scored 0. There is no partial credit.


**Output format:**
You can use the response as rough work space for reasoning if needed. Then you must output the following blocks for parsing.
You MUST reply in this exact format (no additional text, no markdown formatting inside these blocks):


<score>
0 or 1
</score>


<scoring_reason>
Your brief explanation here (under 20 words)
</scoring_reason>


**Input:**
- Response: "{response}"
- Objective: "{objective}"
"""

_MARKDOWN_PATTERNS = {
    "headings":        re.compile(r"^\s*#+\s", re.MULTILINE),
    "lists":           re.compile(r"^\s*([\*\-]\s|\d+\.\s)", re.MULTILINE),
    "code_blocks":     re.compile(r"```", re.MULTILINE),
    "blockquotes":     re.compile(r"^\s*>\s", re.MULTILINE),
    "tables":          re.compile(r"\|.*\|", re.MULTILINE),
    "bold_or_italics": re.compile(
        r"(\*\*.*?\*\*|\*[^*\n]+\*|__.*?__|_[^_\n]+_)", re.MULTILINE
    ),
}
_PROHIBITED_IN_THINK = {
    k: v for k, v in _MARKDOWN_PATTERNS.items() if k != "bold_or_italics"
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM-as-judge scoring and structural analysis on a TIMEBench "
            "output file, producing all artefacts needed for compare_results.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-file",
        required=True,
        type=str,
        help="Path to the *_output.json file from run_timebench.py.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help=(
            "Skip the judging step. Use when a *_judged.json already exists "
            "and only structural stats need to be (re)computed."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2-2025-12-11",
        help="Judge model name.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL for the judge.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for the judge.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HuggingFace tokenizer used to count reasoning tokens.",
    )
    parser.add_argument(
        "--bootstrap-replications",
        type=int,
        default=10000,
        help="Bootstrap replications for confidence intervals.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level (e.g. 0.95 for 95%% CI).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for bootstrap reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 — LLM-as-judge
# ---------------------------------------------------------------------------


def _parse_judge_response(response_text: str) -> tuple:
    """Return ``(score, scoring_reason)`` parsed from judge output."""
    try:
        score_match = re.search(
            r"<score>\s*(.*?)\s*</score>",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if not score_match:
            return None, None
        score_text = score_match.group(1).strip()
        score = int(score_text) if score_text in ("0", "1") else None

        reason_match = re.search(
            r"<scoring_reason>\s*(.*?)\s*</scoring_reason>",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        scoring_reason = reason_match.group(1).strip() if reason_match else None

        if score is not None and scoring_reason:
            return score, scoring_reason
    except Exception:
        return None, None
    return None, None


def _get_instance_id(result: dict) -> tuple:
    """Unique identifier for a single test instance (scenario + seed)."""
    return (json.dumps(result.get("test_messages", []), sort_keys=True), result.get("seed"))


def _load_existing_judged(output_file: str) -> tuple:
    """Load already-judged results and return ``(results, completed_ids)``."""
    if not os.path.exists(output_file):
        return [], set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        judged = [r for r in existing if r.get("score") in (0, 1)]
        completed = {_get_instance_id(r) for r in judged}
        if completed:
            print(f"  Found {len(judged)} already-judged results. Resuming.")
        return judged, completed
    except (json.JSONDecodeError, IOError) as exc:
        print(f"  Warning: could not read existing judged file: {exc}. Starting fresh.")
        return [], set()


def _judge_with_retries(
    client: OpenAI,
    judge_model: str,
    prompt: str,
    max_retries: int = 5,
    base_seed: int = 3407,
) -> tuple:
    """Call the judge with retries. Returns ``(score, reason, fingerprint)``."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="none",
                temperature=0.0,
                seed=base_seed + attempt,
            )
            text = resp.choices[0].message.content
            score, reason = _parse_judge_response(text)
            if score is not None:
                return score, reason, resp.system_fingerprint
        except Exception as exc:
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {exc}")
        time.sleep(1)
    return None, f"Failed after {max_retries} attempts", None


def _bootstrap_ci(values: np.ndarray, n_rep: int, level: float) -> tuple:
    """Return ``(lower, upper)`` bootstrap CI for the mean of *values*."""
    if len(values) < 2:
        m = float(np.mean(values))
        return m, m
    samples = np.random.choice(values, (n_rep, len(values)), replace=True)
    means = np.mean(samples, axis=1)
    lower = np.percentile(means, (1.0 - level) / 2.0 * 100)
    upper = np.percentile(means, (1.0 + level) / 2.0 * 100)
    return lower, upper


def run_judge(
    all_results: list,
    judged_file: str,
    client: OpenAI,
    judge_model: str,
    n_replications: int,
    confidence_level: float,
) -> tuple:
    """Score all results with the judge, then compute accuracy statistics.

    Returns ``(final_summary, scenario_mean_scores)``.
    """
    print("\n[Phase 1/2] LLM-as-judge scoring")
    judged_results, completed_ids = _load_existing_judged(judged_file)
    to_judge = [r for r in all_results if _get_instance_id(r) not in completed_ids]

    if not to_judge:
        print("  All results already judged.")
    else:
        print(f"  Judging {len(to_judge)} of {len(all_results)} results...")
        bar = tqdm(
            to_judge,
            initial=len(judged_results),
            total=len(all_results),
            desc="  Judging",
            unit="item",
        )
        for result in bar:
            objective = result.get("objective")
            response_text = result.get("response")
            missing = not objective or not response_text
            failed = response_text and response_text.startswith("API_CALL_FAILED")

            if missing or failed:
                result.update({
                    "score": -1,
                    "scoring_reason": "Skipped: missing data or prior API failure.",
                })
            else:
                prompt = _JUDGE_PROMPT_TEMPLATE.format(
                    objective=objective, response=response_text
                )
                score, reason, fingerprint = _judge_with_retries(client, judge_model, prompt)
                if score is not None:
                    result.update({"score": score, "scoring_reason": reason})
                    if fingerprint:
                        result["judge_system_fingerprint"] = fingerprint
                else:
                    result.update({"score": -1, "scoring_reason": reason})

            judged_results.append(result)
            with open(judged_file, "w", encoding="utf-8") as f:
                json.dump(judged_results, f, indent=4)

    # Compute accuracy statistics
    print("  Computing accuracy statistics...")
    scenario_scores: dict = defaultdict(lambda: defaultdict(list))
    for result in judged_results:
        if result.get("score") in (0, 1):
            sid = get_base_scenario_id(result)
            cat = result.get("category", "Uncategorized")
            scenario_scores[cat][sid].append(result["score"])

    scenario_mean_scores: dict = {}
    category_averages: dict = defaultdict(list)
    for cat, scenarios in scenario_scores.items():
        for sid, scores in scenarios.items():
            mean = float(np.mean(scores))
            category_averages[cat].append(mean)
            scenario_mean_scores[sid] = mean

    per_category: dict = {}
    cat_bootstrap_means: dict = {}
    cat_weights: dict = {}
    total_scenarios = sum(len(v) for v in category_averages.values())

    for cat, averages in sorted(category_averages.items()):
        arr = np.array(averages)
        n = len(arr)
        if n == 0:
            continue
        cat_weights[cat] = n / total_scenarios
        obs = float(np.mean(arr))
        lower, upper = _bootstrap_ci(arr, n_replications, confidence_level)
        samples = np.random.choice(arr, (n_replications, n), replace=True)
        cat_bootstrap_means[cat] = np.mean(samples, axis=1)
        per_category[cat] = {
            "observed_mean_accuracy_percent": round(obs * 100, 2),
            "confidence_interval_95_percent": (round(lower * 100, 2), round(upper * 100, 2)),
            "scenarios": n,
        }

    stratified = np.zeros(n_replications)
    for cat, means_dist in cat_bootstrap_means.items():
        stratified += means_dist * cat_weights[cat]
    overall_lower = np.percentile(stratified, (1.0 - confidence_level) / 2.0 * 100)
    overall_upper = np.percentile(stratified, (1.0 + confidence_level) / 2.0 * 100)
    obs_overall = sum(
        per_category[c]["observed_mean_accuracy_percent"] / 100 * cat_weights[c]
        for c in per_category
    )
    overall = {
        "observed_mean_accuracy_percent": round(obs_overall * 100, 2),
        "confidence_interval_95_percent": (
            round(overall_lower * 100, 2),
            round(overall_upper * 100, 2),
        ),
        "scenarios": total_scenarios,
    }

    final_summary = {
        "analysis_type": (
            "Accuracy based on mean score per scenario. "
            "Overall CI calculated with stratified bootstrap."
        ),
        "confidence_level": confidence_level,
        "bootstrap_replications": n_replications,
        "seed": int(np.random.get_state()[1][0]),
        "overall_accuracy": overall,
        "per_category_accuracy": per_category,
    }
    return final_summary, scenario_mean_scores


# ---------------------------------------------------------------------------
# Phase 2 — Structural stats
# ---------------------------------------------------------------------------


def _stat_summary(values) -> dict:
    """Return {min, max, mean, median} for a numeric sequence."""
    return {
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": round(float(np.mean(values)), 2),
        "median": round(float(np.median(values))),
    }


def _annotate_single_run(result: dict, tokenizer) -> None:
    """Annotate *result* in-place with a ``stats`` dict."""
    import copy

    empty = {
        "has_response": False,
        "output_length": 0,
        "non_reasoning_length": 0,
        "light_markdown_types_found": [],
        "heavy_markdown_types_found": [],
        "is_malformed": False,
        "has_infinite_repetitions": False,
        "has_reasoning_leakage": False,
        "formatting_leakage_types": [],
        "has_think_blocks": False,
        "number_of_think_blocks": 0,
        "total_reasoning_tokens": 0,
        "reasoning_token_percentage": 0.0,
        "think_block_positions": {"beginning": 0, "in_between": 0, "at_end": 0},
        "individual_think_block_tokens": [],
    }
    result["stats"] = copy.deepcopy(empty)
    stats = result["stats"]

    response_text = result.get("response", "")
    check_inf, response_text = find_endless_repetition(response_text)
    if check_inf:
        result["output_length"] = len(tokenizer.encode(response_text))

    if not response_text or response_text.startswith("API_CALL_FAILED"):
        return

    stats["has_response"] = True
    stats["output_length"] = result.get("output_length", 0)

    if check_inf:
        stats["has_infinite_repetitions"] = True

    for md_type, pattern in _MARKDOWN_PATTERNS.items():
        if pattern.search(response_text):
            if md_type == "bold_or_italics":
                stats["light_markdown_types_found"].append(md_type)
            else:
                stats["heavy_markdown_types_found"].append(md_type)

    text_outside_think = re.sub(
        r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE
    )
    if "the user" in text_outside_think.lower():
        stats["has_reasoning_leakage"] = True

    raw_blocks = re.findall(
        r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE
    )
    valid_blocks = [c.strip() for c in raw_blocks if c.strip()]

    for content in valid_blocks:
        for md_type, pattern in _PROHIBITED_IN_THINK.items():
            if pattern.search(content) and md_type not in stats["formatting_leakage_types"]:
                stats["formatting_leakage_types"].append(md_type)

    open_tags = len(re.findall(r"<think>", response_text, re.IGNORECASE))
    close_tags = len(re.findall(r"</think>", response_text, re.IGNORECASE))
    stats["is_malformed"] = open_tags != close_tags

    num_valid = len(valid_blocks)
    if num_valid > 0:
        stats["has_think_blocks"] = True
        stats["number_of_think_blocks"] = num_valid
        block_tokens = [len(tokenizer.encode(c)) for c in valid_blocks]
        stats["individual_think_block_tokens"] = block_tokens
        reasoning_tokens = sum(block_tokens)
        stats["total_reasoning_tokens"] = reasoning_tokens
        stats["non_reasoning_length"] = stats["output_length"] - reasoning_tokens
        if stats["output_length"] > 0:
            stats["reasoning_token_percentage"] = (
                reasoning_tokens / stats["output_length"] * 100
            )
        stripped = response_text.strip()
        is_beg = stripped.lower().startswith("<think>")
        is_end = stripped.lower().endswith("</think>")
        stats["think_block_positions"]["beginning"] = 1 if is_beg else 0
        stats["think_block_positions"]["at_end"] = (
            1 if is_end and (num_valid > 1 or not is_beg) else 0
        )
        stats["think_block_positions"]["in_between"] = (
            num_valid
            - stats["think_block_positions"]["beginning"]
            - stats["think_block_positions"]["at_end"]
        )
    else:
        stats["non_reasoning_length"] = stats["output_length"]


def run_stats(
    annotated_data: list,
    n_replications: int,
    confidence_level: float,
    seed: int,
) -> tuple:
    """Compute descriptive and bootstrapped structural statistics.

    Returns ``(final_summary, per_scenario_aggregated)``.
    """
    all_runs = [r["stats"] for r in annotated_data if r["stats"]["has_response"]]
    if not all_runs:
        return {}, {}

    total_runs = len(all_runs)

    total_lengths = [s["output_length"] for s in all_runs]
    nr_lengths = [s["non_reasoning_length"] for s in all_runs]
    think_counts = [s["number_of_think_blocks"] for s in all_runs]
    rt_counts = [s["total_reasoning_tokens"] for s in all_runs]
    block_tokens = [t for s in all_runs for t in s["individual_think_block_tokens"]]

    md_counts: dict = defaultdict(int)
    for s in all_runs:
        for t in s["heavy_markdown_types_found"] + s["light_markdown_types_found"]:
            md_counts[t] += 1

    total_think = sum(think_counts)
    at_beg = sum(s["think_block_positions"]["beginning"] for s in all_runs)
    in_bet = sum(s["think_block_positions"]["in_between"] for s in all_runs)
    at_end = sum(s["think_block_positions"]["at_end"] for s in all_runs)

    n_malformed = sum(1 for s in all_runs if s["is_malformed"])
    n_infinite = sum(1 for s in all_runs if s["has_infinite_repetitions"])
    n_rleak = sum(1 for s in all_runs if s["has_reasoning_leakage"])
    n_fleak = sum(1 for s in all_runs if s["formatting_leakage_types"])
    n_degen = sum(
        1 for s in all_runs
        if any([
            s["is_malformed"],
            s["has_infinite_repetitions"],
            s["has_reasoning_leakage"],
            bool(s["formatting_leakage_types"]),
        ])
    )
    fleak_by_type: dict = defaultdict(int)
    for s in all_runs:
        for t in s["formatting_leakage_types"]:
            fleak_by_type[t] += 1

    def _pct(n: int) -> float:
        return round(n / total_runs * 100, 2)

    descriptive_stats = {
        "output_length_stats": _stat_summary(total_lengths),
        "non_reasoning_output_length_stats": _stat_summary(nr_lengths),
        "markdown_usage_stats": {
            "percentage_of_runs_with_any_markdown": _pct(
                sum(
                    1 for s in all_runs
                    if s["light_markdown_types_found"] or s["heavy_markdown_types_found"]
                )
            ),
            "percentage_with_heavy_markdown": _pct(
                sum(1 for s in all_runs if s["heavy_markdown_types_found"])
            ),
            "percentage_with_light_markdown": _pct(
                sum(1 for s in all_runs if s["light_markdown_types_found"])
            ),
            "usage_by_type_percentage": {
                "heavy": {
                    t: round(c / total_runs * 100, 2)
                    for t, c in md_counts.items()
                    if t != "bold_or_italics"
                },
                "light": (
                    {"bold_or_italics": round(md_counts["bold_or_italics"] / total_runs * 100, 2)}
                    if "bold_or_italics" in md_counts
                    else {}
                ),
            },
        },
        "reasoning_stats": {
            "number_of_think_blocks_per_run": _stat_summary(think_counts),
            "reasoning_tokens_per_run": _stat_summary(rt_counts),
            "reasoning_tokens_per_think_block": (
                _stat_summary(block_tokens)
                if block_tokens
                else {"min": 0, "max": 0, "mean": 0, "median": 0}
            ),
        },
        "structural_stats": {
            "percentage_of_runs_with_think_blocks": _pct(
                sum(1 for s in all_runs if s["has_think_blocks"])
            ),
            "think_block_positioning_percentage": (
                {
                    "at_beginning": round(at_beg / total_think * 100, 2),
                    "in_between": round(in_bet / total_think * 100, 2),
                    "at_end": round(at_end / total_think * 100, 2),
                }
                if total_think > 0
                else {}
            ),
            "percentage_of_runs_with_degenerate_structure": _pct(n_degen),
            "degenerate_structure_breakdown": {
                "malformed_blocks_percent": _pct(n_malformed),
                "infinite_repetitions_percent": _pct(n_infinite),
                "reasoning_leakage_percent": _pct(n_rleak),
                "formatting_leakage_percent": _pct(n_fleak),
                "formatting_leakage_by_type": (
                    {t: round(c / n_fleak * 100, 2) for t, c in fleak_by_type.items()}
                    if n_fleak > 0
                    else {}
                ),
            },
        },
    }

    scenario_runs: dict = defaultdict(list)
    for result in annotated_data:
        if result["stats"]["has_response"]:
            scenario_runs[get_base_scenario_id(result)].append(result["stats"])

    per_scenario: dict = {}
    for sid, runs in scenario_runs.items():
        total_blocks = sum(s["number_of_think_blocks"] for s in runs)

        def _pos(key: str) -> float:
            return (
                sum(s["think_block_positions"][key] for s in runs) / total_blocks * 100
                if total_blocks > 0
                else 0.0
            )

        per_scenario[sid] = {
            "mean_output_length": float(np.mean([s["output_length"] for s in runs])),
            "mean_non_reasoning_length": float(np.mean([s["non_reasoning_length"] for s in runs])),
            "percentage_with_heavy_markdown": float(
                np.mean([bool(s["heavy_markdown_types_found"]) for s in runs]) * 100
            ),
            "percentage_with_light_markdown": float(
                np.mean([bool(s["light_markdown_types_found"]) for s in runs]) * 100
            ),
            "percentage_degenerate": float(
                np.mean([
                    s["is_malformed"]
                    or s["has_infinite_repetitions"]
                    or s["has_reasoning_leakage"]
                    or bool(s["formatting_leakage_types"])
                    for s in runs
                ]) * 100
            ),
            "percentage_malformed": float(np.mean([s["is_malformed"] for s in runs]) * 100),
            "percentage_infinite_repetitions": float(
                np.mean([s["has_infinite_repetitions"] for s in runs]) * 100
            ),
            "percentage_with_reasoning_leakage": float(
                np.mean([s["has_reasoning_leakage"] for s in runs]) * 100
            ),
            "percentage_with_formatting_leakage": float(
                np.mean([bool(s["formatting_leakage_types"]) for s in runs]) * 100
            ),
            "percentage_with_think_blocks": float(
                np.mean([s["has_think_blocks"] for s in runs]) * 100
            ),
            "mean_number_of_think_blocks": float(
                np.mean([s["number_of_think_blocks"] for s in runs])
            ),
            "mean_total_reasoning_tokens": float(
                np.mean([s["total_reasoning_tokens"] for s in runs])
            ),
            "mean_reasoning_token_percentage": float(
                np.mean([s["reasoning_token_percentage"] for s in runs])
            ),
            "mean_think_block_positioning_percentage": {
                "at_beginning": _pos("beginning"),
                "in_between": _pos("in_between"),
                "at_end": _pos("at_end"),
            },
        }

    bootstrapped: dict = {}
    stat_names = list(next(iter(per_scenario.values())).keys())
    for stat_name in stat_names:
        if stat_name == "mean_think_block_positioning_percentage":
            continue
        values = np.array([s[stat_name] for s in per_scenario.values()])
        obs = float(np.mean(values))
        lower, upper = _bootstrap_ci(values, n_replications, confidence_level)
        bootstrapped[stat_name] = {
            "observed_mean": round(obs, 2),
            "confidence_interval_95_percent": (round(lower, 2), round(upper, 2)),
        }

    final_summary = {
        "descriptive_stats": descriptive_stats,
        "bootstrapped_stats": {
            "analysis_type": (
                "Metrics aggregated per-scenario, with CIs on the mean of those aggregates."
            ),
            "replications": n_replications,
            "seed": seed,
            "stats": bootstrapped,
        },
    }
    return final_summary, per_scenario


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full pre-comparison pipeline."""
    args = get_args()
    np.random.seed(args.seed)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at '{args.input_file}'")
        return

    base = os.path.splitext(args.input_file)[0]
    judged_file = f"{base}_judged.json"
    counted_file = f"{base}_counted.json"
    scenario_scores_file = f"{base}_scenario_scores.json"
    scenario_stats_file = f"{base}_scenario_stats.json"
    results_file = f"{base.replace('_output', '')}_results.json"

    with open(args.input_file, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # --- Phase 1: Judge ---
    if args.skip_judge:
        print("[Phase 1/2] Skipping judge (--skip-judge set).")
        if not os.path.exists(judged_file):
            print(f"Error: --skip-judge requires '{judged_file}' to already exist.")
            return
    else:
        if not args.api_key:
            print("Error: OPENAI_API_KEY not set and --api-key not provided.")
            return
        client = OpenAI(api_key=args.api_key, base_url=args.api_base)
        judge_summary, scenario_mean_scores = run_judge(
            all_results, judged_file, client, args.judge_model,
            args.bootstrap_replications, args.confidence_level,
        )
        update_json_file(results_file, judge_summary, "llm_judge_summary")
        update_json_file(scenario_scores_file, scenario_mean_scores, "accuracy_scenario_scores")

        overall = judge_summary["overall_accuracy"]
        ci = overall["confidence_interval_95_percent"]
        print(
            f"\n  Overall accuracy: {overall['observed_mean_accuracy_percent']:.2f}% "
            f"[{ci[0]:.2f}%, {ci[1]:.2f}%] ({overall['scenarios']} scenarios)"
        )

    # --- Phase 2: Structural stats ---
    print("\n[Phase 2/2] Structural analysis")
    print(f"  Loading tokenizer '{args.tokenizer_model}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_model, trust_remote_code=True
        )
    except Exception as exc:
        print(f"  Error loading tokenizer: {exc}")
        return

    with open(judged_file, "r", encoding="utf-8") as f:
        judged_data = json.load(f)

    for result in tqdm(judged_data, desc="  Annotating", unit="run"):
        _annotate_single_run(result, tokenizer)

    with open(counted_file, "w", encoding="utf-8") as f:
        json.dump(judged_data, f, indent=4)
    print(f"  Saved annotated data to '{counted_file}'.")

    stats_summary, per_scenario = run_stats(
        judged_data, args.bootstrap_replications, args.confidence_level, args.seed
    )
    update_json_file(scenario_stats_file, per_scenario, "structural_scenario_stats")
    update_json_file(results_file, stats_summary, "advanced_structural_stats")

    print(f"\nDone. Artefacts written alongside '{args.input_file}':")
    for f in (judged_file, counted_file, scenario_scores_file, scenario_stats_file, results_file):
        print(f"  {f}")


if __name__ == "__main__":
    main()
