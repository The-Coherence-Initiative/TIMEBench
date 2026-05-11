# TIMEBench

Evaluation harness for **TIME** (*Temporally Intelligent Meta-reasoning Engine*), as described in:

> Susmit Das. **TIME: Temporally Intelligent Meta-reasoning Engine for Context-Triggered Explicit Reasoning.** *Findings of ACL 2026*. arXiv:[2601.05300](https://arxiv.org/abs/2601.05300)

This repository contains the complete pipeline to reproduce the paper's evaluation: running models on TIMEBench, scoring responses with an LLM judge, computing structural and behavioral statistics, performing pairwise significance tests, and generating publication-ready tables.

All intermediate artefacts are included in `Outputs/`, so any stage of the pipeline can be entered without a GPU or API access.

---

## What is TIMEBench?

TIMEBench is a 77-scenario diagnostic benchmark (7 categories × 11 scenarios each) for evaluating reasoning from temporal cues and latent contextual state in dialogue. Its central question is not whether a model can recall dated facts, but whether it can use temporal structure to infer underlying context: what has changed, which assumptions may no longer hold, and how the response should adapt accordingly.

TIMEBench is not a benchmark of temporal fact recall or event dating. It uses temporal structure — discontinuities, gaps, invalid timestamps, timezone shifts, and chronological anomalies — as a controlled probe for whether a model can infer underlying context, recognise when assumptions have become unstable, and adapt its response accordingly.

### Benchmark categories

The seven diagnostic categories each isolate a distinct temporal phenomenon. Four are **completely out-of-distribution** relative to the TIME training curriculum (marked ⊘); three reflect **curriculum-intended behaviour**, though all scenarios remain unseen during training (marked ✓).

| Category | Temporal pattern probed | What successful behaviour requires | Train dist. |
|---|---|---|---|
| **Chronological Retrospection** | Non-trivial temporal reconstruction across turns, including partial logs, delayed write-ups, and implicit event windows | Reconstruct a latent timeline from scattered conversational evidence rather than relying on surface order alone; infer exact or bounded temporal relations when the answer depends on sequencing | ⊘ OOD |
| **Invalid Time Detection** | Impossible calendar values such as non-existent dates | Detect that the timestamp itself is invalid and explicitly register the anomaly | ⊘ OOD |
| **Temporal Adaptivity** | Shifts in urgency or actionability caused by imminent deadlines, passed deadlines, or short remaining wait times | Adapt the response style to temporal pressure: be urgent when minutes matter, withhold unnecessary interventions when help is imminent, and switch to fuller explanation once urgency has passed | ✓ In-dist. |
| **Temporal Contextual Awareness** | Time cues that imply situational context, such as festivals, holidays, or late-night study settings | Infer latent context from time itself and use it to shape interpretation and response tone, rather than answering as if the query were temporally generic | ✓ In-dist. |
| **Temporal Flow Anomaly Detection** | Non-monotonic or implausible temporal structure, including backward timestamps and extreme jumps across years or centuries | Notice that conversational time no longer behaves normally and treat the anomaly as a trigger for explicit scrutiny or re-anchoring, even if the model then continues assisting | ⊘ OOD |
| **Time Gap Awareness** | Long but plausible silence between turns, often combined with topic drift or likely changes in the user's situation | Recognise that earlier assumptions may be stale, re-anchor to the new moment, and avoid treating the earlier context as if nothing has changed | ✓ In-dist. |
| **Timezone Sensitivity** | Offset changes across turns that imply changes in local context, location, or circadian state | Use timezone shifts as reasoning evidence, for example to infer approximate location, travel progress, or appropriate advice in the user's new local context | ⊘ OOD |

### Evaluation protocol

- **77 scenarios** total (7 categories × 11 scenarios each)
- **10 trials** per scenario with PCG64-derived seeds — **770 runs** total
- **Decoding**: temperature 0.6, top-p 0.95, top-k 20, min-p 0.0
- **Judge**: GPT-5.2 (2025-12-11 checkpoint) via the OpenAI API, blind to the original prompt, timestamps, and formatting — evaluates only the model response against its binary objective
- **Scoring**: binary objectives (0/1) → scenario mean (10 trials) → category mean (11 scenarios, scaled to %) → overall TimeBench score (mean over 7 categories)
- **Confidence intervals**: 95% CIs via stratified bootstrapping (10,000 resamples) by resampling scenario scores within each category before recomputing all aggregates

### Behavioral instrumentation

In addition to correctness, TIMEBench supports a structural audit of generation. For each response the harness records: whether `<think>` appears, where it appears within the response, the number of `<think>` blocks, reasoning and output token counts, light versus heavy markdown usage, format bleed (markdown inside `<think>` blocks), and degeneracy indicators including infinite repetition and reasoning leakage. This allows analysis of whether explicit reasoning becomes more selective and better aligned with contextual need, not just whether answers are correct.

---

## Hardware and software

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 9 7950X3D |
| RAM | 128 GB DDR5 |
| GPU | NVIDIA RTX Pro 6000 Blackwell (96 GB VRAM) |
| OS | Ubuntu 24.04.3 LTS (WSL2 on Windows 11 Build 26100) |
| CUDA | 13.0 (driver 582.08) |
| Inference | `vllm==0.13.0` |

---

## Installation

Python 3.12 or later is required.

```bash
pip install -r requirements.txt
```

Set your OpenAI API key before running `process_output.py`:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Repository layout

```
.
├── run_timebench.py        Step 1 — run a model on TIMEBench
├── process_output.py       Step 2 — judge + structural stats
├── compare_results.py      Step 3 — Wilcoxon Signed-Rank comparison
├── build_full_tables.py    Step 4 — publication-ready CSV + Markdown tables
│
├── process_all_outputs.sh  Batch wrapper: process every *_output.json in Outputs/
├── requirements.txt
├── tests/
│   └── scenarios.json      TIMEBench scenario definitions (77 scenarios)
├── timebench/              Internal package (shared utilities)
│   └── utils/
│       ├── degenerate.py   Repetition-loop detector
│       ├── io.py           Shared JSON I/O helpers
│       └── testclient.py   OpenAI-compatible HTTP client
└── Outputs/                Model outputs, judgments, stats, and aggregates
```

---

## Pipeline

### Step 1 — Run a model

```bash
python run_timebench.py --model TIME-8B
```

To run a **no-thinking** variant, set an explicit output path and append the `/no_think` suffix:

```bash
python run_timebench.py \
    --model Qwen3-8B \
    -o Outputs/Qwen3-8B-No-Thinking_output.json \
    --content-suffix /no_think
```

Each run produces one `*_output.json` in `Outputs/`. Runs are resumable: the script checkpoints after every response and skips already-completed tasks on restart.

**Key flags**

| Flag | Default | Description |
|---|---|---|
| `--model` | `TIME-32B` | Model name passed to the vLLM server |
| `--runs` | `10` | Repetitions per scenario (770 total at default) |
| `--seed` | `3407` | Master seed for PCG64-derived per-run seed generation |
| `--no-burn-in` | — | Disable KV-cache warm-up before each scenario |
| `--content-suffix` | — | String appended to every final user turn |
| `--api-base` | `$API_BASE` | vLLM server base URL |

### Step 2 — Judge and analyse (pre-comparison)

`process_output.py` runs both evaluation phases sequentially on a single output file, producing all artefacts required before comparison:

```bash
python process_output.py -i Outputs/TIME-8B_output.json
```

This writes five files alongside the input:

| Artefact | Contents |
|---|---|
| `*_output_judged.json` | Per-response binary scores and judge reasoning |
| `*_output_counted.json` | Per-response structural annotations |
| `*_output_scenario_scores.json` | Per-scenario mean accuracy ← consumed by Step 3 |
| `*_output_scenario_stats.json` | Per-scenario structural aggregates ← consumed by Step 3 |
| `*_results.json` | Accuracy + structural CIs (updated in place) |

> **Cost note.** GPT-5.2 judging is billed per token. Reuse existing `*_judged.json` files with `--skip-judge` to run only the structural analysis step.

To process all outputs at once:

```bash
./process_all_outputs.sh
```

### Step 3 — Compare two models (Optional)

Performs **Wilcoxon Signed-Rank (WSR)** tests at the scenario level on accuracy and structural metrics. Both `*_scenario_scores.json` (accuracy) and `*_scenario_stats.json` (structural) files are supported.

```bash
# Accuracy comparison
python compare_results.py \
    --file-a Outputs/Qwen3-8B_output_scenario_scores.json \
    --file-b Outputs/TIME-8B_output_scenario_scores.json

# Structural comparison
python compare_results.py \
    --file-a Outputs/Qwen3-8B_output_scenario_stats.json \
    --file-b Outputs/TIME-8B_output_scenario_stats.json
```

Both commands merge their results into a single file:

```
Outputs/Qwen3-8B_vs_TIME-8B_stat_results.json
```

### Step 4 — Build publication tables (Optional)

Combines per-model result files and the WSR comparison into a single formatted table (Appendix C style):

```bash
python build_full_tables.py \
    --wsr Outputs/Qwen3-8B_vs_TIME-8B_stat_results.json \
    Outputs/Qwen3-8B-Non-Reasoning_results.json \
    Outputs/Qwen3-8B_results.json \
    Outputs/TIME-8B_results.json
```

Outputs:

```
Outputs/Qwen3-8B_vs_TIME-8B_stat_results_full.csv
Outputs/Qwen3-8B_vs_TIME-8B_stat_results_full.md
```

---

## Reproducibility

All intermediate `.json` files are included in `Outputs/`, allowing entry at any stage without a GPU or API key. Model checkpoints are not included.

Results are numerically stable across the reference hardware. Minor variations may occur due to floating-point nondeterminism on different hardware configurations.

---

## Citation

```bibtex
@article{das2026time,
  title     = {{TIME}: Temporally Intelligent Meta-reasoning Engine for
               Context-Triggered Explicit Reasoning},
  author    = {Susmit Das},
  journal   = {arXiv preprint arXiv:2601.05300},
  year      = {2026}
}
```
