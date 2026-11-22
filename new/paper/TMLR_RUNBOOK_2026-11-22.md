# Run the TMLR routing study on Windows · 22 November 2026

This runbook covers the validation pilot and the complete study. The runner trains fresh checkpoints and produces paired comparisons for the learned router, two shuffle controls, dense LoRA, trained fixed policies, and random masks.

## 1. Install once

Open **PowerShell on native Windows**. Install Git for Windows, 64-bit Python 3.12 with the Windows Python launcher (`py`), and an NVIDIA driver compatible with CUDA 12.4. Allow persistent storage for checkpoints and the Hugging Face cache. The project previously recorded TinyLlama bf16 runs on an **RTX 4060 Laptop GPU with 8 GB VRAM**, including a 500-step run. Use the preflight below to measure the revised protocol on your laptop. Downloads require access to Hugging Face and the package indexes.

In an existing repository clone:

```powershell
git fetch origin
git switch tmlr-routing-pilot
git pull --ff-only origin tmlr-routing-pilot
```

For a fresh clone use:

```powershell
git clone --branch tmlr-routing-pilot https://github.com/N0AHZACH/Research.git
cd Research
```

From the repository root:

```powershell
.\new\scripts\gpu\setup_tmlr_windows.cmd
```

Continue after the setup prints **Windows setup and smoke test passed**. It uses the virtual environment’s Python directly, so you can run every command in ordinary PowerShell.

The offline smoke test exercises tiny randomly initialized Llama and Qwen2 models. It checks one training step each for a static and learned policy, checkpoint reload, padding, gates, next-token loss and the likelihood-task forward path. Passing it means the small numerical paths work in your environment. The real-model pilot is the next check.

The CPU smoke also checks that whole-sequence swaps preserve per-layer usage, that passages return in their original order after length-matched batching, and that paired statistics use the same batch assignments. Use the real-model pilot below to verify the study on your GPU.

## 2. Start small

Open PowerShell in the repository root for each session. Every command below selects the project’s Windows Python explicitly.

```powershell
nvidia-smi
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study freeze
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study preflight --family tiny
```

The preflight runs two optimizer steps each for dense LoRA and the learned router at the study's bf16 precision, batch size and context length. It then runs WikiText validation. It saves training and validation peak PyTorch GPU memory in `new/artifacts/tmlr-routing-controls/preflight-tiny/resources.json`. These are allocator peaks. Use `nvidia-smi` to see GPU memory already in use by Windows or other applications. Run one GPU job at a time, with the laptop plugged in. The preflight checkpoints stay in their own directory.

When TinyLlama passes on your laptop, run:

```powershell
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study pilot
```

The output directory is `new/artifacts/tmlr-routing-controls/`. Earlier runs have a separate output directory. Back up this directory and the model cache. `protocol.json` records the source commit, package versions, model and WikiText revisions, split row indices and text hashes. Freeze uses the revisions available at first execution. The pilot trains one learned router, a dense checkpoint and the fixed policies that bracket the learned depth, then evaluates only on WikiText validation. Inspect its summary:

```powershell
Get-Content .\new\artifacts\tmlr-routing-controls\tiny-42\pilot_summary.json
```

Each shuffle must preserve native per-layer activity. `eligible_fraction` is the fraction of valid tokens that have a donor under that shuffle. `changed_fraction` is the fraction of valid tokens that actually received a different route. Low coverage or little route variation limits what a weak effect can establish. Negative `native_minus_nll` favors native routing. Use the pilot to check the controls, the fixed-policy depths and actual resource use before starting the full sweep. Keep the output whatever the result.

## 3. Finish one seed at a time

Before the first Qwen run, check its bf16 training and WikiText validation footprint separately. Qwen has a larger vocabulary and model, so TinyLlama's peak does not establish that Qwen fits. If Qwen runs out of memory, keep the error and TinyLlama outputs. Run Qwen on a GPU with more available VRAM to use the same protocol. A batch-size change needs a revised protocol and fresh output directory.

```powershell
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study preflight --family qwen
```

Run these commands individually after the TinyLlama pilot and the respective preflight:

```powershell
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study run --family tiny --seed 42
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study run --family tiny --seed 43
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study run --family tiny --seed 44
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study run --family qwen --seed 42
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study run --family qwen --seed 43
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study run --family qwen --seed 44
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study replay
.\.venv-gpu\Scripts\python.exe -m new.experiments.tmlr_study report
```

Run one process per output directory. Each completed training job or evaluation arm is saved separately. On interruption rerun the same command. The current unfinished checkpoint or evaluation arm restarts. A complete family/seed job includes many full benchmark passes. For an OOM or a failed invariant, keep the traceback and diagnose the failing stage. Altering batch size changes the shuffle intervention. A modified protocol needs a new commit and output directory (`--out PATH` on each command).

## 4. What the study measures

- **Scope.** Small-data LoRA adaptation on TinyLlama‑1.1B‑Chat and Qwen2.5‑1.5B. First 300 training passages longer than 100 characters, 128-token packing, final 50 packed blocks held out where available. All variants get expanded rank-8 LoRA, 500 updates, batch four, and the same seed-specific training order. Evaluate all variants with adapters kept separate from the base weights to use the same adapter arithmetic. Dense is the full-depth LoRA reference. The routed controls share the teacher and distillation settings. Two target skip fractions, 0.30 and 0.50, are labels. Report measured skip over routable layers.
- **Calibration.** Rank single-layer deletions by NLL on the first 50 qualifying WikiText validation passages using the dense LoRA checkpoint. For each integer depth that brackets a learned operating point, score the top-ranked deletion set and up to 12 alternatives that each exchange one of the last three selected layers for one of the next four in the ranking. Train the lowest-NLL policy at each depth. This searches combinations of deletions within a bounded candidate set.
- **Routing controls.** Use the same length-matched batch schedule for all WikiText arms. One shuffle swaps route vectors across passages independently at each valid position. The second swaps whole trajectories between passages with identical valid positions, preserving the order of routes within each donor trajectory. In both cases, three predeclared draws permute the learned checkpoint's routes while preserving per-layer activity. Separately, train a stochastic policy at the learned validation skip probability and evaluate three fixed random subsets at the nearest integer depth.
- **Primary endpoint.** Mean next-token NLL weighted by scored tokens on the first 500 qualifying WikiText test passages truncated to 128 tokens. Compare native assignment against each shuffle and trained static endpoints. Show every seed. Report both static endpoints and identify interpolated static comparisons. If the test operating point falls outside the calibrated interval, report the budget mismatch.
- **Secondary endpoints.** Full zero-shot ARC-Challenge and HellaSwag normalized multiple-choice accuracy for dense, native, trained static and random policies. The shuffle intervention is evaluated on WikiText only. Likelihood evaluation uses batch one for downstream routing accounting. No decoding or deployed inference speed claim follows from these experiments.
- **Uncertainty.** `paired.csv` uses 1,000 paired resamples within a seed, by batch on WikiText and by item on the downstream tasks. `seeds.csv` reports variation across the three training seeds. Shuffles and random masks reuse checkpoints. WikiText passages can share an article across batches, so intervals can understate uncertainty about new articles. `static_budget_comparison.csv` reports a descriptive interval at the observed interpolation weight and flags intervals within ±ln(1.02) nats/token, a predeclared study margin corresponding to a 2% perplexity ratio.

## 5. Check the evidence before writing

```powershell
$out = Resolve-Path .\new\artifacts\tmlr-routing-controls -ErrorAction Stop
$rows = @(Get-Content -LiteralPath (Join-Path $out 'SHA256SUMS') -ErrorAction Stop)
if ($rows.Count -eq 0) { throw 'Empty checksum manifest' }
foreach ($line in $rows) {
    if ($line -notmatch '^([0-9a-f]{64})  (.+)$') { throw "Invalid checksum row: $line" }
    $expected = $Matches[1]
    $file = Join-Path $out $Matches[2]
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $file -ErrorAction Stop).Hash -ne $expected) { throw "Checksum mismatch: $file" }
}
"Verified $($rows.Count) artifact files."
```

| Output | Use |
| --- | --- |
| `protocol.json`, per-checkpoint `manifest.json` and `hashes.json` | Reconstruct data, source, trainable parameters, environment, model revisions and checkpoint identity. |
| Per-family/seed evaluation JSON | Per-passage token counts and NLL, task item hashes and scores, gate statistics, masks and throughput for the instrumented perplexity pass. |
| `main.csv`, `quality_depth.pdf` | Per-seed quality versus measured depth. Throughput includes evaluation and instrumentation overhead. |
| `paired.csv`, `seeds.csv` | Native-minus-comparator contrasts. Negative NLL differences and positive accuracy differences favor native routing. |
| `static_budget_comparison.csv` | Per-seed test budget coverage and native-minus-linearly-interpolated static NLL where bracketed. Interpolation estimates a point between measured policies. |
| `tiny-42/replay.json`, `SHA256SUMS` | Same-environment numerical replay and checksums for the evidence bundle. For independent verification, give a colleague the frozen checkout, protocol, environment and checkpoints and have them run replay in a clean checkout. |

Completion means the commands exited successfully, six family/seed jobs have results, the replay passed, the checksums verify, and you have inspected every seed and any budget mismatches. If outcomes vary or a strong baseline ties, write that result. If the intervention seldom changes routes, or budgets fail to overlap, qualify the claim or design a follow-up. Use these generated tables for the new study. Reconcile any legacy values you elect to cite with their original evaluations; the old draft is not a source for the new tables. Venue acceptance and whether reviewers request additional experiments remain open.
