# Run the TMLR routing study · 22 November 2026

**Use this file for the complete study.** The older undated runbook describes the exploratory pilot. This version trains fresh checkpoints, evaluates the learned, shuffled, dense, fixed and stochastic policies, and generates the analysis files. An old checkpoint is optional for the older pilot and unnecessary here.

## 1. Install once

Use Linux or WSL2 with a compatible NVIDIA driver, a bfloat16-capable GPU, Python 3.12, and persistent storage for checkpoints and the Hugging Face cache. A 24 GB GPU is a reasonable starting choice. Total VRAM, elapsed time and disk usage for the full campaign have not been measured. Check free space and measure the first run before booking a long rental. Downloads require access to Hugging Face and the package indexes.

In an existing repository clone:

```bash
git fetch origin
git switch tmlr-routing-pilot
git pull --ff-only origin tmlr-routing-pilot
```

For a fresh clone use:

```bash
git clone --branch tmlr-routing-pilot https://github.com/N0AHZACH/Research.git
cd Research
```

From the repository root:

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r new/scripts/gpu/requirements-gpu.txt matplotlib
python -m pip check
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(), torch.ones(4, device="cuda").sum().item())'
python -m new.tests.tmlr_smoke
```

The offline smoke test exercises tiny randomly initialized Llama and Qwen2 models. It checks one training step each for a static and learned policy, checkpoint reload, padding, gates, next-token loss and the likelihood-task forward path. Passing it means the small numerical paths work in your environment. The real-model pilot is the next check.

**Verification at handoff.** The CPU smoke passed using torch 2.6.0 and the pinned Transformers, PEFT, datasets and lm-eval versions. A synthetic six-job result set generated all four CSVs and the PDF. The real pretrained checkpoints, full benchmarks, CUDA performance and scientific conclusions need the runs below.

## 2. Start small

Activate the environment again in each new shell. Use the repository root for all commands.

```bash
source .venv-gpu/bin/activate
python -m new.experiments.tmlr_study freeze
python -m new.experiments.tmlr_study pilot
```

The output directory is `new/artifacts/tmlr-2026-11-22/`. Back it up together with your model cache. `protocol.json` records the source commit, package versions, model and WikiText revisions, split row indices and text hashes. Freeze uses the revisions available at first execution. Share this exact file for replication. Inspect the pilot:

```bash
python - <<'PY'
import json, pathlib
folder = pathlib.Path("new/artifacts/tmlr-2026-11-22/tiny-42")
for path in sorted(folder.glob("pilot_*.json")):
    row = json.loads(path.read_text())
    print(path.stem, "PPL", row["ppl"]["perplexity"], "skip", row["routing"]["achieved_skip"],
          "routes changed", row.get("changed_fraction", 0))
PY
```

Each shuffle preserves native per-layer activity. If few routes change, the intervention has little power to detect an assignment effect. Inspect this before using paid compute. A weak or reversed effect is a valid result. Keep the output.

## 3. Finish at your pace

**One command** runs or resumes the entire sequence, including the pilot, all six family/seed jobs, checkpoint replay and analysis:

```bash
python -m new.experiments.tmlr_study all
```

**For separate sessions**, run these commands individually after the pilot:

```bash
python -m new.experiments.tmlr_study run --family tiny --seed 42
python -m new.experiments.tmlr_study run --family tiny --seed 43
python -m new.experiments.tmlr_study run --family tiny --seed 44
python -m new.experiments.tmlr_study run --family qwen --seed 42
python -m new.experiments.tmlr_study run --family qwen --seed 43
python -m new.experiments.tmlr_study run --family qwen --seed 44
python -m new.experiments.tmlr_study replay
python -m new.experiments.tmlr_study report
```

Run one process per output directory. Each completed training job or evaluation arm is saved separately. On interruption rerun the same command. The current unfinished checkpoint or evaluation arm restarts. A complete family/seed job includes many full benchmark passes. For an OOM or a failed invariant, keep the traceback and diagnose the failing stage. Altering batch size changes the shuffle intervention. A modified protocol needs a new commit and output directory (`--out PATH` on each command).

## 4. What the study measures

- **Scope.** Small-data LoRA adaptation on TinyLlama‑1.1B‑Chat and Qwen2.5‑1.5B. First 300 training passages longer than 100 characters, 128-token packing, final 50 packed blocks held out where available. All variants get expanded rank-8 LoRA, 500 updates, batch four, and the same seed-specific training order. Dense is the full-depth LoRA reference. The routed controls share the teacher and distillation settings. Two target skip fractions, 0.30 and 0.50, are labels. Report measured skip over routable layers.
- **Calibration.** On the first 50 qualifying WikiText validation passages, bypass each candidate layer of the dense adapted model in turn. Rank by lowest single-deletion NLL. For each learned operating point, train fixed policies dropping the floor and ceiling of its measured mean dropped-layer count. This is a reproducible static heuristic. Joint optimization over all subsets could yield a stronger policy.
- **Random controls.** Train a stochastic policy at the learned validation skip probability. Evaluate three fixed random subsets with the nearest integer count. The token-assignment intervention separately shuffles full route vectors across valid examples at the same position within batches of four, with three predeclared draws on the learned checkpoint.
- **Primary endpoint.** Mean next-token NLL weighted by scored tokens on the first 500 qualifying WikiText test passages truncated to 128 tokens. Compare native assignment against each shuffle and trained static endpoints. Show every seed. Report both static endpoints and identify interpolated static comparisons. If the test operating point falls outside the calibrated interval, report the budget mismatch.
- **Secondary endpoints.** Full zero-shot ARC-Challenge and HellaSwag normalized multiple-choice accuracy for dense, native, trained static and random policies. The shuffle intervention is evaluated on WikiText only. Likelihood evaluation uses batch one for downstream routing accounting. No decoding or deployed inference speed claim follows from these experiments.
- **Uncertainty.** `paired.csv` uses 1,000 paired passage or item resamples within a seed. `seeds.csv` reports variation across the three training seeds. Shuffles and random masks reuse checkpoints. Treat them as repeated interventions, never as additional training seeds. WikiText passages can share an article, so passage intervals can understate uncertainty about new articles.

## 5. Check the evidence before writing

```bash
(cd new/artifacts/tmlr-2026-11-22 && sha256sum --check --quiet SHA256SUMS)
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
