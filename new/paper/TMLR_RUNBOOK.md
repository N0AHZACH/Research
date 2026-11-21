# TMLR experiment runbook

Run one block at a time in Bash from the repository root on a Linux NVIDIA GPU machine. Use a checkout containing the pilot and evaluator changes. Keep completed outputs between sessions. Steps 1–4 are exploratory. Step 5 describes the work needed before a submission decision.

## 1. Set up once

Use a compatible NVIDIA driver and Python 3.11 or 3.12. In a fresh checkout, install the dependencies directly. The older GPU setup script ends by checking for unrelated local checkpoints.

```bash
python3.12 -m venv .venv-gpu   # substitute python3.11 if needed
source .venv-gpu/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r new/scripts/gpu/requirements-gpu.txt
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name())'
```

At the start of each later session, restore the shell settings below. Set CKPT to your recovered 500-step TinyLlama token_dlr_50 seed-42 checkpoint if it lives elsewhere.

```bash
source .venv-gpu/bin/activate
CKPT=new/artifacts/runs/tinyllama1b_token_dlr_50_sw1600_500_seed42
OUT=new/artifacts/tmlr
mkdir -p "$OUT"
```

## 2. Recover or train one checkpoint

Inspect the checkpoint. It needs the adapter config and weights, tokenizer, and router weights. Check the training manifest for model, seed, skip weight and step count.

```bash
ls -lh "$CKPT"
cat "$CKPT/manifest.json"
```

If it is missing, train one replacement. This command uses a fresh destination and can be run independently of the other seeds. If training is interrupted, choose a fresh output directory for the retry. The trainer saves its checkpoint at completion.

```bash
CKPT="$OUT/tinyllama_dlr50_s42"
test ! -e "$CKPT" && python new/experiments/train.py --variant token_dlr_50 --seed 42 --max-steps 500 --skip-weight 1600 --expanded-lora --output-dir "$CKPT"
```

If you trained a replacement, set CKPT to that path again when resuming.

## 3. Run the decisive pilot

```bash
python new/experiments/tmlr_pilot.py "$CKPT" "$OUT/tinyllama_s42_shuffle.json"
python -m json.tool "$OUT/tinyllama_s42_shuffle.json"
```

The pilot first checks forced-open logits against the same adapted model and checks real-token gates and loss under padding and batch duplication. An assertion failure warrants investigation before continuing. It scores native routing and three seeded permutations on the same 50 WikiText validation passages at length 128. Entire route vectors move between real tokens at the same position within each batch. A position with only one eligible example keeps its route. The pilot checks per-layer activity equality and refuses to overwrite a finished report. Keep batch size fixed across comparisons because it determines the shuffle pool.

Compare the native perplexity with each shuffle. A consistent increase under shuffling supports dependence on token assignment for this checkpoint. Three shuffles share one training seed. A small or inconsistent gap makes the static-policy explanation particularly important. Treat either outcome as exploratory.

## 4. Train comparable TinyLlama controls, one job per sitting

```bash
test ! -e "$OUT/dense_s42" && python new/experiments/train.py --variant dense --expanded-lora --max-steps 500 --seed 42 --output-dir "$OUT/dense_s42"
```

```bash
test ! -e "$OUT/stoch30_s42" && python new/experiments/train.py --variant stochastic_30 --expanded-lora --max-steps 500 --seed 42 --skip-weight 1600 --output-dir "$OUT/stoch30_s42"
```

Create a three-row evaluation manifest, then validate and run it. The evaluator resolves checkpoint paths from the repository root. Use a fresh evaluation ID for a repeat.

```bash
python - "$CKPT" "$OUT" <<'PY'
import json, pathlib, sys
checkpoint, out = map(pathlib.Path, sys.argv[1:])
m = json.loads(pathlib.Path("new/configs/eval/tinyllama1b-010-g4-500.json").read_text())
paths = {"dense": out/"dense_s42", "stochastic_30": out/"stoch30_s42", "token_dlr_50": checkpoint}
m["variants"] = [v for v in m["variants"] if v["name"] in paths]
for v in m["variants"]:
    v["checkpoint"] = str(paths[v["name"]])
    if v["type"] == "token_dlr":
        v["router_weights"] = str(checkpoint/"router_weights.pt")
m.update(eval_run_id="tinyllama_equal_lora_s42_v1", batch_size=4)
with (out/"controls.json").open("x") as f:
    json.dump(m, f, indent=2)
PY
python new/experiments/evaluate.py --manifest "$OUT/controls.json" --dry-run
test ! -e "$OUT/evals/tinyllama_equal_lora_s42_v1" && python new/experiments/evaluate.py --manifest "$OUT/controls.json" --output-dir "$PWD/$OUT/evals"
```

Read perplexity.json, routing_metrics.csv and task_results.json together. Record achieved skip for each arm. The control uses one fixed random mask and downstream tasks use at most 100 items. Use this block to diagnose the comparison before launching a final experiment.

## 5. Publication checklist and decision

- [ ] Implement and train a strong static baseline. Rank layers on calibration data by held-out loss increase when bypassed, select the fixed subset for each budget, and adapt with the same LoRA targets, teacher, training tokens and update count. If integer depth prevents an exact match, report the two neighboring depths and the interpolation rule. Report mask sensitivity over multiple random masks separately.
- [ ] Freeze the primary endpoint, achieved budgets, split IDs, seed list, data volume, checkpoint selection rule and task settings before final evaluation. Use a broader documented training sample or justify the small-data adaptation setting. Reserve untouched test data for final claims.
- [ ] Repeat across TinyLlama and Qwen2.5-1.5B, two achieved budgets and at least three training seeds per configuration. Adapt the pilot manifest and anchor count for Qwen. The supplied pilot is TinyLlama-specific.
- [ ] Save per-example scores and valid-token counts. Report paired differences with confidence intervals across held-out documents or task items and show between-training-seed variability separately. Evaluate the full selected likelihood benchmarks. Log achieved depth on each workload and distinguish effective depth from measured speed.
- [ ] Generate the tables from frozen outputs. Archive exact source, model/data revisions, environment, configs and checkpoint hashes. Have another person reproduce one headline row from a clean checkout.
- [ ] Rewrite the draft around the observed result. Reconcile the conflicting legacy ARC scores, explain how the study extends GateSkip, and state the dense-quality and throughput costs. Check the TMLR author guide and acceptance criteria before submitting.

The static trainer, cross-family shuffle audit, per-item inference and final table generator require further implementation and validation. Existing GPU sweep scripts do not complete this checklist. Scale after the corrected pilot and static baseline identify an effect worth estimating. A robust finding that a fixed policy recovers the gain can support a useful paper too.
