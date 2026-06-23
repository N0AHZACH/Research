import os
import sys
import subprocess
import time
import argparse

EXPERIMENTS = {
    "qwen7b": {
        "train": [
            # ("exp23_qwen7b_baseline.py", "Baseline LoRA Fine-Tuning"), # Skipped (already generated & saved)
            ("exp24_qwen7b_stochastic.py", "Stochastic Depth Dropout Control"),
            ("exp25_qwen7b_token_routing.py", "DLR Token-Level Router Training"),
            ("exp30_qwen7b_pareto_sweep.py", "Pareto Sweep Strategy"),
        ],
        "eval": [
            ("exp22_qwen7b_eval_harness.py", "Evaluation Suite (MMLU/GSM8K/ARC/PPL)"),
        ]
    },
    "llama8b": {
        "train": [
            ("exp26_llama8b_baseline.py", "Baseline LoRA Fine-Tuning"),
            ("exp27_llama8b_stochastic.py", "Stochastic Depth Dropout Control"),
            ("exp28_llama8b_token_routing.py", "DLR Token-Level Router Training"),
            ("exp31_llama8b_pareto_sweep.py", "Pareto Sweep Strategy"),
        ],
        "eval": [
            ("exp29_llama8b_eval_harness.py", "Evaluation Suite (MMLU/GSM8K/ARC/PPL)"),
        ]
    }
}

def get_gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        # Fallback to nvidia-smi if torch is not yet configured or available
        try:
            res = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                lines = [line for line in res.stdout.strip().split('\n') if line]
                return len(lines)
        except Exception:
            pass
        return 1

def run_script_sequential(script_name, description):
    print("=" * 80)
    print(f"  RUNNING: {script_name} - {description}")
    print("=" * 80)
    
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"{script_name.replace('.py', '')}.log")
    print(f"[INFO] Outputs are being piped to: {log_file}")
    
    start_time = time.time()
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                
            process.wait()
            
        elapsed = time.time() - start_time
        if process.returncode == 0:
            print(f"\n[SUCCESS] {script_name} completed in {elapsed:.2f}s.\n")
            return True
        else:
            print(f"\n[FAILURE] {script_name} failed with exit code {process.returncode} (time: {elapsed:.2f}s).\n")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Exception occurred while running {script_name}: {e}\n")
        return False

def run_phase_parallel(tasks, num_gpus):
    """Run a list of script tasks in parallel across the available GPUs."""
    print("=" * 80)
    print(f"  STARTING PARALLEL PHASE: {len(tasks)} tasks across {num_gpus} GPUs")
    print("=" * 80)
    
    os.makedirs("logs", exist_ok=True)
    
    # Queue of tasks: (script_name, description)
    task_queue = list(tasks)
    
    # Active jobs: gpu_id -> { "process": Popen, "script": name, "file": file_handle, "start_time": t }
    active_jobs = {}
    completed_jobs = []
    failed_jobs = []
    
    while task_queue or active_jobs:
        # Check active jobs for completion
        finished_gpus = []
        for gpu_id, job in list(active_jobs.items()):
            proc = job["process"]
            ret = proc.poll()
            if ret is not None:
                # Job has finished
                elapsed = time.time() - job["start_time"]
                job["file"].close()
                
                if ret == 0:
                    print(f"\n[SUCCESS] [GPU {gpu_id}] {job['script']} completed in {elapsed:.2f}s.\n")
                    completed_jobs.append(job['script'])
                else:
                    print(f"\n[FAILURE] [GPU {gpu_id}] {job['script']} failed with exit code {ret} in {elapsed:.2f}s. Check log: logs/{job['script'].replace('.py', '')}.log\n")
                    failed_jobs.append((job['script'], ret))
                
                finished_gpus.append(gpu_id)
        
        # Remove finished jobs from tracking
        for gpu_id in finished_gpus:
            del active_jobs[gpu_id]
            
        # If any job failed, we abort to prevent cascading errors
        if failed_jobs:
            print(f"[HALT] Parallel execution halted due to script failures: {failed_jobs}")
            for gpu_id, job in active_jobs.items():
                print(f"[INFO] Terminating active job on GPU {gpu_id}: {job['script']}")
                job["process"].terminate()
            sys.exit(1)
            
        # Assign tasks to idle GPUs
        for gpu_id in range(num_gpus):
            if gpu_id not in active_jobs and task_queue:
                script_name, desc = task_queue.pop(0)
                log_file = os.path.join("logs", f"{script_name.replace('.py', '')}.log")
                print(f"[LAUNCH] [GPU {gpu_id}] {script_name} - {desc} (piping to: {log_file})")
                
                f = open(log_file, "w", encoding="utf-8")
                
                # Copy current environment and set CUDA_VISIBLE_DEVICES
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                
                process = subprocess.Popen(
                    [sys.executable, script_name],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True
                )
                
                active_jobs[gpu_id] = {
                    "process": process,
                    "script": script_name,
                    "file": f,
                    "start_time": time.time()
                }
                
        # Wait a bit before checking again
        if task_queue or active_jobs:
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="DLR Phase 4 Experiment Suite Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qwen", action="store_true", help="Run all Qwen2.5-7B experiments")
    group.add_argument("--llama", action="store_true", help="Run all Llama3.1-8B experiments")
    group.add_argument("--all", action="store_true", help="Run all Qwen7B and Llama8B experiments")
    group.add_argument("--list", action="store_true", help="List all experimental scripts")
    
    parser.add_argument("--no_parallel", action="store_true", help="Force sequential execution even if multiple GPUs are available")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Phase 4 scaling experiments:")
        for model, phases in EXPERIMENTS.items():
            print(f"\n{model.upper()} Suite:")
            for s, desc in phases["train"] + phases["eval"]:
                status = "Exists" if os.path.exists(s) else "Missing!"
                print(f"  - {s:<32} | {desc:<45} | [{status}]")
        print()
        return

    selected_suites = []
    if args.qwen or args.all:
        selected_suites.append("qwen7b")
    if args.llama or args.all:
        selected_suites.append("llama8b")
        
    num_gpus = get_gpu_count()
    use_parallel = num_gpus > 1 and not args.no_parallel
    
    if use_parallel:
        print(f"\n[INFO] Detected {num_gpus} GPUs. Activating multi-GPU parallel execution scheduler!\n")
        # Gather all training scripts for selected suites
        train_tasks = []
        eval_tasks = []
        for suite in selected_suites:
            train_tasks.extend(EXPERIMENTS[suite]["train"])
            eval_tasks.extend(EXPERIMENTS[suite]["eval"])
            
        # Run all training tasks in parallel
        run_phase_parallel(train_tasks, num_gpus)
        
        # Run all evaluation tasks in parallel
        run_phase_parallel(eval_tasks, num_gpus)
    else:
        # Sequential mode
        print(f"\n[INFO] Running in single-GPU / sequential mode.\n")
        for suite in selected_suites:
            print(f"\n=== Starting Experimental Suite: {suite.upper()} ===")
            for script, desc in EXPERIMENTS[suite]["train"] + EXPERIMENTS[suite]["eval"]:
                if not os.path.exists(script):
                    print(f"[ERROR] Script '{script}' does not exist. Skipping.")
                    continue
                    
                success = run_script_sequential(script, desc)
                if not success:
                    print(f"[HALT] Stopping execution due to failure in {script}.")
                    sys.exit(1)
                    
    print("\nAll selected experiment suites completed successfully!")

if __name__ == "__main__":
    main()
