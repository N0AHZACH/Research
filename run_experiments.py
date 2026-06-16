import os
import sys
import subprocess
import time
import argparse

EXPERIMENTS = {
    "qwen7b": [
        ("exp23_qwen7b_baseline.py", "Baseline LoRA Fine-Tuning"),
        ("exp24_qwen7b_stochastic.py", "Stochastic Depth Dropout Control"),
        ("exp25_qwen7b_token_routing.py", "DLR Token-Level Router Training"),
        ("exp22_qwen7b_eval_harness.py", "Evaluation Suite (MMLU/GSM8K/ARC/PPL)"),
    ],
    "llama8b": [
        ("exp26_llama8b_baseline.py", "Baseline LoRA Fine-Tuning"),
        ("exp27_llama8b_stochastic.py", "Stochastic Depth Dropout Control"),
        ("exp28_llama8b_token_routing.py", "DLR Token-Level Router Training"),
        ("exp29_llama8b_eval_harness.py", "Evaluation Suite (MMLU/GSM8K/ARC/PPL)"),
    ]
}

def run_script(script_name, description):
    print("=" * 80)
    print(f"  RUNNING: {script_name} - {description}")
    print("=" * 80)
    
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"{script_name.replace('.py', '')}.log")
    print(f"[INFO] Outputs are being piped to: {log_file}")
    
    start_time = time.time()
    try:
        # We run the command and stream to console and file simultaneously
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

def main():
    parser = argparse.ArgumentParser(description="DLR Phase 4 Experiment Suite Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qwen", action="store_true", help="Run all Qwen2.5-7B experiments")
    group.add_argument("--llama", action="store_true", help="Run all Llama3.1-8B experiments")
    group.add_argument("--all", action="store_true", help="Run all Qwen7B and Llama8B experiments")
    group.add_argument("--list", action="store_true", help="List all experimental scripts")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Phase 4 scaling experiments:")
        for model, scripts in EXPERIMENTS.items():
            print(f"\n{model.upper()} Suite:")
            for s, desc in scripts:
                status = "Exists" if os.path.exists(s) else "Missing!"
                print(f"  - {s:<32} | {desc:<45} | [{status}]")
        print()
        return

    selected_suites = []
    if args.qwen or args.all:
        selected_suites.append("qwen7b")
    if args.llama or args.all:
        selected_suites.append("llama8b")
        
    for suite in selected_suites:
        print(f"\n=== Starting Experimental Suite: {suite.upper()} ===")
        for script, desc in EXPERIMENTS[suite]:
            if not os.path.exists(script):
                print(f"[ERROR] Script '{script}' does not exist. Skipping.")
                continue
                
            success = run_script(script, desc)
            if not success:
                print(f"[HALT] Stopping suite execution due to failure in {script}.")
                sys.exit(1)
                
    print("\nAll selected experiment suites completed successfully!")

if __name__ == "__main__":
    main()
