import os
import glob
import re

# 1. Create Directories
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 2. Refactor Training Scripts (exp23 to exp28, and exp8)
for py_file in glob.glob("exp*.py"):
    with open(py_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    original_content = content

    # Add os.makedirs if it doesn't exist
    if 'CSV_FILENAME = f"exp' in content and 'os.makedirs("results", exist_ok=True)' not in content:
        content = content.replace('TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")', 
                                  'TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\nos.makedirs("results", exist_ok=True)\nos.makedirs("checkpoints", exist_ok=True)')
    
    # Update CSV_FILENAME
    content = re.sub(r'CSV_FILENAME\s*=\s*f"exp([^"]+)\.csv"', r'CSV_FILENAME = f"results/exp\1.csv"', content)
    
    # Update SAVE_DIR
    content = re.sub(r'SAVE_DIR\s*=\s*f"exp([^"]+)"', r'SAVE_DIR     = f"checkpoints/exp\1"', content)

    # 3. Refactor Eval Harnesses
    if 'eval_harness' in py_file:
        content = content.replace('RESEARCH_DIR.glob(f"{pattern}', '(RESEARCH_DIR / "checkpoints").glob(f"{pattern}')
        content = content.replace('RESEARCH_DIR.glob(pattern)', '(RESEARCH_DIR / "checkpoints").glob(pattern)')
        content = content.replace('CSV_OUT     = RESEARCH_DIR / f"exp', 'CSV_OUT     = RESEARCH_DIR / "results" / f"exp')
        content = content.replace('JSON_OUT    = RESEARCH_DIR / f"exp', 'JSON_OUT    = RESEARCH_DIR / "results" / f"exp')
        content = re.sub(r'glob\.glob\(str\(RESEARCH_DIR \/ "exp([^"]+)\.json"\)\)', r'glob.glob(str(RESEARCH_DIR / "results" / "exp\1.json"))', content)

    if content != original_content:
        with open(py_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Refactored: {py_file}")
