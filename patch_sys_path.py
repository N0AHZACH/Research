import os
from pathlib import Path

patch = """import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

def apply_patch():
    for folder in ["qwen_7b", "llama_8b"]:
        d = Path(folder)
        if not d.exists(): continue
        for py_file in d.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if "sys.path.append" not in content:
                py_file.write_text(patch + "\n" + content, encoding="utf-8")
                print(f"Patched: {py_file}")

if __name__ == "__main__":
    apply_patch()
