import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import datetime

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 128
NUM_BATCHES = 50
BATCH_SIZE = 8

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"inference_benchmark_{TIMESTAMP}.csv"

def main():
    print("Loading model for Inference Benchmarking...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy input
    dummy_text = ["This is a test sentence for benchmarking the inference speed of the model."] * BATCH_SIZE
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    TOTAL_LAYERS = len(model.model.layers)
    original_layers = model.model.layers
    
    test_layer_counts = [TOTAL_LAYERS, 18, 14, 10, 6]
    
    headers = ["Active Layers", "Tokens Per Second", "Latency (ms/batch)"]
    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
    model.eval()
    with torch.no_grad():
        for count in test_layer_counts:
            print(f"\nBenchmarking with {count} active layers...")
            
            # Truncate model layers to simulate routing
            model.model.layers = nn.ModuleList(list(original_layers[:count]))
            
            # Warmup
            for _ in range(5):
                _ = model(**inputs)
                
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(NUM_BATCHES):
                _ = model(**inputs)
                
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_batch = total_time / NUM_BATCHES
            ms_per_batch = time_per_batch * 1000
            
            total_tokens = NUM_BATCHES * BATCH_SIZE * MAX_LENGTH
            tps = total_tokens / total_time
            
            print(f"--> Speed: {tps:.2f} Tokens/Sec | Latency: {ms_per_batch:.2f} ms/batch")
            
            with open(CSV_FILENAME, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([count, f"{tps:.2f}", f"{ms_per_batch:.2f}"])

    print(f"\nInference Benchmark complete! Results saved to {CSV_FILENAME}")

if __name__ == "__main__":
    main()
