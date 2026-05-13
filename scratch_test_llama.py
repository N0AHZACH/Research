import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
base = model.model
input_ids = torch.tensor([[1, 2, 3]])
h = base.embed_tokens(input_ids)
position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
print(h.shape, position_ids.shape)
try:
    position_embeddings = base.rotary_emb(h, position_ids)
    print("rotary_emb success")
except Exception as e:
    print("rotary_emb failed:", e)

try:
    out = base.layers[0](h, position_embeddings=position_embeddings)
    print("layer 0 success")
except Exception as e:
    print("layer 0 failed:", e)
