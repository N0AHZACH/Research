import os

scripts = [
    'exp11_large_model_routing.py',
    'exp14_qwen_baseline.py',
    'exp15_qwen_stochastic.py',
    'exp16_openllama_baseline.py',
    'exp17_openllama_stochastic.py'
]

for script in scripts:
    with open(script, 'r') as f:
        code = f.read()
    
    old_init = '''        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = enc["labels"].clone()
            self.labels[self.attention_mask == 0] = -100'''
            
    new_init = '''        def __init__(self, enc):
            import torch
            self.input_ids = torch.as_tensor(enc["input_ids"])
            self.attention_mask = torch.as_tensor(enc["attention_mask"])
            self.labels = self.input_ids.clone()
            self.labels[self.attention_mask == 0] = -100'''
            
    code = code.replace(old_init, new_init)
    
    with open(script, 'w') as f:
        f.write(code)
