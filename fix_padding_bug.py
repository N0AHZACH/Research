import os

scripts = [
    'exp14_qwen_baseline.py',
    'exp15_qwen_stochastic.py',
    'exp16_openllama_baseline.py',
    'exp17_openllama_stochastic.py'
]

for script in scripts:
    with open(script, 'r') as f:
        code = f.read()
    
    # 1. Remove the slow/buggy python loop mask
    bad_mask_logic = '''        labels = o["input_ids"].copy()
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if o["attention_mask"][i][j] == 0:
                    labels[i][j] = -100
        o["labels"] = labels'''
    
    code = code.replace(bad_mask_logic, '        o["labels"] = o["input_ids"].copy()')
    
    # 2. Add the fast tensor-space mask in the RAMDataset init
    dataset_init_old = '''        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = enc["labels"]'''
            
    dataset_init_new = '''        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = enc["labels"].clone()
            self.labels[self.attention_mask == 0] = -100'''
            
    code = code.replace(dataset_init_old, dataset_init_new)
    
    with open(script, 'w') as f:
        f.write(code)
