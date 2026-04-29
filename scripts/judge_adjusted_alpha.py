"""Judge adjusted-alpha responses."""
import json, os, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import src.evaluation.judge_bfi_behavioral as judge

device = "cuda"
judge_name = "unsloth/gemma-2-2b-it"
adj_dir = "/data1/tongjizhou/persona/results/bfi_adjusted_alpha_midlayer"

print("Loading judge model...")
tokenizer = AutoTokenizer.from_pretrained(judge_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    judge_name, trust_remote_code=True, 
    dtype=torch.float16, device_map=device
)
model.eval()
print("Judge loaded.")

for model_dir in sorted(os.listdir(adj_dir)):
    model_path = os.path.join(adj_dir, model_dir)
    if not os.path.isdir(model_path):
        continue
    
    for f in sorted(os.listdir(model_path)):
        if f.startswith("responses_") and f.endswith(".json"):
            fpath = os.path.join(model_path, f)
            # Skip if already judged
            with open(fpath) as fh:
                data = json.load(fh)
            first_alpha = list(data.get("results", {}).keys())[0] if data.get("results") else None
            if first_alpha and "judge_mean" in data["results"][first_alpha]:
                print(f"SKIP (already judged): {model_dir}/{f}")
                continue
            
            trait = f.replace("responses_", "").replace(".json", "")
            print(f"\nJudging {model_dir} / {trait}...")
            judge.process_file(fpath, model, tokenizer, trait, device)
            print(f"  Done: {fpath}")

print("\nAll judging complete!")
