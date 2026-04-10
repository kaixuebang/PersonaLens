"""
Debug script to investigate why steering doesn't work on Qwen models.

Measures:
1. Hidden state norms at each layer
2. Whether the steering hook actually fires
3. Perturbation ratio (alpha * ||v|| / ||h||) at each layer
4. Output difference with and without steering
"""

import os
import sys
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 27
TRAIT = "openness"
ALPHA = 10.0  # Use max alpha for most visible effect

model_short = MODEL.replace("/", "_")
vec_path = f"results/persona_vectors/{model_short}/{TRAIT}/vectors/mean_diff_layer_{LAYER}.npy"

print(f"Model: {MODEL}")
print(f"Steering layer: {LAYER}")
print(f"Vector: {vec_path}")

# Load vector
vec = np.load(vec_path)
print(f"Vector shape: {vec.shape}, L2 norm: {np.linalg.norm(vec):.4f}, RMS: {np.sqrt(np.mean(vec**2)):.6f}")

# Load model
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cuda", trust_remote_code=True
)
model.eval()

# Get layer list
layers = model.model.layers
print(f"Model has {len(layers)} layers")

# Hook to capture hidden states
captured = {}

def capture_hook(layer_idx):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured[layer_idx] = output[0].detach().cpu()
        else:
            captured[layer_idx] = output.detach().cpu()
    return hook_fn

# Register capture hooks on ALL layers
hooks = []
for i, layer in enumerate(layers):
    h = layer.register_forward_hook(capture_hook(i))
    hooks.append(h)

# Test prompt
prompt = "What do you think about trying a completely new hobby you've never considered before?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# Run WITHOUT steering
print("\n=== BASELINE (no steering) ===")
captured.clear()
with torch.no_grad():
    out_baseline = model.generate(**inputs, max_new_tokens=50, do_sample=False)
resp_baseline = tokenizer.decode(out_baseline[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Response: {resp_baseline[:200]}")

# Measure hidden state norms at each layer
print("\nHidden state norms per layer (baseline):")
baseline_norms = {}
for i in sorted(captured.keys()):
    h = captured[i]
    # h shape: [1, seq_len, hidden_dim] - take last token position
    last_token_h = h[0, -1, :].float().numpy()
    norm = np.linalg.norm(last_token_h)
    rms = np.sqrt(np.mean(last_token_h**2))
    baseline_norms[i] = (norm, rms)
    if i <= 3 or i >= len(layers) - 5 or i == LAYER:
        print(f"  Layer {i:2d}: L2={norm:.2f}, RMS={rms:.4f}")

# Now run WITH steering
print(f"\n=== STEERED (alpha={ALPHA}, layer={LAYER}) ===")

# Steering hook
steering_vec = torch.tensor(vec, dtype=torch.float16, device="cuda")
steering_fired = [False]

def steering_hook_fn(module, input, output):
    steering_fired[0] = True
    if isinstance(output, tuple):
        modified = list(output)
        modified[0] = output[0] + ALPHA * steering_vec.unsqueeze(0).unsqueeze(0)
        return tuple(modified)
    return output + ALPHA * steering_vec.unsqueeze(0).unsqueeze(0)

# Register steering hook on target layer
steer_hook = layers[LAYER].register_forward_hook(steering_hook_fn)

# Clear and re-register capture hooks
for h in hooks:
    h.remove()
hooks = []
captured.clear()
for i, layer in enumerate(layers):
    h = layer.register_forward_hook(capture_hook(i))
    hooks.append(h)

captured_steered = {}
# Save steered captured
def capture_steered_hook(layer_idx):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_steered[layer_idx] = output[0].detach().cpu()
        else:
            captured_steered[layer_idx] = output.detach().cpu()
    return hook_fn

# Re-register: capture hooks + steering hook
for h in hooks:
    h.remove()
hooks = []
captured.clear()
captured_steered.clear()

for i, layer in enumerate(layers):
    h = layer.register_forward_hook(capture_steered_hook(i))
    hooks.append(h)

with torch.no_grad():
    out_steered = model.generate(**inputs, max_new_tokens=50, do_sample=False)
resp_steered = tokenizer.decode(out_steered[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Steering hook fired: {steering_fired[0]}")
print(f"Response: {resp_steered[:200]}")

# Compare hidden state norms after steering
print(f"\nHidden state norms per layer (steered):")
for i in sorted(captured_steered.keys()):
    h = captured_steered[i]
    last_token_h = h[0, -1, :].float().numpy()
    norm = np.linalg.norm(last_token_h)
    rms = np.sqrt(np.mean(last_token_h**2))
    
    if i in captured:
        base_norm, base_rms = baseline_norms.get(i, (0, 0))
        delta_norm = norm - base_norm
        delta_pct = (delta_norm / base_norm * 100) if base_norm > 0 else 0
        if i <= 3 or i >= len(layers) - 5 or i == LAYER or i == LAYER + 1:
            print(f"  Layer {i:2d}: L2={norm:.2f} (Δ={delta_pct:+.2f}%), RMS={rms:.4f} (Δ={rms-base_rms:+.4f})")

# Perturbation ratio
if LAYER in captured and LAYER in captured_steered:
    base_h = captured[LAYER][0, -1, :].float().numpy()
    steered_h = captured_steered[LAYER][0, -1, :].float().numpy()
    diff = steered_h - base_h
    diff_norm = np.linalg.norm(diff)
    perturbation_ratio = diff_norm / np.linalg.norm(base_h) * 100
    
    print(f"\nPerturbation analysis at layer {LAYER}:")
    print(f"  ||h_baseline|| = {np.linalg.norm(base_h):.2f}")
    print(f"  ||h_steered|| = {np.linalg.norm(steered_h):.2f}")
    print(f"  ||Δh|| = {diff_norm:.4f}")
    print(f"  Perturbation ratio: {perturbation_ratio:.2f}%")
    print(f"  Expected ||Δh|| = alpha * ||v|| = {ALPHA} * {np.linalg.norm(vec):.4f} = {ALPHA * np.linalg.norm(vec):.4f}")
    print(f"  Actual ||Δh|| = {diff_norm:.4f}")
    print(f"  Match: {'YES' if abs(diff_norm - ALPHA * np.linalg.norm(vec)) < 0.1 else 'NO (RMSNorm overriding?)'}")

# Check if Qwen uses RMSNorm
print(f"\nLayer {LAYER} structure:")
layer = layers[LAYER]
for name, _ in layer.named_modules():
    if 'norm' in name.lower() or 'layernorm' in name.lower():
        print(f"  {name}: {type(getattr(layer, name.split('.')[-1], 'N/A'))}")

# Check next layer's input_layernorm
if LAYER + 1 < len(layers):
    next_layer = layers[LAYER + 1]
    print(f"\nLayer {LAYER+1} input_layernorm type: {type(next_layer.input_layernorm)}")
    # Check if it's RMSNorm
    if hasattr(next_layer.input_layernorm, 'weight'):
        w = next_layer.input_layernorm.weight
        print(f"  weight shape: {w.shape}, mean: {w.float().mean():.4f}, std: {w.float().std():.4f}")

# Cleanup
steer_hook.remove()
for h in hooks:
    h.remove()

print("\n=== DONE ===")
