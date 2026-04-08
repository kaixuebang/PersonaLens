import json
import numpy as np
from pathlib import Path

BFI = Path("/data1/tongjizhou/persona/results/bfi_behavioral_v2")
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

MODELS = [
    ("Qwen_Qwen2.5-0.5B-Instruct", "Q2.5-0.5B"),
    ("Qwen_Qwen3-0.6B", "Q3-0.6B"),
    ("TinyLlama_TinyLlama-1.1B-Chat-v1.0", "TinyLlama"),
    ("unsloth_Llama-3.2-1B-Instruct", "Llama-3.2"),
    ("Qwen_Qwen2.5-1.5B-Instruct", "Q2.5-1.5B"),
    ("unsloth_gemma-2-2b-it", "Gemma-2B"),
    ("Qwen_Qwen2.5-7B-Instruct", "Q2.5-7B"),
    ("mistralai_Mistral-7B-Instruct-v0.1", "Mistral-7B"),
]

print("=" * 90)
print("CROSS-TRAIT INTERFERENCE ANALYSIS")
print("=" * 90)

summary = {}

for model_dir, name in MODELS:
    all_primary = []
    all_off = []

    for steered in TRAITS:
        path = BFI / model_dir / f"responses_{steered}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)

        alphas = sorted(d["results"].keys(), key=lambda x: float(x))
        alpha_max = alphas[-1]
        alpha_min = alphas[0]

        for target in TRAITS:
            sk = f"judge_rating_{target}"
            hi = [e[sk] for e in d["results"][alpha_max]["scenario_results"] if sk in e]
            lo = [e[sk] for e in d["results"][alpha_min]["scenario_results"] if sk in e]
            if hi and lo:
                delta = abs(np.mean(hi) - np.mean(lo))
                if target == steered:
                    all_primary.append(delta)
                else:
                    all_off.append(delta)

    mp = np.mean(all_primary) if all_primary else 0
    mo = np.mean(all_off) if all_off else 0
    mx = np.max(all_off) if all_off else 0
    sel = mp / mo if mo > 0 else float("inf")

    summary[name] = {"mp": mp, "mo": mo, "mx": mx, "sel": sel}

print(f"\n{'Model':<15} {'Primary':>8} {'OffDiag':>8} {'MaxOff':>8} {'Select.':>8}")
print("-" * 50)
for _, name in MODELS:
    s = summary.get(name, {})
    mp = s.get("mp", 0)
    mo = s.get("mo", 0)
    mx = s.get("mx", 0)
    sel = s.get("sel", 0)
    print(f"{name:<15} {mp:>8.2f} {mo:>8.2f} {mx:>8.2f} {sel:>7.1f}x")

# Detailed per-steered-trait matrix for Mistral (strong steer)
print("\n" + "=" * 90)
print("DETAILED INTERFERENCE MATRIX: Mistral-7B (strong steer)")
print("=" * 90)
model_dir = "mistralai_Mistral-7B-Instruct-v0.1"
TS = {"openness": "Open", "conscientiousness": "Consc", "extraversion": "Extra",
      "agreeableness": "Agree", "neuroticism": "Neuro"}

header = f"{'Steer→':>8} | " + " | ".join(f"{TS[t]:>5}" for t in TRAITS)
print(header)
print("-" * len(header))

for steered in TRAITS:
    path = BFI / model_dir / f"responses_{steered}.json"
    with open(path) as f:
        d = json.load(f)
    alphas = sorted(d["results"].keys(), key=lambda x: float(x))
    hi_a = alphas[-1]
    lo_a = alphas[0]

    row = f"{TS[steered]:>8} |"
    for target in TRAITS:
        sk = f"judge_rating_{target}"
        hi = [e[sk] for e in d["results"][hi_a]["scenario_results"] if sk in e]
        lo = [e[sk] for e in d["results"][lo_a]["scenario_results"] if sk in e]
        delta = np.mean(hi) - np.mean(lo) if hi and lo else 0
        row += f" {delta:>5.2f} |"
    print(row)

# Same for Q2.5-7B (weak steer)
print("\n" + "=" * 90)
print("DETAILED INTERFERENCE MATRIX: Q2.5-7B (weak steer)")
print("=" * 90)
model_dir = "Qwen_Qwen2.5-7B-Instruct"

print(header)
print("-" * len(header))

for steered in TRAITS:
    path = BFI / model_dir / f"responses_{steered}.json"
    with open(path) as f:
        d = json.load(f)
    alphas = sorted(d["results"].keys(), key=lambda x: float(x))
    hi_a = alphas[-1]
    lo_a = alphas[0]

    row = f"{TS[steered]:>8} |"
    for target in TRAITS:
        sk = f"judge_rating_{target}"
        hi = [e[sk] for e in d["results"][hi_a]["scenario_results"] if sk in e]
        lo = [e[sk] for e in d["results"][lo_a]["scenario_results"] if sk in e]
        delta = np.mean(hi) - np.mean(lo) if hi and lo else 0
        row += f" {delta:>5.2f} |"
    print(row)

# Same for Gemma-2B
print("\n" + "=" * 90)
print("DETAILED INTERFERENCE MATRIX: Gemma-2B (weak original)")
print("=" * 90)
model_dir = "unsloth_gemma-2-2b-it"

print(header)
print("-" * len(header))

for steered in TRAITS:
    path = BFI / model_dir / f"responses_{steered}.json"
    with open(path) as f:
        d = json.load(f)
    alphas = sorted(d["results"].keys(), key=lambda x: float(x))
    hi_a = alphas[-1]
    lo_a = alphas[0]

    row = f"{TS[steered]:>8} |"
    for target in TRAITS:
        sk = f"judge_rating_{target}"
        hi = [e[sk] for e in d["results"][hi_a]["scenario_results"] if sk in e]
        lo = [e[sk] for e in d["results"][lo_a]["scenario_results"] if sk in e]
        delta = np.mean(hi) - np.mean(lo) if hi and lo else 0
        row += f" {delta:>5.2f} |"
    print(row)
