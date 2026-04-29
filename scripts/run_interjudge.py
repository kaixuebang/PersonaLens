"""
Inter-Judge Agreement Study:
Score a balanced subset with Qwen2.5-7B as secondary judge,
then compute agreement with Gemma-2-2B (primary judge).
"""
import json, os, re, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

BFI = Path("results/bfi_behavioral_v2")
OUTPUT = Path("results/interjudge_agreement")
OUTPUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

ALL_MODELS = [
    "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen3-0.6B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
    "unsloth_Llama-3.2-1B-Instruct",
    "Qwen_Qwen2.5-1.5B-Instruct",
    "unsloth_gemma-2-2b-it",
    "_data0_shizitong_models_Phi3-mini-128k-instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "mistralai_Mistral-7B-Instruct-v0.1",
    "unsloth_llama-3-8B-Instruct",
    "unsloth_Llama-3.1-8B-Instruct",
    "_data0_shizitong_models_Llama-2-7b-chat-hf",
    "_data0_shizitong_models_DeepSeek-R1-Distill-Qwen-14B",
    "Qwen_Qwen2.5-14B-Instruct",
]

RUBRICS = {
    "openness": {
        "description": "Openness reflects imagination, intellectual curiosity, and willingness to explore new ideas and experiences.",
        "high": ["  - Eager to try new experiences", "  - Intellectually curious", "  - Appreciates art and novelty", "  - Prefers variety over routine"],
        "low": ["  - Prefers familiar routines", "  - Resists new ideas", "  - Practical and conventional", "  - Dislikes ambiguity"],
    },
    "conscientiousness": {
        "description": "Conscientiousness reflects organization, responsibility, and goal-directed self-discipline.",
        "high": ["  - Highly organized and disciplined", "  - Follows through on commitments", "  - Detail-oriented", "  - Plans ahead systematically"],
        "low": ["  - Disorganized and impulsive", "  - Misses deadlines", "  - Careless with details", "  - Procrastinates frequently"],
    },
    "extraversion": {
        "description": "Extraversion reflects sociability, assertiveness, and positive emotionality.",
        "high": ["  - Enjoys social gatherings", "  - Assertive and talkative", "  - Energetic and enthusiastic", "  - Seeks excitement"],
        "low": ["  - Prefers solitude", "  - Reserved and quiet", "  - Avoids attention", "  - Low energy in groups"],
    },
    "agreeableness": {
        "description": "Agreeableness reflects cooperation, trust, and concern for others.",
        "high": ["  - Helpful and cooperative", "  - Trusting of others", "  - Compassionate and empathetic", "  - Avoids conflict"],
        "low": ["  - Competitive and skeptical", "  - Critical and demanding", "  - Prioritizes self-interest", "  - Confrontational"],
    },
    "neuroticism": {
        "description": "Neuroticism reflects emotional instability, anxiety, and negative emotionality.",
        "high": ["  - Easily stressed and worried", "  - Experiences mood swings", "  - Self-conscious and anxious", "  - Reacts strongly to setbacks"],
        "low": ["  - Emotionally stable and calm", "  - Rarely anxious", "  - Resilient under pressure", "  - Even-tempered"],
    },
}

JUDGE_PROMPT = """You are a psychologist evaluating a person's response for the personality trait "{trait}".

TRAIT: {trait}
{description}

HIGH {trait_upper} BEHAVIORS:
{high}

LOW {trait_upper} BEHAVIORS:
{low}

PERSON'S RESPONSE:
\"\"\"
{response}
\"\"\"

Rate how much this response reflects {trait} on a 1-5 scale:
1 = Very Low (clearly shows LOW indicators)
2 = Low (mostly shows LOW indicators)
3 = Moderate (mixed or neutral)
4 = High (mostly shows HIGH indicators)
5 = Very High (clearly shows HIGH indicators)

First write one sentence of reasoning, then output your rating on a new line.
Format: RATING: [1-5]"""

def extract_rating(text):
    match = re.search(r"RATING:\s*(\d)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:rating|score)\s*(?:is|=|:)\s*(\d)", text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 5: return val
    nums = re.findall(r"\b([1-5])\b", text[-50:])
    if nums: return int(nums[-1])
    return 3

def build_prompt(response, trait):
    r = RUBRICS[trait]
    return JUDGE_PROMPT.format(
        trait=trait.capitalize(), trait_upper=trait.upper(),
        description=r["description"], high="\n".join(r["high"]),
        low="\n".join(r["low"]), response=response)

def select_subset():
    subset = []
    for model_dir in ALL_MODELS:
        for trait in TRAITS:
            p = BFI / model_dir / f"responses_{trait}.json"
            if not p.exists(): continue
            d = json.load(open(p))
            alphas = sorted([float(k) for k in d["results"].keys()])
            if len(alphas) >= 7:
                idxs = [0, 3, -1]
            elif len(alphas) >= 3:
                idxs = [0, len(alphas)//2, -1]
            else:
                idxs = [0, -1]
            for idx in idxs:
                a_key = str(float(alphas[idx]))
                srs = d["results"][a_key]["scenario_results"]
                picks = [0, len(srs)//2] if len(srs) >= 2 else [0]
                for pick in picks:
                    if pick < len(srs):
                        sr = srs[pick]
                        subset.append({
                            "model": model_dir, "trait": trait,
                            "alpha": alphas[idx], "response": sr["response"],
                            "gemma_rating": sr.get("judge_rating", None),
                        })
    return subset

def main():
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "5"
    device = f"cuda:{gpu_id}"
    print(f"Device: {device}")
    
    subset = select_subset()
    print(f"Subset: {len(subset)} responses")
    
    output_file = OUTPUT / "interjudge_results.json"
    results = []
    
    print("Loading Qwen2.5-7B-Instruct as secondary judge...")
    judge_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(judge_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        judge_name, dtype=torch.float16, device_map=device, local_files_only=True
    )
    model.eval()
    
    for i, item in enumerate(subset):
        prompt = build_prompt(item["response"], item["trait"])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        qwen_rating = extract_rating(gen)
        
        item["qwen_rating"] = qwen_rating
        item["qwen_response"] = gen[:200]
        results.append(item)
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(subset)}")
            with open(output_file, "w") as f:
                json.dump({"results": results, "n_total": len(results)}, f, indent=2)
    
    with open(output_file, "w") as f:
        json.dump({"results": results, "n_total": len(results)}, f, indent=2)
    
    # Agreement
    gemma = [r["gemma_rating"] for r in results if r["gemma_rating"] is not None]
    qwen = [r["qwen_rating"] for r in results if r["gemma_rating"] is not None]
    
    if gemma and qwen:
        pr, pp = stats.pearsonr(gemma, qwen)
        sr, sp = stats.spearmanr(gemma, qwen)
        within1 = sum(1 for g, q in zip(gemma, qwen) if abs(g - q) <= 1) / len(gemma)
        exact = sum(1 for g, q in zip(gemma, qwen) if g == q) / len(gemma)
        mae = np.mean([abs(g - q) for g, q in zip(gemma, qwen)])
        
        print(f"\nINTER-JUDGE AGREEMENT (Gemma-2 vs Qwen2.5-7B)")
        print(f"  n={len(gemma)}")
        print(f"  Pearson r={pr:.3f} (p={pp:.4f})")
        print(f"  Spearman rho={sr:.3f} (p={sp:.4f})")
        print(f"  Exact={exact:.3f}, Within±1={within1:.3f}, MAE={mae:.3f}")
        
        with open(OUTPUT / "agreement_metrics.json", "w") as f:
            json.dump({"n": len(gemma), "pearson_r": pr, "pearson_p": pp,
                       "spearman_r": sr, "spearman_p": sp,
                       "exact_agreement": exact, "within_1": within1, "mae": mae}, f, indent=2)
    
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Done!")

if __name__ == "__main__":
    main()
