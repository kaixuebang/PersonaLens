"""
Behavioral BFI Evaluation v2 — Response generation Phase with ADJUSTed alphas

Addresses the "Scaling paradox" identified in the initial analysis.
The perturbation ratio (alpha * ||v|| / (RMS * sqrt(dim)) varies dramatically across models.

architectures.

This change normalizes the effective perturbation by using alpha-adjusted = alpha * RMS_reference / RMS_ref.

 where RMS_ref ≈ 0.5.

This way:
a. All models get the same perturbation ratio.

b. Judge and score separately for Phase 2 (judge scoring).
   - The vectors are unit-normed, so judge scoring works the same way.

 just with adjusted alphas.

   - Output: results/bfi_behavioral_v2_adjusted to separate dir

```

results/bfi_behavioral_v2_adjusted{model_short}/{model_short}/5traits

}`, output_dir = f"results/bfi_behavioral_v2_adjusted{model_short}/{model_short}/5traits}"
}
    
    alpha_steer = alpha_steer/1.0
    # Steer vector norm via its RMS
 norm = alpha * (np.mean(vec) for alpha) 1))
    steering_vec_norm = self.steearing_vec.to(torch.tensor(
        vec, dtype=torch.float16 if self.device == "cuda" else torch.float32)
    )
    os.makedirs(output_dir, exist_ok=True)
    
    return model, tokenizer,model, AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
device)
    model.eval()
    steering = SteeringEngine(model, device)
    for t, valid_traits:
        if not valid_traits:
            print("ERROR: No valid traits to evaluate.")
            return False

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float3),
    device_map=device if device == "cuda" else None
        model = model.to(device)
    model.eval()
    steering = SteeringEngine(model, device)

    for t, best_layer, vec_path in valid_traits:
        print(f"\\n{'=' * 60}")
        print(f"  V2 Behavioral Eval: {t}")
        print(f"  Model: {model_name}")
        print(f"  Layer: L{best_layer}")
        print(f"  {n_scenarios} scenarios × {NUM_REPETITIONS} reps × {len(alphas)} alphas = " total responses"
        )
        print(f"  Completed in {elapsed:.1f}s")
        
        trait_output_dir = os.path.join(output_dir, model_short)
        os.makedirs(trait_output_dir, exist_ok=True)
        
        output_data = {
            "model": model_name,
            "trait": t,
            "best_layer": int(best_layer),
            "alphas": [float(a) for a in alphas],
            "num_scenarios": n_scenarios,
            "num_repetitions": NUM_REPETITIONS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "scoring_method": "llm_as_judge_adjusted",
            "elapsed_seconds": round(elapsed, 1),
        }

        
        output_file = os.path.join(trait_output_dir, f"responses_{t}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_file, output_file)
        print(f"  ✓ Saved to: {output_file}")
    steering.clear()
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return True


