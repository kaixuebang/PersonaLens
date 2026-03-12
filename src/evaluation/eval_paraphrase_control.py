"""
Paraphrase Control Experiment - Address Prompt Confounding

This script tests whether linear probes detect personality representations
or just prompt keywords by training on one prompt template and testing on another.

Experimental Design:
1. Collect activations using Template A (original prompts)
2. Collect activations using Template B (paraphrased prompts)
3. Train linear probe on Template A activations
4. Test probe on Template B activations
5. If accuracy remains high, vectors represent personality (not keywords)

Usage:
    python eval_paraphrase_control.py --model Qwen/Qwen3-0.6B --trait openness
"""

import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.prompts.contrastive_prompts import (
    get_contrastive_pairs,
    apply_chat_template_safe,
)
from src.prompts.paraphrase_prompts import (
    get_paraphrase_pairs,
    get_available_paraphrase_traits,
)


def collect_hidden_states(model, tokenizer, messages, device):
    """Collect hidden states at last token position for all layers."""
    text = apply_chat_template_safe(
        tokenizer, messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = {}
    for layer_idx in range(1, len(outputs.hidden_states)):
        h = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
        hidden_states[layer_idx - 1] = h

    return hidden_states


def collect_activations_for_template(model, tokenizer, trait_name, template, device):
    """Collect activations for a specific prompt template."""
    if template == "A":
        pairs = get_contrastive_pairs(trait_name)
    elif template == "B":
        pairs = get_paraphrase_pairs(trait_name, template="B")
    else:
        raise ValueError(f"Unknown template: {template}")

    n_pairs = len(pairs)
    positive_acts = {}
    negative_acts = {}

    print(
        f"  Collecting activations for {trait_name} (Template {template}, {n_pairs} pairs)"
    )

    for i, (pos_msgs, neg_msgs) in enumerate(
        tqdm(pairs, desc=f"  Template {template}")
    ):
        pos_hidden = collect_hidden_states(model, tokenizer, pos_msgs, device)
        neg_hidden = collect_hidden_states(model, tokenizer, neg_msgs, device)

        if i == 0:
            for layer_idx in pos_hidden:
                hidden_dim = pos_hidden[layer_idx].shape[0]
                positive_acts[layer_idx] = np.zeros(
                    (n_pairs, hidden_dim), dtype=np.float32
                )
                negative_acts[layer_idx] = np.zeros(
                    (n_pairs, hidden_dim), dtype=np.float32
                )

        for layer_idx in pos_hidden:
            positive_acts[layer_idx][i] = pos_hidden[layer_idx]
            negative_acts[layer_idx][i] = neg_hidden[layer_idx]

    return positive_acts, negative_acts


def train_probe(pos_acts, neg_acts, C=0.01):
    """Train a linear probe on given activations."""
    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

    clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs")
    clf.fit(X, y)

    return clf


def test_probe(clf, pos_acts, neg_acts):
    """Test probe on given activations."""
    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    return accuracy


def run_paraphrase_control(model, tokenizer, trait_name, device):
    """
    Run paraphrase control experiment:
    - Train on Template A, test on Template B
    - Train on Template B, test on Template A
    """
    print(f"\n{'=' * 60}")
    print(f"Paraphrase Control Experiment: {trait_name}")
    print(f"{'=' * 60}")

    # Collect activations for both templates
    print("\n[1/4] Collecting Template A activations...")
    pos_A, neg_A = collect_activations_for_template(
        model, tokenizer, trait_name, "A", device
    )

    print("\n[2/4] Collecting Template B activations...")
    pos_B, neg_B = collect_activations_for_template(
        model, tokenizer, trait_name, "B", device
    )

    layers = sorted(pos_A.keys())
    results = {
        "trait": trait_name,
        "n_layers": len(layers),
        "template_A_samples": pos_A[layers[0]].shape[0],
        "template_B_samples": pos_B[layers[0]].shape[0],
        "layers": {},
    }

    print("\n[3/4] Training on Template A, testing on Template B...")
    for layer in tqdm(layers, desc="  A→B"):
        # Train on A, test on B
        clf_A = train_probe(pos_A[layer], neg_A[layer])
        acc_A_to_B = test_probe(clf_A, pos_B[layer], neg_B[layer])

        # Also compute within-template accuracy for reference
        acc_A_to_A = test_probe(clf_A, pos_A[layer], neg_A[layer])

        results["layers"][layer] = {
            "train_A_test_A": float(acc_A_to_A),
            "train_A_test_B": float(acc_A_to_B),
        }

    print("\n[4/4] Training on Template B, testing on Template A...")
    for layer in tqdm(layers, desc="  B→A"):
        # Train on B, test on A
        clf_B = train_probe(pos_B[layer], neg_B[layer])
        acc_B_to_A = test_probe(clf_B, pos_A[layer], neg_A[layer])

        # Also compute within-template accuracy
        acc_B_to_B = test_probe(clf_B, pos_B[layer], neg_B[layer])

        results["layers"][layer]["train_B_test_B"] = float(acc_B_to_B)
        results["layers"][layer]["train_B_test_A"] = float(acc_B_to_A)

    # Find best cross-template layer
    best_layer = max(layers, key=lambda l: results["layers"][l]["train_A_test_B"])
    best_acc_A_to_B = results["layers"][best_layer]["train_A_test_B"]
    best_acc_B_to_A = results["layers"][best_layer]["train_B_test_A"]

    results["summary"] = {
        "best_layer": int(best_layer),
        "best_cross_template_acc_A_to_B": float(best_acc_A_to_B),
        "best_cross_template_acc_B_to_A": float(best_acc_B_to_A),
        "mean_cross_template_acc": float((best_acc_A_to_B + best_acc_B_to_A) / 2),
    }

    print(f"\n{'=' * 60}")
    print(f"Results Summary:")
    print(f"  Best Layer: L{best_layer}")
    print(f"  Train A → Test B: {best_acc_A_to_B:.3f}")
    print(f"  Train B → Test A: {best_acc_B_to_A:.3f}")
    print(f"  Mean Cross-Template: {results['summary']['mean_cross_template_acc']:.3f}")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase control experiment for prompt confounding"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name"
    )
    parser.add_argument("--trait", type=str, default="openness", help="Trait to test")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paraphrase_control_results",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (auto-detect if None)"
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # Check if trait has paraphrase template
    available_traits = get_available_paraphrase_traits()
    if args.trait not in available_traits:
        print(f"ERROR: Trait '{args.trait}' does not have paraphrase template.")
        print(f"Available traits: {', '.join(available_traits)}")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()

    # Run experiment
    results = run_paraphrase_control(model, tokenizer, args.trait, args.device)

    # Save results
    model_short = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"paraphrase_control_{args.trait}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Interpretation
    mean_cross = results["summary"]["mean_cross_template_acc"]
    if mean_cross >= 0.85:
        print(
            "\n✓ STRONG EVIDENCE: High cross-template accuracy suggests probes detect"
        )
        print("  personality representations, not prompt keywords.")
    elif mean_cross >= 0.70:
        print(
            "\n⚠ MODERATE EVIDENCE: Cross-template accuracy is decent but not conclusive."
        )
        print("  Some prompt confounding may exist.")
    else:
        print("\n✗ WEAK EVIDENCE: Low cross-template accuracy suggests probes may be")
        print("  detecting prompt-specific features rather than personality.")


if __name__ == "__main__":
    main()
