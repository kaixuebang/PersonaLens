"""
Phase 2: Activation Collection Pipeline

Hooks into transformer layers to collect hidden-state activations from
contrastive prompt pairs. Saves activations per layer for downstream
persona vector extraction.

Usage:
    python collect_activations.py --model Qwen/Qwen3-0.6B --trait openness
    python collect_activations.py --model Qwen/Qwen3-0.6B --trait all
"""

import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.prompts.contrastive_prompts import (
    apply_chat_template_safe,
    get_contrastive_pairs,
    get_all_trait_names,
    BIG_FIVE_PROMPTS,
    DEFENSE_MECHANISM_PROMPTS,
)


def collect_hidden_states(model, tokenizer, messages, device, max_new_tokens=1):
    """
    Run a forward pass through the model with the given chat messages.
    Collect hidden states at ALL layers for the last input token position.

    Returns:
        hidden_states: dict mapping layer_idx -> numpy array of shape (hidden_dim,)
    """
    # Apply chat template
    text = apply_chat_template_safe(
        tokenizer, messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # outputs.hidden_states is a tuple: (embedding_output, layer_0, layer_1, ..., layer_N)
    # We skip the embedding layer (index 0) and take the last token position
    hidden_states = {}
    for layer_idx in range(1, len(outputs.hidden_states)):
        # Shape: (batch_size, seq_len, hidden_dim) -> take last token
        h = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
        hidden_states[layer_idx - 1] = h  # 0-indexed layers

    return hidden_states


def collect_for_trait(model, tokenizer, trait_name, device):
    """
    Collect activations for all contrastive pairs of a given trait.

    Returns:
        positive_activations: dict[layer_idx] -> np.array of shape (n_pairs, hidden_dim)
        negative_activations: dict[layer_idx] -> np.array of shape (n_pairs, hidden_dim)
    """
    pairs = get_contrastive_pairs(trait_name)
    n_pairs = len(pairs)

    positive_acts = {}
    negative_acts = {}

    print(f"\n  Collecting activations for trait: {trait_name} ({n_pairs} pairs)")

    for i, (pos_msgs, neg_msgs) in enumerate(tqdm(pairs, desc=f"  {trait_name}")):
        pos_hidden = collect_hidden_states(model, tokenizer, pos_msgs, device)
        neg_hidden = collect_hidden_states(model, tokenizer, neg_msgs, device)

        if i == 0:
            # Initialize arrays
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


def main():
    parser = argparse.ArgumentParser(
        description="Collect hidden-state activations for persona analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--trait",
        type=str,
        default="all",
        help="Trait name or 'all' for all traits. Options: "
        + ", ".join(get_all_trait_names()),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/activations",
        help="Directory to save activation files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
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

    # Print model info
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Determine traits to process
    if args.trait == "all":
        traits = get_all_trait_names()
    else:
        traits = [args.trait]

    # Create model-specific output directory
    model_short = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_short)
    os.makedirs(output_dir, exist_ok=True)

    # Collect activations for each trait
    for trait_name in traits:
        print(f"\n{'=' * 60}")
        print(f"Processing trait: {trait_name}")
        print(f"{'=' * 60}")

        pos_acts, neg_acts = collect_for_trait(
            model, tokenizer, trait_name, args.device
        )

        # Save activations
        trait_dir = os.path.join(output_dir, trait_name)
        os.makedirs(trait_dir, exist_ok=True)

        for layer_idx in pos_acts:
            np.save(
                os.path.join(trait_dir, f"pos_layer_{layer_idx}.npy"),
                pos_acts[layer_idx],
            )
            np.save(
                os.path.join(trait_dir, f"neg_layer_{layer_idx}.npy"),
                neg_acts[layer_idx],
            )

        print(f"  Saved activations to {trait_dir}/")
        print(f"  Layers: {len(pos_acts)}, Samples per layer: {pos_acts[0].shape[0]}")

    print("\n✓ Activation collection complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
