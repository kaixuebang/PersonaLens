"""
Contrastive Prompts - Unified Interface

Refactored to use personality_config.py for better maintainability and extensibility.

This module provides backward-compatible interface while internally using
the new configuration-driven approach.
"""

import random
from typing import List, Tuple, Dict, Optional

# Import from new config system
from src.prompts.personality_config import (
    get_framework_config,
    list_frameworks,
    get_all_traits,
    TraitConfig,
    FrameworkConfig,
)


# ============================================================
# Backward-compatible constants (for existing code)
# These are now dynamically generated from config
# ============================================================


def _load_framework_traits(framework_name: str) -> Dict:
    """Load traits from config and convert to legacy format."""
    try:
        config = get_framework_config(framework_name)
        traits = {}
        for trait_name, trait_config in config.traits.items():
            traits[trait_name] = {
                "high_system": trait_config.high_prompts,
                "low_system": trait_config.low_prompts,
                "scenarios": trait_config.scenarios,
            }
        return traits
    except ValueError:
        return {}


# Lazy-loaded trait dictionaries for backward compatibility
_BIG_FIVE_PROMPTS = None
_DEFENSE_MECHANISM_PROMPTS = None
_MBTI_PROMPTS = None


def _get_big_five_prompts() -> Dict:
    """Get Big Five prompts (lazy loading)."""
    global _BIG_FIVE_PROMPTS
    if _BIG_FIVE_PROMPTS is None:
        _BIG_FIVE_PROMPTS = _load_framework_traits("bigfive")
    return _BIG_FIVE_PROMPTS


def _get_defense_prompts() -> Dict:
    """Get defense mechanism prompts (lazy loading)."""
    global _DEFENSE_MECHANISM_PROMPTS
    if _DEFENSE_MECHANISM_PROMPTS is None:
        _DEFENSE_MECHANISM_PROMPTS = _load_framework_traits("defense")
    return _DEFENSE_MECHANISM_PROMPTS


def _get_mbti_prompts() -> Dict:
    """Get MBTI prompts (lazy loading)."""
    global _MBTI_PROMPTS
    if _MBTI_PROMPTS is None:
        _MBTI_PROMPTS = _load_framework_traits("mbti")
    return _MBTI_PROMPTS


class _LazyDict:
    """Module-level lazy-loading dict proxy for backward compatibility."""
    def __init__(self, loader):
        self._loader = loader
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = self._loader()
        return self._data

    def __getitem__(self, key):
        return self._load()[key]

    def __contains__(self, key):
        return key in self._load()

    def keys(self):
        return self._load().keys()

    def values(self):
        return self._load().values()

    def items(self):
        return self._load().items()

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def __repr__(self):
        return repr(self._load())


BIG_FIVE_PROMPTS = _LazyDict(_get_big_five_prompts)
DEFENSE_MECHANISM_PROMPTS = _LazyDict(_get_defense_prompts)


# ============================================================
# New Unified Interface
# ============================================================


class ContrastivePromptGenerator:
    """
    Unified generator for contrastive prompts across all personality frameworks.

    Usage:
        # Initialize generator
        gen = ContrastivePromptGenerator(framework='bigfive', seed=42)

        # Get prompts for a specific trait
        pairs = gen.get_contrastive_pairs('openness')

        # Get all available traits
        traits = gen.get_available_traits()

        # Switch framework
        gen.set_framework('mbti')
        mbti_pairs = gen.get_contrastive_pairs('extraversion_mbti')
    """

    def __init__(self, framework: str = "bigfive", seed: int = 42):
        """
        Initialize generator.

        Args:
            framework: Personality framework ('bigfive', 'mbti', 'defense')
            seed: Random seed for reproducible prompt selection
        """
        self.framework_name = framework
        self.seed = seed
        self.config = get_framework_config(framework)
        self.rng = random.Random(seed)

    def set_framework(self, framework: str) -> None:
        """Switch to a different framework."""
        self.framework_name = framework
        self.config = get_framework_config(framework)

    def get_available_traits(self) -> List[str]:
        """Return list of available trait names."""
        return self.config.get_trait_names()

    def get_trait_info(self, trait_name: str) -> Dict:
        """Get metadata about a trait."""
        trait = self.config.get_trait(trait_name)
        if trait is None:
            raise ValueError(f"Trait '{trait_name}' not found in {self.framework_name}")
        return {
            "name": trait.name,
            "description": trait.description,
            "n_high_prompts": len(trait.high_prompts),
            "n_low_prompts": len(trait.low_prompts),
            "n_scenarios": len(trait.scenarios),
        }

    def get_contrastive_pairs(
        self,
        trait_name: str,
        n_pairs: Optional[int] = None,
        use_all_scenarios: bool = True,
        sampling_mode: str = "cartesian",
    ) -> List[Tuple[List[Dict], List[Dict]]]:
        trait = self.config.get_trait(trait_name)
        if trait is None:
            available = self.get_available_traits()
            raise ValueError(
                f"Unknown trait: {trait_name}. "
                f"Available in {self.framework_name}: {available}"
            )

        trait_rng = random.Random(self.seed + hash(trait_name))

        scenarios_to_use = (
            trait.scenarios
            if use_all_scenarios
            else trait_rng.sample(
                trait.scenarios,
                min(n_pairs or len(trait.scenarios), len(trait.scenarios)),
            )
        )

        pairs = []

        if sampling_mode == "cartesian":
            for scenario in scenarios_to_use:
                for high_sys in trait.high_prompts:
                    for low_sys in trait.low_prompts:
                        high_msgs = [
                            {"role": "system", "content": high_sys},
                            {"role": "user", "content": scenario},
                        ]
                        low_msgs = [
                            {"role": "system", "content": low_sys},
                            {"role": "user", "content": scenario},
                        ]
                        pairs.append((high_msgs, low_msgs))
        else:
            for scenario in scenarios_to_use:
                high_sys = trait_rng.choice(trait.high_prompts)
                low_sys = trait_rng.choice(trait.low_prompts)
                high_msgs = [
                    {"role": "system", "content": high_sys},
                    {"role": "user", "content": scenario},
                ]
                low_msgs = [
                    {"role": "system", "content": low_sys},
                    {"role": "user", "content": scenario},
                ]
                pairs.append((high_msgs, low_msgs))

        if n_pairs is not None and sampling_mode == "cartesian":
            if len(pairs) > n_pairs:
                trait_rng.shuffle(pairs)
                pairs = pairs[:n_pairs]

        return pairs

    def get_framework_metadata(self) -> Dict:
        """Get metadata about the current framework."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "n_traits": len(self.config.traits),
            "trait_names": self.get_available_traits(),
            "metadata": self.config.metadata,
        }


# ============================================================
# Backward-Compatible Functions
# ============================================================


def get_all_trait_names() -> List[str]:
    """Return all trait names from all frameworks (backward compatible)."""
    return get_all_traits()


def get_contrastive_pairs(
    trait_name: str,
    framework: Optional[str] = None,
    seed: int = 42,
    sampling_mode: str = "cartesian",
    n_pairs: Optional[int] = None,
) -> List[Tuple[List[Dict], List[Dict]]]:
    if framework is None:
        framework = _detect_framework(trait_name)
    gen = ContrastivePromptGenerator(framework=framework, seed=seed)
    return gen.get_contrastive_pairs(
        trait_name, n_pairs=n_pairs, sampling_mode=sampling_mode
    )


def _detect_framework(trait_name: str) -> str:
    """Auto-detect which framework a trait belongs to."""
    # Check each framework
    for fw_name in list_frameworks():
        config = get_framework_config(fw_name)
        if trait_name in config.traits:
            return fw_name

    # Default fallback for backward compatibility
    if trait_name in [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]:
        return "bigfive"
    else:
        raise ValueError(f"Cannot auto-detect framework for trait: {trait_name}")


def apply_chat_template_safe(tokenizer, messages, **kwargs):
    """
    Safely apply chat template with fallback for models without system role support.

    Args:
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role' and 'content' keys
        **kwargs: Additional arguments to pass to apply_chat_template

    Returns:
        Formatted chat template string
    """
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        if any(
            pattern in error_msg
            for pattern in [
                "system role not supported",
                "system",
                "role",
                "jinja2",
            ]
        ):
            # Fallback: merge system message into first user message
            system_content = ""
            user_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                else:
                    user_messages.append(msg)

            # Prepend system content to first user message
            if system_content and user_messages:
                if user_messages[0].get("role") == "user":
                    user_messages[0]["content"] = (
                        f"{system_content}\n\n{user_messages[0]['content']}"
                    )

            try:
                return tokenizer.apply_chat_template(user_messages, **kwargs)
            except Exception as e2:
                # Final fallback: manual formatting
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                return "\n\n".join(prompt_parts) + "\n\nAssistant:"
        else:
            raise


# ============================================================
# New Utility Functions for Analysis
# ============================================================


def compare_framework_traits(framework1: str, framework2: str) -> Dict:
    """
    Compare traits between two frameworks.

    Returns:
        Dict with comparison metrics
    """
    config1 = get_framework_config(framework1)
    config2 = get_framework_config(framework2)

    return {
        "framework1": {
            "name": config1.name,
            "n_traits": len(config1.traits),
            "traits": config1.get_trait_names(),
        },
        "framework2": {
            "name": config2.name,
            "n_traits": len(config2.traits),
            "traits": config2.get_trait_names(),
        },
        "total_prompts_fw1": sum(
            len(t.high_prompts) + len(t.low_prompts) for t in config1.traits.values()
        ),
        "total_prompts_fw2": sum(
            len(t.high_prompts) + len(t.low_prompts) for t in config2.traits.values()
        ),
        "total_scenarios_fw1": sum(len(t.scenarios) for t in config1.traits.values()),
        "total_scenarios_fw2": sum(len(t.scenarios) for t in config2.traits.values()),
    }


def get_trait_overlap(framework1: str, framework2: str) -> List[str]:
    """
    Find traits that appear in both frameworks (by name).

    This helps identify potentially overlapping constructs
    (e.g., Big Five Extraversion vs MBTI Extraversion).
    """
    config1 = get_framework_config(framework1)
    config2 = get_framework_config(framework2)

    traits1 = set(config1.get_trait_names())
    traits2 = set(config2.get_trait_names())

    return list(traits1 & traits2)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Test new unified interface
    print("Testing new ContrastivePromptGenerator...")

    for framework in ["bigfive", "mbti"]:
        print(f"\n{framework.upper()}:")
        gen = ContrastivePromptGenerator(framework=framework)

        traits = gen.get_available_traits()
        print(f"  Traits: {traits}")

        # Test first trait
        if traits:
            first_trait = traits[0]
            pairs = gen.get_contrastive_pairs(first_trait)
            print(f"  {first_trait}: {len(pairs)} prompt pairs")

            # Show first pair
            if pairs:
                high, low = pairs[0]
                print(f"    High: {high[0]['content'][:50]}...")
                print(f"    Low:  {low[0]['content'][:50]}...")

    # Test backward compatibility
    print("\n\nTesting backward compatibility...")
    old_style_pairs = get_contrastive_pairs("openness")
    print(f"Old-style openness: {len(old_style_pairs)} pairs")

    # Test framework comparison
    print("\n\nComparing Big Five vs MBTI:")
    comparison = compare_framework_traits("bigfive", "mbti")
    print(f"  Big Five: {comparison['framework1']['n_traits']} traits")
    print(f"  MBTI: {comparison['framework2']['n_traits']} traits")
