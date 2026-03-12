"""
BFI-44 (Big Five Inventory) Evaluation for LLM Personality Steering

This module implements validated psychological assessment using the BFI-44 scale
to replace keyword-based heuristics. The BFI-44 is a widely-used 44-item questionnaire
for measuring Big Five personality traits.

Two evaluation modes:
1. Self-report: Model answers BFI-44 questions directly
2. LLM-as-Judge: GPT-4 rates generated text against BFI rubrics

References:
- John, O. P., & Srivastava, S. (1999). The Big Five trait taxonomy
- John, O. P., Donahue, E. M., & Kentle, R. L. (1991). The Big Five Inventory
"""

# BFI-44 Items (Simplified version with 8-10 items per trait)
# Format: (item_text, is_positive_keyed, trait)

BFI_44_ITEMS = {
    "openness": {
        "positive": [
            "I am original, come up with new ideas",
            "I am curious about many different things",
            "I am ingenious, a deep thinker",
            "I have an active imagination",
            "I am inventive",
            "I value artistic, aesthetic experiences",
            "I prefer work that is routine (R)",  # Reverse-scored
            "I like to reflect, play with ideas",
        ],
        "negative": [
            "I have few artistic interests",
            "I prefer work that is routine",
        ],
    },
    "conscientiousness": {
        "positive": [
            "I do a thorough job",
            "I am a reliable worker",
            "I persevere until the task is finished",
            "I do things efficiently",
            "I make plans and follow through with them",
            "I am systematic, like to keep things in order",
        ],
        "negative": [
            "I tend to be disorganized",
            "I tend to be lazy",
            "I can be somewhat careless",
            "I am easily distracted",
        ],
    },
    "extraversion": {
        "positive": [
            "I am talkative",
            "I am full of energy",
            "I generate a lot of enthusiasm",
            "I have an assertive personality",
            "I am outgoing, sociable",
        ],
        "negative": [
            "I tend to be quiet",
            "I am sometimes shy, inhibited",
            "I am reserved",
        ],
    },
    "agreeableness": {
        "positive": [
            "I am helpful and unselfish with others",
            "I have a forgiving nature",
            "I am generally trusting",
            "I am considerate and kind to almost everyone",
            "I like to cooperate with others",
        ],
        "negative": [
            "I tend to find fault with others",
            "I start quarrels with others",
            "I can be cold and aloof",
            "I am sometimes rude to others",
        ],
    },
    "neuroticism": {
        "positive": [
            "I can be tense",
            "I worry a lot",
            "I can be moody",
            "I get nervous easily",
            "I remain calm in tense situations (R)",  # Reverse-scored
        ],
        "negative": [
            "I am relaxed, handle stress well",
            "I am emotionally stable, not easily upset",
            "I remain calm in tense situations",
        ],
    },
}


def get_bfi_prompt(item_text, response_format="likert"):
    """
    Generate a prompt for BFI item assessment.

    Args:
        item_text: The BFI item statement
        response_format: "likert" (1-5 scale) or "binary" (agree/disagree)

    Returns:
        Formatted prompt string
    """
    if response_format == "likert":
        return f"""Rate how much you agree with the following statement about yourself on a scale from 1 to 5:

1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree

Statement: "{item_text}"

Respond with ONLY a single digit (1, 2, 3, 4, or 5):"""

    elif response_format == "binary":
        return f"""Do you agree or disagree with the following statement about yourself?

Statement: "{item_text}"

Respond with ONLY "Agree" or "Disagree":"""

    else:
        raise ValueError(f"Unknown response format: {response_format}")


def compute_bfi_score_from_ratings(ratings, trait_name):
    """
    Compute BFI trait score from item ratings.

    Args:
        ratings: Dict mapping item_text -> rating (1-5)
        trait_name: Name of the trait

    Returns:
        Mean score for the trait (1-5 scale)
    """
    items = BFI_44_ITEMS[trait_name]
    scores = []

    for item in items["positive"]:
        if item in ratings:
            scores.append(ratings[item])

    for item in items["negative"]:
        if item in ratings:
            # Reverse-score negative items
            scores.append(6 - ratings[item])

    if not scores:
        return None

    return sum(scores) / len(scores)


# LLM-as-Judge Rubrics for BFI Traits

BFI_JUDGE_RUBRICS = {
    "openness": {
        "description": "Openness to Experience reflects imagination, curiosity, and preference for variety.",
        "high_indicators": [
            "Uses abstract or philosophical language",
            "Expresses curiosity or interest in novel ideas",
            "Mentions artistic, creative, or unconventional perspectives",
            "Shows willingness to explore unfamiliar concepts",
            "Demonstrates intellectual curiosity",
        ],
        "low_indicators": [
            "Focuses on practical, concrete details",
            "Expresses preference for routine or tradition",
            "Dismisses abstract or theoretical ideas",
            "Shows skepticism toward novelty",
            "Emphasizes proven methods over experimentation",
        ],
    },
    "conscientiousness": {
        "description": "Conscientiousness reflects organization, responsibility, and goal-directed behavior.",
        "high_indicators": [
            "Mentions planning, organization, or structure",
            "Expresses commitment to completing tasks",
            "Shows attention to detail",
            "Discusses goals, deadlines, or schedules",
            "Emphasizes reliability and follow-through",
        ],
        "low_indicators": [
            "Expresses spontaneity or flexibility",
            "Mentions procrastination or last-minute approaches",
            "Shows disregard for structure or planning",
            "Emphasizes improvisation over preparation",
            "Downplays importance of deadlines or commitments",
        ],
    },
    "extraversion": {
        "description": "Extraversion reflects sociability, assertiveness, and energy in social situations.",
        "high_indicators": [
            "Expresses enthusiasm for social interaction",
            "Mentions enjoying crowds, parties, or group activities",
            "Shows high energy or excitement",
            "Discusses seeking out social opportunities",
            "Emphasizes being talkative or outgoing",
        ],
        "low_indicators": [
            "Expresses preference for solitude or quiet",
            "Mentions feeling drained by social interaction",
            "Shows reserved or subdued communication style",
            "Discusses need for alone time",
            "Emphasizes listening over speaking",
        ],
    },
    "agreeableness": {
        "description": "Agreeableness reflects compassion, cooperation, and trust in others.",
        "high_indicators": [
            "Expresses empathy or concern for others",
            "Mentions cooperation, harmony, or helping",
            "Shows trusting or forgiving attitude",
            "Discusses avoiding conflict",
            "Emphasizes kindness and consideration",
        ],
        "low_indicators": [
            "Expresses skepticism or distrust of others",
            "Mentions competition or self-interest",
            "Shows confrontational or critical attitude",
            "Discusses prioritizing own needs over others",
            "Emphasizes directness over diplomacy",
        ],
    },
    "neuroticism": {
        "description": "Neuroticism reflects emotional instability, anxiety, and stress reactivity.",
        "high_indicators": [
            "Expresses worry, anxiety, or stress",
            "Mentions emotional volatility or mood swings",
            "Shows sensitivity to criticism or setbacks",
            "Discusses overthinking or catastrophizing",
            "Emphasizes vulnerability or insecurity",
        ],
        "low_indicators": [
            "Expresses calm or emotional stability",
            "Mentions resilience or stress tolerance",
            "Shows even-tempered responses",
            "Discusses confidence and self-assurance",
            "Emphasizes ability to handle pressure",
        ],
    },
}


def get_judge_prompt(text, trait_name):
    """
    Generate LLM-as-Judge prompt for rating text on BFI trait.

    Args:
        text: Generated text to evaluate
        trait_name: BFI trait to assess

    Returns:
        Prompt for GPT-4 or similar judge model
    """
    rubric = BFI_JUDGE_RUBRICS[trait_name]

    high_indicators_str = "\n".join([f"  - {ind}" for ind in rubric["high_indicators"]])
    low_indicators_str = "\n".join([f"  - {ind}" for ind in rubric["low_indicators"]])

    prompt = f"""You are an expert psychologist evaluating text for personality traits using the Big Five Inventory (BFI).

TRAIT: {trait_name.capitalize()}
DESCRIPTION: {rubric["description"]}

HIGH {trait_name.upper()} INDICATORS:
{high_indicators_str}

LOW {trait_name.upper()} INDICATORS:
{low_indicators_str}

TEXT TO EVALUATE:
\"\"\"
{text}
\"\"\"

Rate this text on {trait_name.capitalize()} from 1 to 5:
1 = Very Low {trait_name.capitalize()}
2 = Low {trait_name.capitalize()}
3 = Moderate {trait_name.capitalize()}
4 = High {trait_name.capitalize()}
5 = Very High {trait_name.capitalize()}

Provide your rating in this exact format:
RATING: [1-5]
REASONING: [Brief explanation of your rating based on the indicators]"""

    return prompt


def parse_judge_response(response):
    """
    Parse LLM-as-Judge response to extract rating and reasoning.

    Args:
        response: Raw response from judge model

    Returns:
        Tuple of (rating, reasoning)
    """
    import re

    # Extract rating
    rating_match = re.search(r"RATING:\s*(\d)", response)
    rating = int(rating_match.group(1)) if rating_match else None

    # Extract reasoning
    reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    return rating, reasoning
