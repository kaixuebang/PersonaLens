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


# ============================================================
# Behavioral Evaluation (replaces self-report)
# ============================================================

BEHAVIORAL_SCENARIOS_V2 = {
    "openness": [
        "Your team has always used Python for data analysis. A new member suggests switching to a functional programming language you've never heard of, arguing it would make code more elegant but offering no performance gain. Do you support trying it?",
        "A friend offers you free tickets to an experimental avant-garde theater performance where actors perform without dialogue and the audience sits in complete silence. Do you accept?",
        "You have a free weekend. Do you spend it exploring a neighborhood you've never visited, or return to your favorite familiar spot?",
        "Your company offers a sabbatical program where you can study any topic for 6 months at full pay. What subject do you choose, or do you skip it?",
        "A colleague enthusiastically recommends a science fiction novel exploring consciousness and reality. You normally read only biographies. Do you read it?",
        "You're asked to evaluate three competing product designs: one proven bestseller, one radical concept, one experimental prototype. Which do you advocate for?",
        "You discover your neighbor keeps a journal of unusual personal theories and philosophical speculations. Do you ask to borrow it?",
    ],
    "conscientiousness": [
        "You have a major project due in 30 days. Do you create a detailed schedule today, or start when the mood strikes?",
        "You notice a small numerical error in a report that has already been submitted to a client. Nobody else caught it. What do you do?",
        "You're working from home with no oversight. Do you follow a strict schedule, or work when you feel productive?",
        "A colleague asks you to skip a final review step to ship a product on time. The risk is small but real. What do you do?",
        "Your todo list has 15 items. Do you prioritize and complete them systematically, or handle whatever feels most urgent in the moment?",
        "You committed to meeting a weekly gym goal. This week you're busy and unmotivated. Do you go anyway?",
        "Your apartment is messy but guests are coming tomorrow. It's late and you're tired. What do you do?",
        "You find a security vulnerability in your company's code. Reporting it would require hours of extra documentation. What do you do?",
    ],
    "extraversion": [
        "You arrive alone at a large party where you know nobody. Do you introduce yourself to strangers, or find a quiet corner?",
        "Your team needs someone to present the quarterly results to 50 executives. Do you volunteer?",
        "It's Saturday night and friends invite you out, but you're comfortable at home. What do you choose?",
        "You're in a group meeting that's stalled. Do you speak up to break the deadlock, or wait for someone else to?",
        "A friend invites you to join a recreational sports league that meets weekly. You haven't played before. Do you join?",
        "You're at a dinner party seated next to a stranger. Do you start a conversation, or wait for them to speak first?",
        "Your phone rings with a social invitation you're not excited about. Do you go anyway to maintain connections, or decline?",
        "You have the option to work on a team project or independently. Which do you prefer?",
    ],
    "agreeableness": [
        "A colleague takes full credit for your idea during a team meeting. Everyone praises them. What do you do?",
        "You're in a rush and someone cuts in front of you in line at the store. What is your reaction?",
        "A close friend asks to borrow a significant amount of money and has a history of not repaying. What do you do?",
        "Your team made a decision you strongly disagree with, but the vote was democratic. How do you proceed?",
        "A stranger makes a rude comment about your appearance. How do you respond?",
        "Your partner wants to name your child something you find embarrassing. They feel strongly about it. How do you handle this?",
        "A coworker is struggling with their workload and asks for your help during your busiest week. What do you do?",
        "A friend makes an offensive joke at dinner. Others laugh. What is your reaction?",
    ],
    "neuroticism": [
        "You make a minor typo in an email to your entire department. Nobody has replied. What is your reaction over the next hour?",
        "You're waiting for medical test results that will arrive in 3 days. How much does this occupy your thoughts?",
        "A close friend hasn't responded to your message for 6 hours. They usually reply within 30 minutes. What goes through your mind?",
        "You need to give a 10-minute presentation to 20 colleagues next Monday. How do you feel as the weekend approaches?",
        "You hear an unexpected loud thud from downstairs while home alone at midnight. What is your immediate reaction?",
        "You get an email from your bank flagging an unusual transaction that was actually you. How quickly do you check it?",
        "Your boss schedules a 1-on-1 meeting with no agenda specified. What goes through your mind before the meeting?",
        "You see a social media post about an old friend having a serious health issue. How do you react emotionally?",
    ],
}


def score_behavioral_response(response_text: str, trait_name: str) -> dict:
    """
    Score a behavioral response against BFI trait rubric indicators.

    Uses keyword/pattern matching against high/low indicators to compute
    a trait score from generated behavioral text (NOT self-report).

    Args:
        response_text: Generated text response to a behavioral scenario
        trait_name: BFI trait name

    Returns:
        dict with:
            - 'score': float 1-5
            - 'high_matches': list of matched high indicators
            - 'low_matches': list of matched low indicators
    """
    rubric = BFI_JUDGE_RUBRICS[trait_name]
    text_lower = response_text.lower()

    _stop_words = frozenset(
        {
            "their",
            "these",
            "those",
            "about",
            "would",
            "could",
            "should",
            "which",
            "where",
            "there",
            "other",
            "some",
            "than",
            "been",
            "have",
            "that",
            "with",
            "from",
            "into",
            "over",
        }
    )

    high_count = 0
    high_matches = []
    for indicator in rubric["high_indicators"]:
        keywords = [
            w for w in indicator.lower().split() if len(w) > 4 and w not in _stop_words
        ]
        if any(kw in text_lower for kw in keywords):
            high_count += 1
            high_matches.append(indicator)

    low_count = 0
    low_matches = []
    for indicator in rubric["low_indicators"]:
        keywords = [
            w for w in indicator.lower().split() if len(w) > 4 and w not in _stop_words
        ]
        if any(kw in text_lower for kw in keywords):
            low_count += 1
            low_matches.append(indicator)

    total = len(rubric["high_indicators"]) + len(rubric["low_indicators"])
    if total == 0:
        return {"score": 3.0, "high_matches": [], "low_matches": []}

    # Score: high indicators increase, low indicators decrease
    raw = (high_count - low_count) / max(total / 2, 1)  # range roughly -1 to 1
    score = 3.0 + raw * 2.0  # map to 1-5 scale
    score = max(1.0, min(5.0, score))

    return {
        "score": score,
        "high_matches": high_matches,
        "low_matches": low_matches,
    }
