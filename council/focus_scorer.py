from __future__ import annotations

from typing import List, Dict
from dataclasses import dataclass
from langfuse.openai import OpenAI


@dataclass
class FocusScore:
    """Score indicating how well an argument stays on-topic"""
    score: float  # 0.0 to 1.0, higher = more focused
    reasoning: str
    is_focused: bool  # True if score >= threshold


def score_argument_focus(
    client: OpenAI,
    question: str,
    argument: str,
    author: str,
    threshold: float = 0.7,
) -> FocusScore:
    """
    Evaluate how well an argument stays focused on the question.

    Args:
        client: OpenAI client
        question: The debate question
        argument: The argument to score
        author: Name of the argument author
        threshold: Minimum score to be considered "focused"

    Returns:
        FocusScore with score, reasoning, and boolean focus flag
    """
    system_prompt = (
        "Kamu adalah FOCUS EVALUATOR yang menilai apakah sebuah argumen tetap ON-TOPIC.\n\n"
        "Tugas: Evaluasi seberapa relevan dan fokus argumen terhadap pertanyaan.\n\n"
        "Kriteria penilaian:\n"
        "1.0 = Sempurna fokus, semua poin alamat pertanyaan langsung\n"
        "0.8-0.9 = Sebagian besar relevan, ada 1-2 tangent minor\n"
        "0.6-0.7 = Lumayan fokus, tapi ada digresi yang mengurangi impact\n"
        "0.4-0.5 = Setengah topik, setengah melebar ke hal lain\n"
        "0.0-0.3 = Kebanyakan off-topic, tidak alamat pertanyaan inti\n\n"
        "Berikan:\n"
        "1. Score (0.0 - 1.0)\n"
        "2. Reasoning singkat (1-2 kalimat)"
    )

    user_prompt = (
        f"PERTANYAAN DEBAT:\n{question}\n\n"
        f"ARGUMEN oleh {author}:\n{argument}\n\n"
        "Evaluasi focus score-nya. Format jawaban:\n"
        "SCORE: [angka 0.0-1.0]\n"
        "REASONING: [penjelasan singkat]"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = client.chat.completions.create(
            model="gemma3:1b",
            messages=messages,
            temperature=0.3,  # Lower temp for more consistent scoring
        )
        response_text = resp.choices[0].message.content.strip()

        # Parse response
        score = 0.5  # Default
        reasoning = response_text

        lines = response_text.split("\n")
        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip()
                    score = float(score_text)
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return FocusScore(
            score=score,
            reasoning=reasoning,
            is_focused=score >= threshold,
        )

    except Exception as e:
        return FocusScore(
            score=0.5,
            reasoning=f"Error evaluating focus: {str(e)}",
            is_focused=False,
        )


def batch_score_arguments(
    client: OpenAI,
    question: str,
    arguments: List[tuple[str, str]],  # List of (author, argument)
    threshold: float = 0.7,
) -> Dict[str, FocusScore]:
    """
    Score multiple arguments for focus.

    Args:
        client: OpenAI client
        question: The debate question
        arguments: List of (author, argument) tuples
        threshold: Minimum score to be considered "focused"

    Returns:
        Dict mapping author -> FocusScore
    """
    results = {}
    for author, argument in arguments:
        score = score_argument_focus(client, question, argument, author, threshold)
        results[author] = score

    return results


def generate_focus_report(
    question: str,
    scores: Dict[str, FocusScore],
) -> str:
    """
    Generate a formatted report of focus scores.

    Args:
        question: The debate question
        scores: Dict mapping author -> FocusScore

    Returns:
        Formatted markdown report
    """
    lines = [
        "## Focus Score Report",
        f"**Pertanyaan:** {question}\n",
        "| Pembicara | Score | Status | Reasoning |",
        "|-----------|-------|--------|-----------|",
    ]

    sorted_scores = sorted(scores.items(), key=lambda x: x[1].score, reverse=True)

    for author, score in sorted_scores:
        status = "✓ Fokus" if score.is_focused else "⚠ Melebar"
        lines.append(
            f"| {author} | {score.score:.2f} | {status} | {score.reasoning} |"
        )

    avg_score = sum(s.score for s in scores.values()) / len(scores) if scores else 0
    focused_count = sum(1 for s in scores.values() if s.is_focused)
    total_count = len(scores)

    lines.extend([
        "",
        f"**Summary:** {focused_count}/{total_count} argumen fokus. Rata-rata score: {avg_score:.2f}",
    ])

    return "\n".join(lines)


def get_focus_warnings(
    scores: Dict[str, FocusScore],
    threshold: float = 0.7,
) -> List[str]:
    """
    Generate warnings for arguments that are off-topic.

    Args:
        scores: Dict mapping author -> FocusScore
        threshold: Threshold for focus

    Returns:
        List of warning messages
    """
    warnings = []

    for author, score in scores.items():
        if not score.is_focused:
            warnings.append(
                f"⚠ {author}: Argumen melebar (score {score.score:.2f}). {score.reasoning}"
            )

    return warnings
