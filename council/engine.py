from __future__ import annotations

from typing import List, Dict, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langfuse.openai import OpenAI
from .types import Personality, DebateConfig, DebateState, Argument, Vote, IterationResult
from .clients import get_ollama_client


console = Console()

COLOR_PALETTE = [
    "cyan",
    "magenta",
    "green",
    "yellow",
    "blue",
    "bright_cyan",
    "bright_magenta",
    "bright_green",
    "bright_yellow",
    "bright_blue",
]


def _color_for(name: str) -> str:
    idx = abs(hash(name)) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]


def _stream_completion(client: OpenAI, model: str, messages: List[Dict[str, str]], on_chunk: Callable[[str], None]) -> str:
    parts: List[str] = []
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for event in stream:
        try:
            delta = event.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            parts.append(delta)
            on_chunk(delta)
    return "".join(parts).strip()


def _prompt_for_argument(
    client: OpenAI,
    persona: Personality,
    question: str,
    prior_arguments: List[Argument],
    reasoning_depth: int,
    on_chunk: Callable[[str], None],
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                f"Kamu adalah '{persona.name}'. Traits: {persona.traits}. Perspektif: {persona.perspective}. "
                f"Berikan argumen dengan kedalaman penalaran {reasoning_depth}. "
                "Gunakan nada profesional, ringkas, dan berbasis bukti bila memungkinkan."
            ),
        },
        {
            "role": "user",
            "content": (
                "Pertanyaan: " + question + "\n\n"
                "Argumen sebelumnya (jika ada):\n" +
                ("\n".join([f"- {a.author}: {a.content}" for a in prior_arguments]) if prior_arguments else "(belum ada)") +
                "\n\nBerikan argumenmu yang jelas dan terstruktur."
            ),
        },
    ]
    return _stream_completion(client=client, model=persona.model, messages=messages, on_chunk=on_chunk)


def _prompt_for_vote(client: OpenAI, persona: Personality, question: str, arguments: List[Argument]) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": (
                f"Kamu adalah '{persona.name}'. Tugas: lakukan pemeringkatan argumen terbaik->terlemah "
                "berdasarkan kekuatan logika, relevansi, dan dukungan bukti. Kembalikan hanya daftar nama penulis berurut."
            ),
        },
        {
            "role": "user",
            "content": (
                "Pertanyaan: " + question + "\n\n"
                "Argumen:\n" +
                "\n".join([f"- {a.author}: {a.content}" for a in arguments]) +
                "\n\nKembalikan daftar nama penulis, urut terbaik ke terlemah, dipisah koma."
            ),
        },
    ]
    resp = client.chat.completions.create(model=persona.model, messages=messages)
    raw = resp.choices[0].message.content.strip()
    names = [n.strip() for n in raw.replace("\n", ",").split(",") if n.strip()]
    # Keep only valid names in order, deduplicate
    valid = []
    seen = set()
    authors = [a.author for a in arguments]
    for n in names:
        if n in authors and n not in seen:
            valid.append(n)
            seen.add(n)
    # Append missing authors arbitrarily to ensure full ranking
    for a in authors:
        if a not in seen:
            valid.append(a)
    return valid


def _consensus_from_votes(votes: List[Vote], threshold: float) -> tuple[bool, str | None]:
    first_place_counts: Dict[str, int] = {}
    total = len(votes)
    for v in votes:
        if v.ranking:
            first = v.ranking[0]
            first_place_counts[first] = first_place_counts.get(first, 0) + 1
    if not total:
        return False, None
    # Find top candidate
    top_candidate, top_count = None, 0
    for k, cnt in first_place_counts.items():
        if cnt > top_count:
            top_candidate, top_count = k, cnt
    if top_candidate and (top_count / total) >= threshold:
        return True, top_candidate
    return False, top_candidate


def _prompt_for_judge(
    client: OpenAI,
    judge_model: str,
    question: str,
    iterations: List[IterationResult],
    on_chunk: Callable[[str], None],
) -> str:
    flat = []
    for it in iterations:
        flat.append(f"Iterasi {it.iteration}:")
        for a in it.arguments:
            flat.append(f"- {a.author}: {a.content}")
        flat.append("Voting:")
        for v in it.votes:
            flat.append(f"- {v.voter}: {v.ranking}")
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah hakim yang berorientasi kebenaran, ringkas, dan adil. "
                "Sintesis argumen, pertimbangkan voting, dan berikan keputusan final yang menimbang kekuatan bukti."
            ),
        },
        {
            "role": "user",
            "content": (
                "Pertanyaan: " + question + "\n\n"
                "Ringkasan debat:\n" + "\n".join(flat) +
                "\n\nBerikan keputusan final dengan alasan singkat dan poin-poin rekomendasi tindakan jika relevan."
            ),
        },
    ]
    return _stream_completion(client=client, model=judge_model, messages=messages, on_chunk=on_chunk)


def run_debate(config: DebateConfig, personalities: List[Personality], save_callback=None) -> DebateState:
    state = DebateState(config=config, personalities=personalities)
    client = get_ollama_client()

    # Iterations: opening arguments first
    for i in range(0, config.max_iterations):
        console.rule(f"[bold]Iterasi {i}[/bold]")
        prior_args: List[Argument] = []
        if state.iterations:
            # Prior arguments are from previous iteration for context
            prior_args = state.iterations[-1].arguments

        arguments: List[Argument] = []
        for persona in personalities:
            color = _color_for(persona.name)
            console.print(f"[bold {color}]{persona.name}[/bold {color}]: ", end="")
            content = _prompt_for_argument(
                client=client,
                persona=persona,
                question=config.question,
                prior_arguments=prior_args if i > 0 else [],
                reasoning_depth=persona.reasoning_depth,
                on_chunk=lambda chunk, _color=color: console.print(chunk, style=_color, end=""),
            )
            console.print()
            arguments.append(Argument(author=persona.name, content=content, iteration=i))

        votes: List[Vote] = []
        # Voting begins after first arguments are visible
        for persona in personalities:
            ranking = _prompt_for_vote(
                client=client,
                persona=persona,
                question=config.question,
                arguments=arguments,
            )
            votes.append(Vote(voter=persona.name, ranking=ranking, iteration=i))

        consensus, candidate = _consensus_from_votes(votes, threshold=config.consensus_threshold)
        it_result = IterationResult(
            iteration=i,
            arguments=arguments,
            votes=votes,
            consensus_reached=consensus,
            consensus_candidate=candidate,
        )
        state.iterations.append(it_result)

        # Save after each iteration (checkpoint)
        if save_callback:
            save_callback(state)

        # Respect min_iterations before early stop
        if consensus and i + 1 >= config.min_iterations:
            break

    console.print("\n[bold]Hakim menyimpulkan...[/bold]", justify="left")
    decision = _prompt_for_judge(
        client=client,
        judge_model=config.judge_model,
        question=config.question,
        iterations=state.iterations,
        on_chunk=lambda chunk: console.print(chunk, style="bold white", end=""),
    )
    console.print()
    state.judge_decision = decision
    if save_callback:
        save_callback(state)

    # Render final table
    table = Table(title="Hasil Voting Terakhir")
    table.add_column("Pemilih")
    table.add_column("Peringkat")
    for v in state.iterations[-1].votes:
        voter_color = _color_for(v.voter)
        table.add_row(f"[{voter_color}]{v.voter}[/{voter_color}]", " > ".join(v.ranking))
    console.print(table)
    console.print(Panel.fit(decision, title="Keputusan Hakim"))
    return state


