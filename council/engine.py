from __future__ import annotations

from typing import List, Dict, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from langfuse.openai import OpenAI
from .types import Personality, DebateConfig, DebateState, Argument, Vote, IterationResult
from .clients import get_ollama_client
from .focus_scorer import batch_score_arguments, generate_focus_report, get_focus_warnings


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
                f"Kamu adalah '{persona.name}'. Traits: {persona.traits}. Perspektif: {persona.perspective}.\n\n"
                f"PENTING - Aturan Ketat Debat:\n"
                f"1. FOKUS MUTLAK pada pertanyaan yang diberikan - jangan melebar ke topik lain\n"
                f"2. Berikan argumen dengan kedalaman penalaran level {reasoning_depth}\n"
                f"3. Maksimal 3-4 poin utama, setiap poin harus RELEVAN dengan pertanyaan\n"
                f"4. Gunakan bukti konkret, data, atau contoh spesifik jika memungkinkan\n"
                f"5. Hindari generalisasi berlebihan - tetap pada scope pertanyaan\n"
                f"6. Nada profesional, ringkas, dan langsung ke inti\n"
                f"7. Jika merespons argumen lain, alamat poin spesifik mereka\n\n"
                f"Truth-seeking level: {persona.truth_seeking} - prioritaskan kebenaran objektif."
            ),
        },
        {
            "role": "user",
            "content": (
                "PERTANYAAN DEBAT: " + question + "\n\n"
                "Argumen sebelumnya:\n" +
                ("\n".join([f"- {a.author}: {a.content}" for a in prior_arguments]) if prior_arguments else "(Argumen pembuka - belum ada argumen sebelumnya)") +
                "\n\n**Berikan argumen Anda sekarang. Tetap fokus pada pertanyaan di atas. Jangan melebar.**"
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
        flat.append(f"\n=== Iterasi {it.iteration} ===")
        for a in it.arguments:
            flat.append(f"[{a.author}]: {a.content}")
        flat.append("\nVoting Iterasi Ini:")
        for v in it.votes:
            flat.append(f"  {v.voter} → {' > '.join(v.ranking[:3])}")
        if it.consensus_reached:
            flat.append(f"  ✓ Konsensus: {it.consensus_candidate}")
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah HAKIM DEBAT profesional dengan kualifikasi tinggi.\n\n"
                "Tugas Anda:\n"
                "1. Analisis OBJEKTIF semua argumen berdasarkan kekuatan logika, bukti, dan relevansi\n"
                "2. Identifikasi argumen terkuat dan terlemah dengan alasan spesifik\n"
                "3. Pertimbangkan hasil voting sebagai indikator persuasivitas\n"
                "4. Berikan keputusan final yang ADIL dan TERUKUR\n"
                "5. Fokus pada KEBENARAN faktual, bukan popularitas\n\n"
                "Format keputusan Anda:\n"
                "1. Ringkasan posisi utama (1-2 kalimat)\n"
                "2. Argumen terkuat dan mengapa (spesifik)\n"
                "3. Kelemahan dalam debat (jika ada)\n"
                "4. Keputusan final atau sintesis\n"
                "5. Rekomendasi tindakan (jika relevan)\n\n"
                "Gunakan bahasa tegas, jelas, dan profesional."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PERTANYAAN DEBAT: {question}\n\n"
                "=== TRANSKRIP LENGKAP DEBAT ===\n" + "\n".join(flat) +
                "\n\n=== BERIKAN KEPUTUSAN FINAL ANDA ==="
            ),
        },
    ]
    return _stream_completion(client=client, model=judge_model, messages=messages, on_chunk=on_chunk)


def _aggregate_ranks(votes: List[Vote]) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for v in votes:
        for idx, name in enumerate(v.ranking):
            scores[name] = scores.get(name, 0) + (idx + 1)  # rank 1 -> 1 point
    return scores


def run_debate(config: DebateConfig, personalities: List[Personality], save_callback=None, elimination: bool = False) -> DebateState:
    state = DebateState(config=config, personalities=personalities)
    client = get_ollama_client()

    # Display debate header
    header_content = (
        f"**Question:** {config.question}\n"
        f"**Participants:** {len(personalities)} agents\n"
        f"**Consensus Threshold:** {config.consensus_threshold:.0%}\n"
        f"**Max Iterations:** {config.max_iterations}\n"
        f"**Elimination Mode:** {'Enabled' if elimination else 'Disabled'}"
    )
    console.print(Panel(header_content, title="🏛️  DEBATE COUNCIL", border_style="bold blue", expand=False))
    console.print()

    # Iterations: opening arguments first
    for i in range(0, config.max_iterations):
        console.rule(f"[bold]Iterasi {i}[/bold]")
        prior_args: List[Argument] = []
        if state.iterations:
            # Prior arguments are from previous iteration for context
            prior_args = state.iterations[-1].arguments

        arguments: List[Argument] = []
        console.print(f"[dim]Debaters: {', '.join([p.name for p in personalities])}[/dim]\n")

        for idx, persona in enumerate(personalities, 1):
            color = _color_for(persona.name)

            # Agent header with metadata
            agent_header = (
                f"[bold {color}]{persona.name}[/bold {color}] "
                f"[dim]({idx}/{len(personalities)})[/dim] "
                f"[dim italic]│ Depth:{persona.reasoning_depth} Truth:{persona.truth_seeking:.2f}[/dim italic]"
            )
            console.print(agent_header)
            console.print(f"[{color}]▸[/{color}] ", end="")

            content = _prompt_for_argument(
                client=client,
                persona=persona,
                question=config.question,
                prior_arguments=prior_args if i > 0 else [],
                reasoning_depth=persona.reasoning_depth,
                on_chunk=lambda chunk, _color=color: console.print(chunk, style=_color, end=""),
            )
            console.print("\n")
            arguments.append(Argument(author=persona.name, content=content, iteration=i))

        # Focus scoring: evaluate how on-topic each argument is
        console.print("\n[dim]Evaluating focus scores...[/dim]")
        argument_pairs = [(arg.author, arg.content) for arg in arguments]
        focus_scores = batch_score_arguments(client, config.question, argument_pairs, threshold=0.65)

        # Display focus warnings if any
        warnings = get_focus_warnings(focus_scores, threshold=0.65)
        if warnings:
            console.print("\n[yellow]Focus Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  {warning}")
        else:
            console.print("[green]✓ Semua argumen fokus dan relevan[/green]")

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

        # Elimination step: drop worst performer by aggregate rank
        if elimination and len(personalities) > 2:
            ranks = _aggregate_ranks(votes)
            # Highest score = worst
            worst = max(ranks.items(), key=lambda kv: kv[1])[0]
            console.print(f"[red]Eliminasi:[/red] {worst}")
            personalities = [p for p in personalities if p.name != worst]

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

    # Render final voting table with enhanced information
    console.print("\n")
    table = Table(title="📊 Hasil Voting Terakhir", border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("Pemilih", style="bold")
    table.add_column("Peringkat Top 3", style="dim")
    table.add_column("First Choice", style="bold green")

    for v in state.iterations[-1].votes:
        voter_color = _color_for(v.voter)
        top3 = " → ".join(v.ranking[:3]) if len(v.ranking) >= 3 else " → ".join(v.ranking)
        first = v.ranking[0] if v.ranking else "N/A"
        table.add_row(
            f"[{voter_color}]{v.voter}[/{voter_color}]",
            top3,
            first
        )

    console.print(table)
    console.print()

    # Show consensus status
    last_iter = state.iterations[-1]
    if last_iter.consensus_reached:
        consensus_panel = Panel(
            f"✅ **Konsensus tercapai!**\n\n"
            f"Kandidat pemenang: **{last_iter.consensus_candidate}**\n"
            f"Tercapai pada iterasi: {last_iter.iteration}",
            title="Consensus Status",
            border_style="bold green"
        )
        console.print(consensus_panel)
    else:
        consensus_panel = Panel(
            f"⚠️  **Konsensus tidak tercapai**\n\n"
            f"Kandidat terdepan: {last_iter.consensus_candidate or 'N/A'}\n"
            f"Maksimal iterasi tercapai: {len(state.iterations)}",
            title="Consensus Status",
            border_style="bold yellow"
        )
        console.print(consensus_panel)

    console.print()
    console.print(Panel(decision, title="⚖️  Keputusan Hakim Final", border_style="bold white"))
    return state


