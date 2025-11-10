from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal, Callable

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm, IntPrompt

from .types import DebateConfig, Personality, DebateState
from .personalities import default_personalities
from .engine import run_debate
from .consciousness import run_council_of_consciousness, CouncilConfig

console = Console()


def _choose_personalities(all_personas: List[Personality]) -> List[Personality]:
    console.print(Markdown("### Pilih Agen"))
    for i, p in enumerate(all_personas, 1):
        console.print(f"[bold]{i}[/bold]. {p.name} — [dim]{p.model}[/dim] | {p.traits}")
    sel = Prompt.ask("Masukkan nomor agen (mis. 1,3,6) atau 'all'", default="all").strip().lower()
    if sel == "all":
        return all_personas
    idx = []
    for part in sel.replace(" ", "").split(","):
        if part.isdigit():
            n = int(part)
            if 1 <= n <= len(all_personas):
                idx.append(n - 1)
    chosen = [all_personas[i] for i in idx] if idx else all_personas
    console.print(f"Terpilih: {', '.join(p.name for p in chosen)}")
    return chosen


def _choose_consensus() -> Literal["majority", "supermajority", "unanimity"]:
    console.print(Markdown("### Pilih Konsensus"))
    console.print("- majority (>50%)\n- supermajority (>66%)\n- unanimity (100%)")
    value = Prompt.ask("Konsensus", choices=["majority", "supermajority", "unanimity"], default="supermajority")
    return value  # type: ignore


def _mk_markdown_writer(title: Optional[str]) -> Callable[[DebateState], None]:
    out_dir = Path("debates")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = (title or "Debate").replace(" ", "_")[:48]
    md_path = out_dir / f"{ts}_{base}.md"

    # Write header once
    md_path.write_text(f"# {title or 'Debate'}\n\n")

    def writer(state: DebateState) -> None:
        # Append last iteration block or final judge decision
        if state.iterations:
            it = state.iterations[-1]
            with md_path.open("a", encoding="utf-8") as f:
                f.write(f"## Iterasi {it.iteration}\n\n")
                f.write("### Argumen\n")
                for a in it.arguments:
                    f.write(f"- **{a.author}**: {a.content}\n")
                f.write("\n### Voting\n")
                for v in it.votes:
                    f.write(f"- {v.voter}: {' > '.join(v.ranking)}\n")
                if it.consensus_reached:
                    f.write(f"\n> Konsensus sementara pada: **{it.consensus_candidate}**\n\n")
        if state.judge_decision:
            with md_path.open("a", encoding="utf-8") as f:
                f.write("\n## Keputusan Hakim\n\n")
                f.write(state.judge_decision + "\n")

    return writer


def run_interactive() -> None:
    load_dotenv(override=False)
    console.print(Markdown("## Council Interactive Wizard"))

    question = Prompt.ask("Pertanyaan/Topik")
    title = Prompt.ask("Judul (opsional)", default="").strip() or None
    mode = Prompt.ask("Mode (debate/council)", choices=["debate", "council"], default="council")

    if mode == "council":
        eliminate = Confirm.ask("Aktifkan eliminasi agent dalam refleksi?", default=False)
        config = CouncilConfig(question=question, title=title, elimination=eliminate)
        run_council_of_consciousness(config)
        return

    min_it = IntPrompt.ask("Minimal iterasi sebelum cek konsensus", default=2)
    max_it = IntPrompt.ask("Maksimal iterasi", default=5)
    consensus = _choose_consensus()
    eliminate = Confirm.ask("Aktifkan eliminasi agen berkinerja terendah per iterasi?", default=False)

    all_personas = default_personalities()
    chosen_personas = _choose_personalities(all_personas)

    if consensus == "majority":
        threshold = 0.5
    elif consensus == "supermajority":
        threshold = 2.0 / 3.0
    else:
        threshold = 1.0

    config = DebateConfig(
        title=title,
        question=question,
        judge_model="gemma3:1b",
        min_iterations=min_it,
        max_iterations=max_it,
        consensus_threshold=threshold,
    )

    console.print(Markdown("### Mulai Debat"))
    save_md = _mk_markdown_writer(title or "Debate")
    run_debate(config=config, personalities=chosen_personas, save_callback=save_md, elimination=eliminate)


