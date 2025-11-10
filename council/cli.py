from __future__ import annotations

import os
from typing import Optional, List, Literal
from dotenv import load_dotenv
import typer
from rich.console import Console
from .types import DebateConfig
from .personalities import default_personalities
from .engine import run_debate
from .storage import autosave_json
from .interactive import run_interactive
from .consciousness import run_council_of_consciousness, CouncilConfig

app = typer.Typer(add_completion=False)
console = Console()


@app.command("debate")
def debate(
    question: List[str] = typer.Argument(..., help="Pertanyaan/topik untuk didebatkan"),
    title: Optional[str] = typer.Option(None, "--title", help="Judul debat (optional)"),
    judge_model: str = typer.Option("kimi-k2:1t-cloud", "--judge", help="Model hakim"),
    min_iterations: int = typer.Option(2, "--min-it", help="Minimal iterasi sebelum cek konsensus"),
    max_iterations: int = typer.Option(5, "--max-it", help="Maksimal iterasi"),
    consensus: Literal["majority", "supermajority", "unanimity"] = typer.Option(
        "supermajority",
        "--consensus",
        help="Preset ambang konsensus: majority(>50%), supermajority(>66%), unanimity(100%)",
    ),
    elimination: bool = typer.Option(
        False, "--eliminate/--no-eliminate", help="Aktifkan eliminasi agen per iterasi (opsional)"
    ),
):
    """
    Jalankan council debate dengan beberapa kepribadian model Ollama.
    Hasil akan otomatis disimpan di folder 'debates/' setiap iterasi.
    """
    load_dotenv(override=False)
    # Ensure Langfuse env present; if missing, user gets clearer error from main example
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        if not os.getenv(var):
            console.print(f"[yellow]Peringatan: {var} belum di-set. Tracing Langfuse mungkin tidak aktif.[/yellow]")

    qtext = " ".join(question).strip()
    if not qtext:
        console.print("[red]Pertanyaan kosong.[/red]")
        raise typer.Exit(code=2)

    if consensus == "majority":
        threshold = 0.5
    elif consensus == "supermajority":
        threshold = 2.0 / 3.0
    else:
        threshold = 1.0

    config = DebateConfig(
        title=title,
        question=qtext,
        judge_model=judge_model,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        consensus_threshold=threshold,
    )
    personas = default_personalities()
    run_debate(config=config, personalities=personas, save_callback=autosave_json, elimination=elimination)


@app.command("interactive")
def interactive():
    """
    Wizard interaktif di terminal dengan pilihan agen, konsensus, eliminasi,
    dan log proses dalam Markdown secara realtime.
    """
    run_interactive()


@app.command("consciousness")
def consciousness(
    question: Optional[str] = typer.Option(None, "--question", help="Pertanyaan/topik"),
    title: Optional[str] = typer.Option(None, "--title", help="Judul debat (opsional)"),
    elimination: bool = typer.Option(False, "--eliminate/--no-eliminate", help="Eliminasi refleksi"),
):
    """
    Jalankan Council of Consciousness: moderator, filosof, humanis, kritikus, spiritualis, teknokrat.
    Berjalan dengan memori episodik & log Markdown otomatis.
    """
    load_dotenv(override=False)
    if not question:
        question = typer.prompt("Pertanyaan/Topik")
    config = CouncilConfig(question=question, title=title, elimination=elimination)
    run_council_of_consciousness(config)


def main():
    app()


if __name__ == "__main__":
    main()


