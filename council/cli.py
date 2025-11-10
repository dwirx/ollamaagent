from __future__ import annotations

import os
from typing import Optional, List
from dotenv import load_dotenv
import typer
from rich.console import Console
from .types import DebateConfig
from .personalities import default_personalities
from .engine import run_debate
from .storage import autosave_json

app = typer.Typer(add_completion=False)
console = Console()


@app.command("debate")
def debate(
    question: List[str] = typer.Argument(..., help="Pertanyaan/topik untuk didebatkan"),
    title: Optional[str] = typer.Option(None, "--title", help="Judul debat (optional)"),
    judge_model: str = typer.Option("gemma3:1b", "--judge", help="Model hakim"),
    min_iterations: int = typer.Option(2, "--min-it", help="Minimal iterasi sebelum cek konsensus"),
    max_iterations: int = typer.Option(5, "--max-it", help="Maksimal iterasi"),
    consensus_threshold: float = typer.Option(0.6, "--threshold", help="Ambang konsensus (0-1)"),
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

    config = DebateConfig(
        title=title,
        question=qtext,
        judge_model=judge_model,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        consensus_threshold=consensus_threshold,
    )
    personas = default_personalities()
    run_debate(config=config, personalities=personas, save_callback=autosave_json)


def main():
    app()


if __name__ == "__main__":
    main()


