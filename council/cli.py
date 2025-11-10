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
from .rag_system import RAGSystem, RAGConfig
from .enhanced_memory import EnhancedCouncilMemory
from .clients import get_ollama_client

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
    rag: bool = typer.Option(
        False, "--rag/--no-rag", help="Aktifkan RAG (Retrieval Augmented Generation)"
    ),
    rag_use_memory: bool = typer.Option(
        True, "--rag-memory/--no-rag-memory", help="Gunakan debate memory (ChromaDB) untuk RAG"
    ),
    rag_use_docs: bool = typer.Option(
        False, "--rag-docs/--no-rag-docs", help="Gunakan external documents untuk RAG"
    ),
):
    """
    Jalankan council debate dengan beberapa kepribadian model Ollama.
    Hasil akan otomatis disimpan di folder 'debates/' setiap iterasi.

    RAG (Retrieval Augmented Generation):
    - Enhance agent arguments dengan context dari past debates dan documents
    - Gunakan --rag untuk mengaktifkan
    - Gunakan --rag-memory untuk menggunakan ChromaDB memory
    - Gunakan --rag-docs untuk menggunakan external documents dari folder 'docs/'
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

    # Initialize RAG system if enabled
    rag_system = None
    if rag:
        from pathlib import Path

        client = get_ollama_client()
        memory = EnhancedCouncilMemory() if rag_use_memory else None

        rag_config = RAGConfig(
            enabled=True,
            use_memory=rag_use_memory,
            use_external_docs=rag_use_docs,
            external_docs_path=Path("docs") if rag_use_docs else None,
            retrieval_limit=3,
            min_similarity=0.6,
        )

        rag_system = RAGSystem(rag_config, memory, client)

        # Load external docs if enabled
        if rag_use_docs:
            docs_path = Path("docs")
            if docs_path.exists():
                rag_system.load_external_documents(docs_path)
            else:
                console.print("[yellow]Warning: docs/ folder tidak ditemukan untuk RAG documents[/yellow]")

        console.print("[cyan]🧠 RAG System Enabled[/cyan]")
        stats = rag_system.get_rag_stats()
        console.print(f"[dim]  - Memory: {stats['memory_enabled']}, External Docs: {stats['external_docs_count']}[/dim]")

    run_debate(config=config, personalities=personas, save_callback=autosave_json, elimination=elimination, rag_system=rag_system)


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


@app.command("web")
def web_dashboard(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address"),
    port: int = typer.Option(8000, "--port", help="Port number"),
):
    """
    Launch web dashboard for interactive debate management.
    Access at http://localhost:8000
    """
    console.print("[bold cyan]🚀 Starting Council Debate Web Dashboard...[/bold cyan]")
    console.print(f"[green]📡 Server will be available at http://{host}:{port}[/green]")
    console.print("[yellow]Press Ctrl+C to stop the server[/yellow]\n")

    try:
        from web.server import run_server
        run_server(host=host, port=port)
    except ImportError as e:
        console.print(f"[red]Error: Web dependencies not installed. Run: uv sync[/red]")
        console.print(f"[red]Details: {e}[/red]")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()


