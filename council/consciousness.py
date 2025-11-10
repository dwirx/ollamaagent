from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from langfuse.openai import OpenAI

from .clients import get_ollama_client
from .memory import CouncilMemory, embed_text, summarize_memory
from .roles import CouncilRole, council_of_consciousness_roles

console = Console()


@dataclass
class CouncilConfig:
    question: str
    title: Optional[str] = None
    elimination: bool = False


def _ensure_log_file(title: Optional[str]) -> Path:
    out_dir = Path("debates")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = (title or "Council").replace(" ", "_")[:64]
    path = out_dir / f"{ts}_{base}_council.md"
    path.write_text(f"# {title or 'Council of Consciousness'}\n\n")
    return path


def _append_markdown(path: Path, content: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(content)


def _stream_role_output(
    client: OpenAI,
    role: CouncilRole,
    question: str,
    summary: str,
    phase: str,
    previous: List[Dict[str, str]],
    on_chunk: Callable[[str], None],
) -> str:
    system_prompt = (
        f"Kamu adalah {role.title}. "
        f"Arketipe: {role.archetype} "
        f"Perspektif: {role.perspective} "
        f"Gaya bicara: {role.signature} "
        f"Fase: {phase}. Jawab ringkas, terstruktur, reflektif. "
        "Gunakan bahasa Indonesia baku dengan nada sesuai peran."
    )
    memory_snippet = (
        f"Konteks memori kolektif terkini:\n{summary}\n\n"
        if summary
        else ""
    )
    previous_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in previous) if previous else ""
    user_prompt = (
        f"{memory_snippet}"
        f"Pertanyaan utama: {question}\n\n"
        f"Riwayat singkat:\n{previous_text}\n\n"
        "Berikan kontribusi yang memperdalam pemahaman kolektif."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    parts: List[str] = []
    stream = client.chat.completions.create(model=role.model, messages=messages, stream=True)
    for event in stream:
        try:
            delta = event.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            parts.append(delta)
            on_chunk(delta)
    return "".join(parts).strip()


def _choose_elimination(
    client: OpenAI,
    question: str,
    contributions: Dict[str, str],
    reflections: Dict[str, str],
) -> Optional[str]:
    prompt_lines = [
        f"Pertanyaan: {question}",
        "Kontribusi utama:",
    ]
    for role, text in contributions.items():
        prompt_lines.append(f"- {role}: {text}")
    prompt_lines.append("\nRefleksi:")
    for role, text in reflections.items():
        prompt_lines.append(f"- {role}: {text}")
    prompt_lines.append(
        "\nPilih satu peran yang kontribusinya paling lemah atau tidak relevan. "
        "Jika semua layak dipertahankan tulis 'None'. Jawab hanya nama peran."
    )
    messages = [
        {"role": "system", "content": "Anda evaluator netral. Jawab sangat singkat."},
        {"role": "user", "content": "\n".join(prompt_lines)},
    ]
    resp = client.chat.completions.create(model="gemma3:1b", messages=messages)
    result = resp.choices[0].message.content.strip()
    if result.lower() == "none":
        return None
    return result


def run_council_of_consciousness(config: CouncilConfig) -> None:
    client = get_ollama_client()
    memory = CouncilMemory()
    roles = council_of_consciousness_roles()
    moderator = roles[0]
    speakers = [r for r in roles if r.key not in {"moderator", "critic"}]
    critic = next(r for r in roles if r.key == "critic")

    log_path = _ensure_log_file(config.title or config.question)
    _append_markdown(log_path, f"**Pertanyaan:** {config.question}\n\n")

    total_phases = 5 + (1 if config.elimination else 0)
    phase_index = 1

    def announce_phase(name: str, heading: str) -> None:
        nonlocal phase_index
        console.rule(f"[bold cyan]Fase {phase_index}/{total_phases}: {name}[/bold cyan]")
        console.print(Markdown(f"### {name}"))
        _append_markdown(log_path, f"\n## {heading}\n\n")
        phase_index += 1

    # Retrieve context
    similar_records = []
    try:
        query_emb = embed_text(client, config.question)
        similar_records = memory.fetch_similar(query_emb, limit=3)
    except Exception:
        similar_records = []
    summary = ""
    try:
        recent = memory.fetch_recent(limit=5)
        summary = summarize_memory(client, config.question, recent + similar_records)
    except Exception:
        summary = ""

    announce_phase("Pembukaan Moderator", "Pembukaan Moderator")
    console.print(f"[bold]{moderator.title}[/bold]: ", end="")
    prev_messages: List[Dict[str, str]] = []
    moderator_opening = _stream_role_output(
        client,
        moderator,
        config.question,
        summary,
        phase="Pembukaan Moderator",
        previous=[],
        on_chunk=lambda chunk: console.print(chunk, style=moderator.color, end=""),
    )
    console.print()
    prev_messages.append({"role": "moderator", "content": moderator_opening})
    _append_markdown(log_path, moderator_opening + "\n\n")
    try:
        emb = embed_text(client, moderator_opening)
    except Exception:
        emb = None
    memory.record_episode(
        question=config.question,
        agent=moderator.title,
        role=moderator.key,
        phase="opening",
        content=moderator_opening,
        embedding=emb,
    )

    announce_phase("Putaran Argumen", "Putaran Argumen")
    contributions: Dict[str, str] = {}
    for role in speakers:
        console.print(f"[bold {role.color}]{role.title}[/bold {role.color}]: ", end="")
        content = _stream_role_output(
            client,
            role,
            config.question,
            summary,
            phase="Argumen Awal",
            previous=prev_messages,
            on_chunk=lambda chunk, color=role.color: console.print(chunk, style=color, end=""),
        )
        console.print()
        prev_messages.append({"role": role.key, "content": content})
        _append_markdown(log_path, f"### {role.title}\n\n{content}\n\n")
        contributions[role.title] = content
        try:
            emb = embed_text(client, content)
        except Exception:
            emb = None
        memory.record_episode(
            question=config.question,
            agent=role.title,
            role=role.key,
            phase="argument",
            content=content,
            embedding=emb,
        )

    announce_phase("Sesi Kritik & Sanggahan", "Sesi Kritik & Sanggahan")
    console.print(f"[bold {critic.color}]{critic.title}[/bold {critic.color}]: ", end="")
    critic_content = _stream_role_output(
        client,
        critic,
        config.question,
        summary,
        phase="Analisis & Kritik",
        previous=prev_messages,
        on_chunk=lambda chunk, color=critic.color: console.print(chunk, style=color, end=""),
    )
    console.print()
    prev_messages.append({"role": critic.key, "content": critic_content})
    _append_markdown(log_path, critic_content + "\n\n")
    try:
        emb = embed_text(client, critic_content)
    except Exception:
        emb = None
    memory.record_episode(
        question=config.question,
        agent=critic.title,
        role=critic.key,
        phase="critique",
        content=critic_content,
        embedding=emb,
    )

    announce_phase("Refleksi Kolektif", "Refleksi Kolektif")
    reflections: Dict[str, str] = {}
    for role in speakers:
        console.print(f"[bold {role.color}]{role.title}[/bold {role.color}] refleksi: ", end="")
        content = _stream_role_output(
            client,
            role,
            config.question,
            summary,
            phase="Refleksi",
            previous=prev_messages,
            on_chunk=lambda chunk, color=role.color: console.print(chunk, style=color, end=""),
        )
        console.print()
        reflections[role.title] = content
        prev_messages.append({"role": role.key, "content": content})
        _append_markdown(log_path, f"- **{role.title}:** {content}\n")
        try:
            emb = embed_text(client, content)
        except Exception:
            emb = None
        memory.record_episode(
            question=config.question,
            agent=role.title,
            role=role.key,
            phase="reflection",
            content=content,
            embedding=emb,
        )

    announce_phase("Penutupan Moderator", "Penutupan Moderator")
    console.print(f"[bold]{moderator.title}[/bold]: ", end="")
    closing = _stream_role_output(
        client,
        moderator,
        config.question,
        summary,
        phase="Penutupan",
        previous=prev_messages,
        on_chunk=lambda chunk: console.print(chunk, style=moderator.color, end=""),
    )
    console.print()
    _append_markdown(log_path, closing + "\n")
    try:
        emb = embed_text(client, closing)
    except Exception:
        emb = None
    memory.record_episode(
        question=config.question,
        agent=moderator.title,
        role=moderator.key,
        phase="closing",
        content=closing,
        embedding=emb,
    )

    if config.elimination:
        announce_phase("Evaluasi Eliminasi", "Evaluasi Eliminasi")
        elimination_target = _choose_elimination(client, config.question, contributions, reflections)
        if elimination_target:
            console.print(Markdown(f"**Eliminasi yang disarankan:** {elimination_target}"))
            _append_markdown(log_path, f"\n**Eliminasi yang disarankan:** {elimination_target}\n")
            memory.record_episode(
                question=config.question,
                agent="Evaluator",
                role="eliminator",
                phase="elimination",
                content=f"Eliminasi yang disarankan: {elimination_target}",
                embedding=None,
            )

    memory.close()
    console.print(Markdown(f"\n**Log Markdown:** `{log_path}`"))


