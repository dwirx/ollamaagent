from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterable, Dict

from langfuse.openai import OpenAI

EMBED_MODEL = "granite-embedding:latest"


@dataclass
class MemoryRecord:
    id: int
    timestamp: datetime
    question: str
    agent: str
    role: str
    phase: str
    content: str
    embedding: Optional[List[float]]


class CouncilMemory:

    def __init__(self, db_path: Path = Path("memory/council_memory.db")) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                agent TEXT NOT NULL,
                role TEXT NOT NULL,
                phase TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def record_episode(
        self,
        *,
        question: str,
        agent: str,
        role: str,
        phase: str,
        content: str,
        embedding: Optional[List[float]] = None,
    ) -> None:
        data = (
            datetime.utcnow().isoformat(),
            question,
            agent,
            role,
            phase,
            content,
            json.dumps(embedding) if embedding is not None else None,
        )
        self.conn.execute(
            """
            INSERT INTO episodes (timestamp, question, agent, role, phase, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        self.conn.commit()

    def fetch_recent(self, limit: int = 5, question: Optional[str] = None) -> List[MemoryRecord]:
        cur = self.conn.cursor()
        if question:
            cur.execute(
                """
                SELECT id, timestamp, question, agent, role, phase, content, embedding
                FROM episodes
                WHERE question = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (question, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, timestamp, question, agent, role, phase, content, embedding
                FROM episodes
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cur.fetchall()
        return [
            MemoryRecord(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                question=row[2],
                agent=row[3],
                role=row[4],
                phase=row[5],
                content=row[6],
                embedding=json.loads(row[7]) if row[7] else None,
            )
            for row in rows
        ]

    def fetch_similar(self, query_embedding: List[float], limit: int = 3) -> List[MemoryRecord]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, question, agent, role, phase, content, embedding
            FROM episodes
            WHERE embedding IS NOT NULL
            """
        )
        scored: List[tuple[float, MemoryRecord]] = []
        for row in cur.fetchall():
            emb = json.loads(row[7])
            score = cosine_similarity(query_embedding, emb)
            scored.append(
                (
                    score,
                    MemoryRecord(
                        id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        question=row[2],
                        agent=row[3],
                        role=row[4],
                        phase=row[5],
                        content=row[6],
                        embedding=emb,
                    ),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:limit]]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if not norm_a or not norm_b:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def embed_text(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def summarize_memory(client: OpenAI, question: str, records: List[MemoryRecord], max_items: int = 5) -> str:
    if not records:
        return "Belum ada memori relevan."
    snippets = []
    for rec in records[:max_items]:
        timestamp = rec.timestamp.strftime("%Y-%m-%d %H:%M")
        snippets.append(f"- [{timestamp}] {rec.agent} ({rec.phase}): {rec.content}")
    prompt = (
        "Ringkas memori berikut untuk memberikan konteks singkat:\n\n"
        + "\n".join(snippets)
        + f"\n\nPertanyaan sekarang: {question}\n"
        "Fokus pada poin kunci, hindari pengulangan."
    )
    messages = [
        {"role": "system", "content": "Anda adalah summarizer yang ringkas dan informatif."},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(model="gemma3:1b", messages=messages)
    return resp.choices[0].message.content.strip()


