from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langfuse.openai import OpenAI

EMBED_MODEL = "granite-embedding:latest"


@dataclass
class MemoryRecord:
    id: str
    timestamp: datetime
    question: str
    agent: str
    role: str
    phase: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_chroma(cls, doc_id: str, document: str, metadata: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=doc_id,
            timestamp=datetime.fromisoformat(metadata['timestamp']),
            question=metadata['question'],
            agent=metadata['agent'],
            role=metadata['role'],
            phase=metadata['phase'],
            content=document,
            metadata={k: v for k, v in metadata.items() if k not in ['timestamp', 'question', 'agent', 'role', 'phase']}
        )


class ChromaCouncilMemory:
    """
    Advanced vector memory system using ChromaDB with semantic search capabilities.
    Provides better retrieval than SQLite+cosine by leveraging ChromaDB's optimized vector indexing.
    """

    def __init__(
        self,
        collection_name: str = "council_memory",
        persist_directory: Path = Path("memory/chroma_db"),
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        self._doc_counter = 0

    def close(self) -> None:
        """Close connection (ChromaDB auto-persists)"""
        pass

    def record_episode(
        self,
        *,
        question: str,
        agent: str,
        role: str,
        phase: str,
        content: str,
        embedding: Optional[List[float]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record an episode with semantic embedding.

        Args:
            question: The main question/topic
            agent: Name of the agent
            role: Role of the agent
            phase: Phase of deliberation
            content: The actual content/contribution
            embedding: Pre-computed embedding vector
            extra_metadata: Additional metadata to store

        Returns:
            Document ID
        """
        if embedding is None:
            raise ValueError("Embedding is required for ChromaDB storage")

        # Generate unique ID
        doc_id = f"{role}_{phase}_{datetime.utcnow().timestamp()}_{self._doc_counter}"
        self._doc_counter += 1

        # Prepare metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "agent": agent,
            "role": role,
            "phase": phase,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        # Add to ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

        return doc_id

    def fetch_recent(
        self,
        limit: int = 5,
        question: Optional[str] = None,
        role: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """
        Fetch recent episodes, optionally filtered by question/role/phase.

        Note: ChromaDB doesn't have native sorting by timestamp, so we fetch more
        and sort in memory.
        """
        where_filter = {}
        if question:
            where_filter["question"] = question
        if role:
            where_filter["role"] = role
        if phase:
            where_filter["phase"] = phase

        try:
            results = self.collection.get(
                where=where_filter if where_filter else None,
                limit=limit * 3,  # Fetch more to sort
            )
        except Exception:
            # If collection is empty
            return []

        if not results['ids']:
            return []

        # Convert to MemoryRecords
        records = []
        for i in range(len(results['ids'])):
            record = MemoryRecord.from_chroma(
                doc_id=results['ids'][i],
                document=results['documents'][i],
                metadata=results['metadatas'][i],
            )
            records.append(record)

        # Sort by timestamp descending
        records.sort(key=lambda r: r.timestamp, reverse=True)

        return records[:limit]

    def fetch_similar(
        self,
        query_embedding: List[float],
        limit: int = 3,
        question: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[tuple[float, MemoryRecord]]:
        """
        Semantic similarity search using vector embeddings.

        Args:
            query_embedding: Query vector
            limit: Max number of results
            question: Optional filter by question
            min_similarity: Minimum cosine similarity threshold (0-1)

        Returns:
            List of (similarity_score, MemoryRecord) tuples, sorted by relevance
        """
        where_filter = {"question": question} if question else None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter,
            )
        except Exception:
            return []

        if not results['ids'] or not results['ids'][0]:
            return []

        # Convert to MemoryRecords with scores
        scored_records = []
        for i in range(len(results['ids'][0])):
            # ChromaDB returns distances; convert to similarity
            # For cosine distance: similarity = 1 - distance
            distance = results['distances'][0][i]
            similarity = 1.0 - distance

            if similarity < min_similarity:
                continue

            record = MemoryRecord.from_chroma(
                doc_id=results['ids'][0][i],
                document=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
            )
            scored_records.append((similarity, record))

        return scored_records

    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 10,
    ) -> List[MemoryRecord]:
        """
        Search by metadata fields (agent, role, phase, etc.)
        """
        try:
            results = self.collection.get(
                where=filters,
                limit=limit,
            )
        except Exception:
            return []

        if not results['ids']:
            return []

        records = []
        for i in range(len(results['ids'])):
            record = MemoryRecord.from_chroma(
                doc_id=results['ids'][i],
                document=results['documents'][i],
                metadata=results['metadatas'][i],
            )
            records.append(record)

        return records

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory collection"""
        count = self.collection.count()
        return {
            "total_memories": count,
            "collection_name": self.collection.name,
        }

    def clear_collection(self) -> None:
        """Clear all memories (use with caution!)"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )


def embed_text(client: OpenAI, text: str) -> List[float]:
    """Generate embedding for text using Ollama embedding model"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def summarize_memory(
    client: OpenAI,
    question: str,
    records: List[MemoryRecord],
    scored_records: Optional[List[tuple[float, MemoryRecord]]] = None,
    max_items: int = 5,
) -> str:
    """
    Summarize memory records to provide context.

    Args:
        client: OpenAI client
        question: Current question
        records: List of recent records
        scored_records: List of (score, record) from similarity search
        max_items: Max items to include in summary
    """
    if not records and not scored_records:
        return "Belum ada memori relevan."

    snippets = []

    # Add scored records first (most relevant)
    if scored_records:
        snippets.append("**Memori paling relevan (by semantic similarity):**")
        for score, rec in scored_records[:max_items]:
            timestamp = rec.timestamp.strftime("%Y-%m-%d %H:%M")
            snippets.append(
                f"- [{timestamp}] {rec.agent} ({rec.phase}) [relevance: {score:.2f}]: {rec.content[:150]}..."
            )

    # Add recent records
    if records:
        snippets.append("\n**Memori terkini:**")
        for rec in records[:max_items]:
            timestamp = rec.timestamp.strftime("%Y-%m-%d %H:%M")
            snippets.append(f"- [{timestamp}] {rec.agent} ({rec.phase}): {rec.content[:150]}...")

    prompt = (
        "Ringkas memori berikut untuk memberikan konteks singkat:\n\n"
        + "\n".join(snippets)
        + f"\n\n**Pertanyaan sekarang:** {question}\n\n"
        "Tugas: Ekstrak 3-5 poin kunci yang paling relevan untuk pertanyaan sekarang. "
        "Fokus pada insights, patterns, dan lessons learned. Hindari pengulangan."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah memory synthesizer yang ringkas dan tajam. "
                "Ekstrak HANYA insights yang relevan dengan pertanyaan baru. "
                "Format: bullet points, maksimal 5 poin."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        resp = client.chat.completions.create(model="gemma3:1b", messages=messages)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing memory: {str(e)}"
