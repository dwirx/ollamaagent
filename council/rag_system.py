from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from langfuse.openai import OpenAI
from .enhanced_memory import EnhancedCouncilMemory, embed_text


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    enabled: bool = False
    use_memory: bool = True  # Use debate memory
    use_external_docs: bool = False  # Use uploaded documents
    external_docs_path: Optional[Path] = None
    retrieval_limit: int = 3  # Number of relevant memories to retrieve
    min_similarity: float = 0.6  # Minimum similarity threshold


class RAGSystem:
    """
    Retrieval Augmented Generation System

    Enhances agent arguments with relevant context from:
    1. Past debate memories (ChromaDB)
    2. External documents (optional)
    3. Web search results (future)
    """

    def __init__(
        self,
        config: RAGConfig,
        memory: Optional[EnhancedCouncilMemory] = None,
        client: Optional[OpenAI] = None,
    ):
        self.config = config
        self.memory = memory or EnhancedCouncilMemory()
        self.client = client
        self.external_docs_index: Dict[str, str] = {}  # doc_id -> content

    def load_external_documents(self, docs_path: Path) -> int:
        """
        Load external documents for RAG

        Supports: .txt, .md, .json files

        Args:
            docs_path: Path to documents directory

        Returns:
            Number of documents loaded
        """
        if not docs_path.exists():
            print(f"Warning: Documents path {docs_path} does not exist")
            return 0

        count = 0

        # Load text files
        for file_path in docs_path.glob("*.txt"):
            try:
                content = file_path.read_text(encoding='utf-8')
                self.external_docs_index[file_path.stem] = content
                count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Load markdown files
        for file_path in docs_path.glob("*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                self.external_docs_index[file_path.stem] = content
                count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Load JSON files
        for file_path in docs_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert JSON to text
                    content = json.dumps(data, indent=2)
                    self.external_docs_index[file_path.stem] = content
                count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Loaded {count} external documents for RAG")
        return count

    def retrieve_context(
        self,
        question: str,
        agent_name: str,
        current_iteration: int = 0,
    ) -> str:
        """
        Retrieve relevant context for an agent's argument

        Args:
            question: Main debate question
            agent_name: Name of the agent
            current_iteration: Current debate iteration

        Returns:
            Formatted context string to augment prompt
        """
        if not self.config.enabled:
            return ""

        context_parts = []

        # 1. Retrieve from memory
        if self.config.use_memory and self.memory and self.client:
            try:
                query_embedding = embed_text(self.client, question)

                similar_memories = self.memory.fetch_similar_with_decay(
                    query_embedding=query_embedding,
                    limit=self.config.retrieval_limit,
                    min_similarity=self.config.min_similarity,
                )

                if similar_memories:
                    context_parts.append("=== KONTEKS DARI DEBAT SEBELUMNYA ===")
                    for score, mem in similar_memories:
                        context_parts.append(
                            f"\n[{mem.agent}] ({mem.timestamp.strftime('%Y-%m-%d')}): "
                            f"{mem.content[:300]}..."
                        )
                        context_parts.append(f"  → Relevance: {score:.2f}")

            except Exception as e:
                print(f"Error retrieving memory context: {e}")

        # 2. Retrieve from external documents
        if self.config.use_external_docs and self.external_docs_index:
            context_parts.append("\n=== DOKUMEN REFERENSI ===")

            # Simple keyword matching for now
            # In production, use embeddings for all docs too
            question_keywords = set(question.lower().split())

            relevant_docs = []
            for doc_id, content in self.external_docs_index.items():
                doc_keywords = set(content.lower().split())
                overlap = len(question_keywords.intersection(doc_keywords))

                if overlap >= 2:  # At least 2 keywords match
                    relevant_docs.append((doc_id, content, overlap))

            # Sort by relevance
            relevant_docs.sort(key=lambda x: x[2], reverse=True)

            # Take top 2 docs
            for doc_id, content, score in relevant_docs[:2]:
                context_parts.append(f"\n[Dokumen: {doc_id}]")
                context_parts.append(content[:500] + "...")

        if not context_parts:
            return ""

        # Format context
        context = "\n".join(context_parts)

        return f"""
╔═══════════════════════════════════════════════════════════════╗
║  RAG CONTEXT AUGMENTATION (untuk {agent_name})                ║
╚═══════════════════════════════════════════════════════════════╝

{context}

╔═══════════════════════════════════════════════════════════════╗
║  INSTRUKSI: Gunakan konteks di atas untuk memperkuat argumen  ║
║  Anda. Cite sumber bila relevan. Jangan copy verbatim.        ║
╚═══════════════════════════════════════════════════════════════╝
"""

    def enhance_prompt_with_rag(
        self,
        base_prompt: str,
        question: str,
        agent_name: str,
        iteration: int = 0,
    ) -> str:
        """
        Enhance agent prompt with RAG context

        Args:
            base_prompt: Original system prompt
            question: Debate question
            agent_name: Agent name
            iteration: Current iteration

        Returns:
            Enhanced prompt with RAG context
        """
        if not self.config.enabled:
            return base_prompt

        context = self.retrieve_context(question, agent_name, iteration)

        if not context:
            return base_prompt

        # Insert context after base prompt
        enhanced = f"""{base_prompt}

{context}

CATATAN RAG:
- Konteks di atas dari retrieval system (past debates + documents)
- Gunakan untuk memperkaya argumen Anda dengan precedents dan data
- SELALU verify informasi sebelum menggunakan
- Cite sumber: "Berdasarkan debat sebelumnya..." atau "Menurut dokumen X..."
"""

        return enhanced

    def add_document_inline(self, doc_id: str, content: str) -> None:
        """
        Add a document to RAG index at runtime

        Args:
            doc_id: Document identifier
            content: Document content
        """
        self.external_docs_index[doc_id] = content
        print(f"Added document '{doc_id}' to RAG index")

    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "enabled": self.config.enabled,
            "memory_enabled": self.config.use_memory,
            "external_docs_enabled": self.config.use_external_docs,
            "external_docs_count": len(self.external_docs_index),
            "retrieval_limit": self.config.retrieval_limit,
            "min_similarity": self.config.min_similarity,
        }


def create_rag_system(
    enabled: bool = True,
    use_memory: bool = True,
    external_docs_path: Optional[Path] = None,
    client: Optional[OpenAI] = None,
) -> RAGSystem:
    """
    Factory function to create RAG system

    Args:
        enabled: Enable RAG
        use_memory: Use debate memory
        external_docs_path: Path to external documents
        client: OpenAI client for embeddings

    Returns:
        Configured RAGSystem
    """
    config = RAGConfig(
        enabled=enabled,
        use_memory=use_memory,
        use_external_docs=external_docs_path is not None,
        external_docs_path=external_docs_path,
        retrieval_limit=3,
        min_similarity=0.6,
    )

    memory = EnhancedCouncilMemory() if use_memory else None
    rag = RAGSystem(config, memory, client)

    if external_docs_path:
        rag.load_external_documents(external_docs_path)

    return rag
