from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from .chroma_memory import ChromaCouncilMemory, MemoryRecord, embed_text
from langfuse.openai import OpenAI


@dataclass
class EnhancedMemoryRecord(MemoryRecord):
    """Memory record with additional metadata"""
    tags: Set[str]
    category: str
    importance: float  # 0.0 to 1.0
    access_count: int
    last_accessed: datetime
    decay_factor: float  # Current decay multiplier


class EnhancedCouncilMemory(ChromaCouncilMemory):
    """
    Enhanced memory system with:
    - Tags and categories
    - Memory decay (older memories have less weight)
    - Access tracking
    - Learning insights across debates
    - Export/import capabilities
    """

    def __init__(
        self,
        collection_name: str = "enhanced_council_memory",
        persist_directory: Path = Path("memory/enhanced_chroma"),
        decay_rate: float = 0.1,  # Decay rate per day
    ):
        super().__init__(collection_name, persist_directory)
        self.decay_rate = decay_rate

    def record_episode(
        self,
        *,
        question: str,
        agent: str,
        role: str,
        phase: str,
        content: str,
        embedding: Optional[List[float]] = None,
        tags: Optional[Set[str]] = None,
        category: str = "general",
        importance: float = 0.5,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record episode with enhanced metadata

        Args:
            question: Main question/topic
            agent: Agent name
            role: Role of agent
            phase: Phase of deliberation
            content: Content/contribution
            embedding: Embedding vector
            tags: Set of tags (e.g., {"ethics", "technical", "policy"})
            category: Category (e.g., "opening", "rebuttal", "synthesis")
            importance: Importance score 0.0-1.0
            extra_metadata: Additional metadata

        Returns:
            Document ID
        """
        if embedding is None:
            raise ValueError("Embedding required for enhanced memory")

        # Prepare enhanced metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "agent": agent,
            "role": role,
            "phase": phase,
            "tags": json.dumps(list(tags or set())),
            "category": category,
            "importance": importance,
            "access_count": 0,
            "last_accessed": datetime.utcnow().isoformat(),
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        # Generate unique ID
        doc_id = f"{role}_{phase}_{category}_{datetime.utcnow().timestamp()}_{self._doc_counter}"
        self._doc_counter += 1

        # Add to ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

        return doc_id

    def fetch_similar_with_decay(
        self,
        query_embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.5,
        question: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[tuple[float, EnhancedMemoryRecord]]:
        """
        Semantic search with memory decay applied

        Older memories have reduced weight based on decay_rate
        """
        where_filter = {}
        if question:
            where_filter["question"] = question

        try:
            # Fetch more results to apply decay filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 3,
                where=where_filter if where_filter else None,
            )
        except Exception:
            return []

        if not results['ids'] or not results['ids'][0]:
            return []

        # Process results with decay
        scored_records = []
        now = datetime.utcnow()

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            timestamp = datetime.fromisoformat(metadata['timestamp'])

            # Calculate decay
            age_days = (now - timestamp).days
            decay_factor = math.exp(-self.decay_rate * age_days)

            # Calculate importance-weighted similarity
            base_similarity = 1.0 - results['distances'][0][i]
            importance = metadata.get('importance', 0.5)
            adjusted_similarity = base_similarity * decay_factor * (0.5 + 0.5 * importance)

            if adjusted_similarity < min_similarity:
                continue

            # Check tags filter
            record_tags = set(json.loads(metadata.get('tags', '[]')))
            if tags and not record_tags.intersection(tags):
                continue

            # Update access tracking
            doc_id = results['ids'][0][i]
            self._update_access_tracking(doc_id, metadata)

            record = EnhancedMemoryRecord(
                id=doc_id,
                timestamp=timestamp,
                question=metadata['question'],
                agent=metadata['agent'],
                role=metadata['role'],
                phase=metadata['phase'],
                content=results['documents'][0][i],
                metadata=metadata,
                tags=record_tags,
                category=metadata.get('category', 'general'),
                importance=importance,
                access_count=metadata.get('access_count', 0) + 1,
                last_accessed=now,
                decay_factor=decay_factor,
            )

            scored_records.append((adjusted_similarity, record))

        # Sort by adjusted similarity
        scored_records.sort(key=lambda x: x[0], reverse=True)

        return scored_records[:limit]

    def _update_access_tracking(self, doc_id: str, current_metadata: Dict[str, Any]):
        """Update access count and last accessed time"""
        try:
            access_count = current_metadata.get('access_count', 0) + 1
            new_metadata = current_metadata.copy()
            new_metadata['access_count'] = access_count
            new_metadata['last_accessed'] = datetime.utcnow().isoformat()

            # Update in ChromaDB
            self.collection.update(
                ids=[doc_id],
                metadatas=[new_metadata],
            )
        except Exception as e:
            print(f"Error updating access tracking: {e}")

    def extract_learning_insights(
        self,
        client: OpenAI,
        topic: str,
        min_memories: int = 5,
    ) -> str:
        """
        Extract learning insights from past debates on similar topics

        This enables cross-debate learning where agents can benefit
        from patterns and insights discovered in previous debates.

        Args:
            client: OpenAI client for synthesis
            topic: Current debate topic
            min_memories: Minimum number of memories needed

        Returns:
            Synthesized insights string
        """
        # Get topic embedding
        topic_embedding = embed_text(client, topic)

        # Fetch relevant memories
        similar_memories = self.fetch_similar_with_decay(
            query_embedding=topic_embedding,
            limit=min_memories * 2,
            min_similarity=0.6,
        )

        if len(similar_memories) < min_memories:
            return "Insufficient historical data for learning insights."

        # Group by categories
        by_category: Dict[str, List[tuple[float, EnhancedMemoryRecord]]] = {}
        for score, record in similar_memories:
            category = record.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((score, record))

        # Synthesize insights
        synthesis_prompt = f"Topic: {topic}\n\nMemori relevan dari debat sebelumnya:\n\n"

        for category, memories in by_category.items():
            synthesis_prompt += f"\n### {category.title()}:\n"
            for score, mem in memories[:3]:  # Top 3 per category
                synthesis_prompt += f"- [{mem.agent}]: {mem.content[:200]}...\n"

        synthesis_prompt += (
            "\n\nBerdasarkan memori di atas, ekstrak 3-5 insight kunci yang dapat membantu "
            "debat baru tentang topik ini. Format:\n"
            "1. [Insight]\n2. [Insight]\n..."
        )

        try:
            response = client.chat.completions.create(
                model="gemma3:1b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a learning synthesizer extracting insights from past debates.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error synthesizing insights: {str(e)}"

    def export_memory(self, export_path: Path) -> None:
        """
        Export entire memory database to JSON

        Args:
            export_path: Path to export file
        """
        try:
            # Get all memories
            all_results = self.collection.get()

            export_data = {
                "collection_name": self.collection.name,
                "export_timestamp": datetime.utcnow().isoformat(),
                "decay_rate": self.decay_rate,
                "total_memories": len(all_results['ids']),
                "memories": [],
            }

            for i in range(len(all_results['ids'])):
                export_data["memories"].append(
                    {
                        "id": all_results['ids'][i],
                        "document": all_results['documents'][i],
                        "metadata": all_results['metadatas'][i],
                        # Note: embeddings not exported to reduce size
                    }
                )

            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Exported {len(all_results['ids'])} memories to {export_path}")

        except Exception as e:
            print(f"Error exporting memory: {e}")
            raise

    def import_memory(
        self,
        import_path: Path,
        client: OpenAI,
        regenerate_embeddings: bool = True,
    ) -> None:
        """
        Import memory database from JSON

        Args:
            import_path: Path to import file
            client: OpenAI client for regenerating embeddings
            regenerate_embeddings: Whether to regenerate embeddings (recommended)
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            memories = import_data.get('memories', [])
            print(f"Importing {len(memories)} memories...")

            for mem in memories:
                doc_id = mem['id']
                document = mem['document']
                metadata = mem['metadata']

                # Regenerate embedding
                if regenerate_embeddings:
                    embedding = embed_text(client, document)
                else:
                    # Skip if no embedding available
                    continue

                # Add to collection
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[document],
                    metadatas=[metadata],
                )

            print(f"✓ Imported {len(memories)} memories from {import_path}")

        except Exception as e:
            print(f"Error importing memory: {e}")
            raise

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory collection"""
        try:
            all_results = self.collection.get()
            total = len(all_results['ids'])

            if total == 0:
                return {"total_memories": 0}

            # Analyze metadata
            categories: Dict[str, int] = {}
            all_tags: Set[str] = set()
            total_importance = 0.0
            avg_age_days = 0.0
            now = datetime.utcnow()

            for metadata in all_results['metadatas']:
                # Categories
                category = metadata.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1

                # Tags
                tags = set(json.loads(metadata.get('tags', '[]')))
                all_tags.update(tags)

                # Importance
                total_importance += metadata.get('importance', 0.5)

                # Age
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                age_days = (now - timestamp).days
                avg_age_days += age_days

            return {
                "total_memories": total,
                "categories": categories,
                "unique_tags": len(all_tags),
                "all_tags": list(all_tags),
                "avg_importance": total_importance / total,
                "avg_age_days": avg_age_days / total,
            }

        except Exception as e:
            return {"error": str(e)}
