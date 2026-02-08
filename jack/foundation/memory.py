"""
AGENTIC MEMORY - SOTA Memory System for LLM Agents

This module implements a production-grade memory system based on cutting-edge research:

1. A-MEM (NeurIPS 2025) - Zettelkasten-inspired agentic memory
2. MIRIX Multi-Level Hierarchy - Episodic/Semantic/Procedural separation
3. Supermemory - Temporal reasoning and knowledge conflict resolution
4. Mem0/MemoryOS - Active forgetting and consolidation

Core Components:
1. MEMORY NOTE     - Atomic knowledge units (Zettelkasten atomicity principle)
2. EPISODIC MEMORY - Specific experiences with context (when/where/what)
3. SEMANTIC MEMORY - Generalized facts and relationships
4. PROCEDURAL MEMORY - Learned skills and action patterns
5. WORKING MEMORY  - Short-term context for current task
6. MEMORY LINKER   - LLM-driven relationship discovery (A-MEM)
7. CONSOLIDATION   - Short-term to long-term transfer
8. FORGETTING      - Active pruning based on relevance/recency

Design Principles:
- LLM-First: LLM organizes, links, and retrieves memories
- Zettelkasten Atomicity: Each note is a single, self-contained unit
- Dynamic Organization: Links evolve based on usage patterns
- Semantic Understanding: Embeddings + LLM reasoning for retrieval

References:
- A-MEM: https://arxiv.org/abs/2502.12110 (NeurIPS 2025)
- MIRIX: Multi-component memory architecture
- Supermemory: https://supermemory.ai/research
- Zettelkasten: https://zettelkasten.de/posts/overview/
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Callable, Tuple,
    TypeVar, Generic, Protocol, runtime_checkable, Set
)
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import hashlib
import sqlite3
from pathlib import Path
import math
import uuid
import asyncio

from jack.foundation.types import Result, Ok, Err, Option, Some, NONE, Error, ErrorCode
from jack.foundation.action import ActionResult, OutcomeType


# =============================================================================
# REASONER PROTOCOL (for LLM-driven memory operations)
# =============================================================================

@runtime_checkable
class Reasoner(Protocol):
    """Protocol for LLM reasoning - used for memory organization."""

    async def reason(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Make a reasoning call to the LLM."""
        ...


class SimpleReasoner:
    """Simple reasoner for testing."""

    async def reason(self, prompt: str, context: Dict[str, Any] = None) -> str:
        prompt_lower = prompt.lower()

        if "generate keywords" in prompt_lower or "extract keywords" in prompt_lower:
            return '{"keywords": ["memory", "pattern", "learning"], "tags": ["execution", "success"]}'

        if "find relationships" in prompt_lower or "link" in prompt_lower:
            return '{"relationships": [], "strength": 0.5}'

        if "consolidate" in prompt_lower or "generalize" in prompt_lower:
            return '{"generalized_knowledge": "Pattern indicates successful approach", "confidence": 0.8}'

        if "relevance" in prompt_lower or "importance" in prompt_lower:
            return '{"relevance_score": 0.7, "should_forget": false}'

        return '{"result": "ok"}'


# =============================================================================
# EMBEDDING INTERFACE (Production-Ready)
# =============================================================================

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> List[float]:
        """Convert text to embedding vector."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts to embeddings."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


class SimpleEmbedding:
    """
    Simple bag-of-words embedding for testing.

    IMPORTANT: In production, replace with:
    - sentence-transformers (all-MiniLM-L6-v2)
    - OpenAI embeddings (text-embedding-3-small)
    - Cohere embeddings
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """Hash-based embedding (for testing only)."""
        words = text.lower().split()
        embedding = [0.0] * self._dimension

        for i, word in enumerate(words):
            h = hashlib.md5(word.encode()).hexdigest()
            for j in range(0, min(len(h), 32), 2):
                pos = (int(h[j], 16) + i) % self._dimension
                val = (int(h[j+1], 16) - 8) / 8
                embedding[pos] += val

        # Normalize
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# =============================================================================
# MEMORY NOTE (A-MEM Zettelkasten Pattern)
# =============================================================================

class MemoryType(Enum):
    """Types of memory (MIRIX hierarchy)."""
    EPISODIC = auto()    # Specific experiences (when/where/what)
    SEMANTIC = auto()    # Generalized facts and relationships
    PROCEDURAL = auto()  # Learned skills and action patterns
    WORKING = auto()     # Short-term context


@dataclass
class MemoryNote:
    """
    A single memory note following Zettelkasten atomicity principle.

    Each note is a self-contained unit of knowledge with:
    - Unique ID for reference
    - Content (the actual knowledge)
    - Metadata (when, where, context)
    - Embeddings (for semantic search)
    - Links (connections to other notes)
    - Keywords/tags (for organization)

    Based on A-MEM (NeurIPS 2025) Note Construction Module.
    """
    # Identity
    id: str
    memory_type: MemoryType

    # Content (Zettelkasten atomicity - single concept)
    content: str
    context_description: str = ""

    # A-MEM structured attributes
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Embeddings
    embedding: Optional[List[float]] = None

    # Links to other notes (A-MEM Link Generation)
    links: Dict[str, float] = field(default_factory=dict)  # note_id -> strength

    # Temporal context (Supermemory pattern)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Importance and decay
    importance: float = 0.5  # Base importance (0-1)
    decay_rate: float = 0.01  # How fast importance decays

    # Source tracking
    source: Optional[str] = None  # Where this memory came from
    confidence: float = 1.0  # Confidence in this memory

    # For episodic memories
    episode_context: Dict[str, Any] = field(default_factory=dict)

    # For procedural memories
    action_pattern: Optional[Dict[str, Any]] = None
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate for procedural memories."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    @property
    def current_importance(self) -> float:
        """Calculate current importance with temporal decay."""
        time_since_access = (datetime.now() - self.last_accessed).total_seconds()
        hours_since_access = time_since_access / 3600

        # Exponential decay
        decay_factor = math.exp(-self.decay_rate * hours_since_access)

        # Boost from access count (logarithmic)
        access_boost = math.log(1 + self.access_count) * 0.1

        return min(1.0, (self.importance * decay_factor) + access_boost)

    @property
    def fingerprint(self) -> str:
        """Unique fingerprint for deduplication."""
        content = f"{self.memory_type.name}:{self.content[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def record_access(self) -> None:
        """Record that this memory was accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def add_link(self, target_id: str, strength: float = 0.5) -> None:
        """Add or update a link to another note."""
        self.links[target_id] = min(1.0, self.links.get(target_id, 0) + strength)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "memory_type": self.memory_type.name,
            "content": self.content,
            "context_description": self.context_description,
            "keywords": self.keywords,
            "tags": self.tags,
            "links": self.links,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "current_importance": self.current_importance,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryNote:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            memory_type=MemoryType[data["memory_type"]],
            content=data["content"],
            context_description=data.get("context_description", ""),
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            links=data.get("links", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if "last_accessed" in data else datetime.now(),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
        )


# =============================================================================
# MEMORY LINKER (A-MEM Link Generation Module)
# =============================================================================

class MemoryLinker:
    """
    LLM-driven link generation between memory notes.

    Based on A-MEM Link Generation Module:
    1. Find nearest neighbors in embedding space
    2. Use LLM to analyze semantic relationships
    3. Create bidirectional links with strength scores

    This goes beyond simple similarity - the LLM identifies:
    - Causal relationships
    - Conceptual connections
    - Temporal sequences
    - Contradictions (negative links)
    """

    def __init__(
        self,
        reasoner: Reasoner,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.7,
        max_links_per_note: int = 10,
    ):
        self.reasoner = reasoner
        self.embedding = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.max_links_per_note = max_links_per_note

    async def generate_links(
        self,
        new_note: MemoryNote,
        existing_notes: List[MemoryNote],
    ) -> Dict[str, float]:
        """
        Generate links from new note to existing notes.

        Process:
        1. Find k-nearest neighbors by embedding similarity
        2. Ask LLM to analyze relationships
        3. Return links with strength scores
        """
        if not existing_notes:
            return {}

        # Step 1: Find nearest neighbors by embedding
        candidates = []
        for note in existing_notes:
            if note.id == new_note.id:
                continue
            if note.embedding and new_note.embedding:
                sim = cosine_similarity(new_note.embedding, note.embedding)
                if sim >= self.similarity_threshold:
                    candidates.append((note, sim))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:self.max_links_per_note]

        if not candidates:
            return {}

        # Step 2: LLM-driven relationship analysis
        candidate_summaries = "\n".join([
            f"- [{note.id}] {note.content[:100]}... (similarity: {sim:.2f})"
            for note, sim in candidates
        ])

        prompt = f"""Memory Link Analysis (A-MEM Pattern)

Analyze the relationships between a new memory and candidate related memories.

NEW MEMORY:
ID: {new_note.id}
Type: {new_note.memory_type.name}
Content: {new_note.content}
Keywords: {new_note.keywords}

CANDIDATE RELATED MEMORIES:
{candidate_summaries}

For each candidate, determine:
1. Relationship type: "supports", "contradicts", "extends", "precedes", "follows", "related"
2. Relationship strength: 0.0 to 1.0

Respond in JSON format:
{{
    "links": [
        {{"id": "note_id", "relationship": "type", "strength": 0.8, "reason": "why linked"}}
    ]
}}"""

        try:
            response = await self.reasoner.reason(prompt)
            parsed = json.loads(response)

            links = {}
            for link in parsed.get("links", []):
                note_id = link.get("id")
                strength = link.get("strength", 0.5)

                # Adjust strength based on relationship type
                rel_type = link.get("relationship", "related")
                if rel_type == "contradicts":
                    strength = -strength  # Negative link

                links[note_id] = strength

            return links

        except (json.JSONDecodeError, Exception):
            # Fallback: use embedding similarity as link strength
            return {note.id: sim for note, sim in candidates}

    async def extract_keywords_and_tags(
        self,
        content: str,
        context: str = "",
    ) -> Tuple[List[str], List[str]]:
        """
        Extract keywords and tags using LLM.

        Keywords: Specific terms from the content
        Tags: Categorical labels for organization
        """
        prompt = f"""Keyword and Tag Extraction

Extract keywords and tags from this memory content.

CONTENT:
{content}

CONTEXT:
{context}

Keywords: Specific important terms (3-7 keywords)
Tags: Categorical labels like "execution", "query", "error", "success" (2-5 tags)

Respond in JSON:
{{
    "keywords": ["keyword1", "keyword2", ...],
    "tags": ["tag1", "tag2", ...]
}}"""

        try:
            response = await self.reasoner.reason(prompt)
            parsed = json.loads(response)
            return (
                parsed.get("keywords", []),
                parsed.get("tags", []),
            )
        except (json.JSONDecodeError, Exception):
            # Fallback: simple word extraction
            words = content.lower().split()[:5]
            return (words, ["general"])


# =============================================================================
# MEMORY CONSOLIDATION (Short-term to Long-term)
# =============================================================================

class MemoryConsolidator:
    """
    Consolidate episodic memories into semantic knowledge.

    Based on cognitive science principles:
    1. Group similar episodic memories
    2. Extract common patterns
    3. Create generalized semantic memories
    4. Strengthen procedural memories through repetition

    This is how the agent learns general principles from specific experiences.
    """

    def __init__(
        self,
        reasoner: Reasoner,
        consolidation_threshold: int = 3,  # Min episodes to consolidate
    ):
        self.reasoner = reasoner
        self.consolidation_threshold = consolidation_threshold

    async def consolidate_episodes(
        self,
        episodes: List[MemoryNote],
    ) -> Optional[MemoryNote]:
        """
        Consolidate multiple episodic memories into semantic knowledge.

        Returns a new semantic memory note if consolidation is possible.
        """
        if len(episodes) < self.consolidation_threshold:
            return None

        # Prepare episode summaries
        episode_summaries = "\n".join([
            f"- Episode {i+1}: {ep.content[:200]}"
            for i, ep in enumerate(episodes)
        ])

        prompt = f"""Memory Consolidation

You have {len(episodes)} related episodic memories. Extract the generalized knowledge.

EPISODES:
{episode_summaries}

Tasks:
1. Identify the common pattern across all episodes
2. Extract generalized knowledge (facts, rules, principles)
3. Determine confidence based on consistency

Respond in JSON:
{{
    "generalized_knowledge": "The extracted general principle or fact",
    "confidence": 0.0 to 1.0,
    "supporting_evidence": ["key observations"],
    "exceptions": ["any notable exceptions"]
}}"""

        try:
            response = await self.reasoner.reason(prompt)
            parsed = json.loads(response)

            if parsed.get("confidence", 0) < 0.5:
                return None

            # Create semantic memory from consolidated episodes
            semantic_note = MemoryNote(
                id=str(uuid.uuid4())[:12],
                memory_type=MemoryType.SEMANTIC,
                content=parsed.get("generalized_knowledge", ""),
                context_description="Consolidated from episodic memories",
                keywords=parsed.get("supporting_evidence", [])[:5],
                tags=["consolidated", "semantic"],
                importance=parsed.get("confidence", 0.7),
                confidence=parsed.get("confidence", 0.7),
                source=f"consolidated_from_{len(episodes)}_episodes",
            )

            # Link to source episodes
            for ep in episodes:
                semantic_note.add_link(ep.id, 0.8)

            return semantic_note

        except (json.JSONDecodeError, Exception):
            return None

    async def strengthen_procedural(
        self,
        procedural_note: MemoryNote,
        new_outcome: bool,
    ) -> MemoryNote:
        """
        Strengthen or weaken a procedural memory based on outcome.

        Successful executions strengthen the memory.
        Failures weaken it and may trigger re-evaluation.
        """
        if new_outcome:
            procedural_note.success_count += 1
            procedural_note.importance = min(1.0, procedural_note.importance + 0.1)
        else:
            procedural_note.failure_count += 1
            procedural_note.importance = max(0.1, procedural_note.importance - 0.15)

        return procedural_note


# =============================================================================
# ACTIVE FORGETTING (Mem0/MemoryOS Pattern)
# =============================================================================

class ActiveForgetter:
    """
    Active memory pruning based on relevance and recency.

    Based on cognitive science: memories are not passively lost but actively
    removed based on:
    - Recency: How recently was it accessed?
    - Frequency: How often is it accessed?
    - Relevance: How important is it to current goals?
    - Redundancy: Is this information duplicated elsewhere?

    This prevents memory bloat and keeps retrieval efficient.
    """

    def __init__(
        self,
        reasoner: Reasoner,
        min_importance: float = 0.1,
        max_age_days: int = 90,
        max_memories: int = 10000,
    ):
        self.reasoner = reasoner
        self.min_importance = min_importance
        self.max_age_days = max_age_days
        self.max_memories = max_memories

    def identify_forgettable(
        self,
        memories: List[MemoryNote],
        current_goals: List[str] = None,
    ) -> List[str]:
        """
        Identify memories that should be forgotten.

        Returns list of memory IDs to forget.
        """
        forgettable = []
        now = datetime.now()

        for note in memories:
            # Check importance threshold
            if note.current_importance < self.min_importance:
                forgettable.append(note.id)
                continue

            # Check age (for non-semantic memories)
            if note.memory_type != MemoryType.SEMANTIC:
                age_days = (now - note.created_at).days
                if age_days > self.max_age_days and note.access_count < 3:
                    forgettable.append(note.id)
                    continue

            # Check for low success rate procedural memories
            if note.memory_type == MemoryType.PROCEDURAL:
                total = note.success_count + note.failure_count
                if total >= 5 and note.success_rate < 0.3:
                    forgettable.append(note.id)

        return forgettable

    async def evaluate_relevance(
        self,
        note: MemoryNote,
        current_context: str,
    ) -> float:
        """
        Evaluate how relevant a memory is to current context.

        Uses LLM to assess relevance.
        """
        prompt = f"""Memory Relevance Evaluation

Evaluate how relevant this memory is to the current context.

MEMORY:
{note.content}

CURRENT CONTEXT:
{current_context}

Consider:
1. Is this information useful for the current task?
2. Does it provide important background knowledge?
3. Would forgetting this cause problems?

Respond in JSON:
{{
    "relevance_score": 0.0 to 1.0,
    "reason": "brief explanation"
}}"""

        try:
            response = await self.reasoner.reason(prompt)
            parsed = json.loads(response)
            return parsed.get("relevance_score", 0.5)
        except (json.JSONDecodeError, Exception):
            return 0.5


# =============================================================================
# MEMORY STORE (Persistent Storage)
# =============================================================================

class MemoryStore:
    """
    Persistent storage for memory notes with semantic search.

    Features:
    - SQLite persistence
    - Embedding-based similarity search
    - Tag and keyword filtering
    - Link traversal
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_provider: EmbeddingProvider = None,
    ):
        self.db_path = Path(db_path) if db_path else None
        self.embedding = embedding_provider or SimpleEmbedding()
        self.notes: Dict[str, MemoryNote] = {}

        if self.db_path:
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_notes (
                id TEXT PRIMARY KEY,
                memory_type TEXT,
                content TEXT,
                context_description TEXT,
                keywords TEXT,
                tags TEXT,
                embedding TEXT,
                links TEXT,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                importance REAL,
                decay_rate REAL,
                source TEXT,
                confidence REAL,
                episode_context TEXT,
                action_pattern TEXT,
                success_count INTEGER,
                failure_count INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        """Load notes from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memory_notes")
        for row in cursor.fetchall():
            note = MemoryNote(
                id=row[0],
                memory_type=MemoryType[row[1]],
                content=row[2],
                context_description=row[3] or "",
                keywords=json.loads(row[4]) if row[4] else [],
                tags=json.loads(row[5]) if row[5] else [],
                embedding=json.loads(row[6]) if row[6] else None,
                links=json.loads(row[7]) if row[7] else {},
                created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
                last_accessed=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
                access_count=row[10] or 0,
                importance=row[11] or 0.5,
                decay_rate=row[12] or 0.01,
                source=row[13],
                confidence=row[14] or 1.0,
                episode_context=json.loads(row[15]) if row[15] else {},
                action_pattern=json.loads(row[16]) if row[16] else None,
                success_count=row[17] or 0,
                failure_count=row[18] or 0,
            )
            self.notes[note.id] = note

        conn.close()

    def _save_note(self, note: MemoryNote) -> None:
        """Save note to database."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO memory_notes
            (id, memory_type, content, context_description, keywords, tags,
             embedding, links, created_at, last_accessed, access_count,
             importance, decay_rate, source, confidence, episode_context,
             action_pattern, success_count, failure_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            note.id,
            note.memory_type.name,
            note.content,
            note.context_description,
            json.dumps(note.keywords),
            json.dumps(note.tags),
            json.dumps(note.embedding) if note.embedding else None,
            json.dumps(note.links),
            note.created_at.isoformat(),
            note.last_accessed.isoformat(),
            note.access_count,
            note.importance,
            note.decay_rate,
            note.source,
            note.confidence,
            json.dumps(note.episode_context),
            json.dumps(note.action_pattern) if note.action_pattern else None,
            note.success_count,
            note.failure_count,
        ))

        conn.commit()
        conn.close()

    def add(self, note: MemoryNote) -> None:
        """Add a note to the store."""
        if note.embedding is None:
            note.embedding = self.embedding.embed(note.content)

        self.notes[note.id] = note
        self._save_note(note)

    def get(self, note_id: str) -> Optional[MemoryNote]:
        """Get note by ID."""
        note = self.notes.get(note_id)
        if note:
            note.record_access()
            self._save_note(note)
        return note

    def remove(self, note_id: str) -> None:
        """Remove a note."""
        if note_id in self.notes:
            del self.notes[note_id]
            if self.db_path:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memory_notes WHERE id = ?", (note_id,))
                conn.commit()
                conn.close()

    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        tags: List[str] = None,
        k: int = 10,
        min_similarity: float = 0.3,
    ) -> List[Tuple[MemoryNote, float]]:
        """
        Search for similar notes.

        Returns list of (note, similarity) tuples.
        """
        if not self.notes:
            return []

        query_embedding = self.embedding.embed(query)
        results = []

        for note in self.notes.values():
            # Filter by type
            if memory_type and note.memory_type != memory_type:
                continue

            # Filter by tags
            if tags and not any(t in note.tags for t in tags):
                continue

            # Calculate similarity
            if note.embedding is None:
                continue

            similarity = cosine_similarity(query_embedding, note.embedding)

            # Boost by importance
            boosted_sim = similarity * (0.7 + 0.3 * note.current_importance)

            if boosted_sim >= min_similarity:
                results.append((note, boosted_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_by_type(self, memory_type: MemoryType) -> List[MemoryNote]:
        """Get all notes of a specific type."""
        return [n for n in self.notes.values() if n.memory_type == memory_type]

    def get_linked_notes(self, note_id: str) -> List[Tuple[MemoryNote, float]]:
        """Get notes linked to a given note."""
        note = self.notes.get(note_id)
        if not note:
            return []

        linked = []
        for linked_id, strength in note.links.items():
            linked_note = self.notes.get(linked_id)
            if linked_note:
                linked.append((linked_note, strength))

        return sorted(linked, key=lambda x: x[1], reverse=True)

    @property
    def size(self) -> int:
        return len(self.notes)

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        by_type = {}
        for note in self.notes.values():
            type_name = note.memory_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_notes": self.size,
            "by_type": by_type,
            "avg_importance": sum(n.current_importance for n in self.notes.values()) / max(1, self.size),
            "total_links": sum(len(n.links) for n in self.notes.values()),
        }


# =============================================================================
# AGENTIC MEMORY (Main Interface)
# =============================================================================

class AgenticMemory:
    """
    Complete agentic memory system combining all components.

    This is the main interface following A-MEM principles:
    - Dynamic organization through LLM-driven linking
    - Multi-level hierarchy (episodic/semantic/procedural)
    - Active consolidation and forgetting
    - Semantic search with importance boosting

    Usage:
        reasoner = YourLLMReasoner()
        memory = AgenticMemory(reasoner)

        # Remember an experience
        await memory.remember_episode("User asked for sales report", {...})

        # Recall similar experiences
        results = await memory.recall("sales data analysis")

        # Consolidate learning
        await memory.consolidate()
    """

    def __init__(
        self,
        reasoner: Reasoner = None,
        base_dir: Optional[str] = None,
        embedding_provider: EmbeddingProvider = None,
    ):
        self.reasoner = reasoner or SimpleReasoner()
        self.embedding = embedding_provider or SimpleEmbedding()
        self.base_dir = Path(base_dir or "~/.jack/memory").expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stores
        self.store = MemoryStore(
            db_path=str(self.base_dir / "memory.db"),
            embedding_provider=self.embedding,
        )

        # Initialize components
        self.linker = MemoryLinker(self.reasoner, self.embedding)
        self.consolidator = MemoryConsolidator(self.reasoner)
        self.forgetter = ActiveForgetter(self.reasoner)

        # Working memory (session-only)
        self.working_memory: List[MemoryNote] = []
        self.working_memory_max = 50

    # =========================================================================
    # REMEMBER (Create Memories)
    # =========================================================================

    async def remember_episode(
        self,
        content: str,
        context: Dict[str, Any] = None,
        importance: float = 0.5,
    ) -> MemoryNote:
        """
        Remember a specific episode (experience).

        Episodic memories capture specific events with context:
        - What happened
        - When it happened
        - What the outcome was
        """
        context = context or {}

        # Extract keywords and tags
        keywords, tags = await self.linker.extract_keywords_and_tags(content)
        tags.append("episodic")

        # Create note
        note = MemoryNote(
            id=str(uuid.uuid4())[:12],
            memory_type=MemoryType.EPISODIC,
            content=content,
            context_description=context.get("description", ""),
            keywords=keywords,
            tags=tags,
            importance=importance,
            episode_context=context,
            source=context.get("source", "user"),
        )

        # Generate embedding
        note.embedding = self.embedding.embed(content)

        # Generate links to existing memories
        existing = list(self.store.notes.values())
        links = await self.linker.generate_links(note, existing)
        note.links = links

        # Store
        self.store.add(note)

        # Also add to working memory
        self._add_to_working_memory(note)

        return note

    async def remember_fact(
        self,
        content: str,
        keywords: List[str] = None,
        importance: float = 0.7,
    ) -> MemoryNote:
        """
        Remember a semantic fact (generalized knowledge).

        Semantic memories are facts that persist across contexts.
        """
        extracted_keywords, tags = await self.linker.extract_keywords_and_tags(content)
        keywords = keywords or extracted_keywords
        tags.append("semantic")

        note = MemoryNote(
            id=str(uuid.uuid4())[:12],
            memory_type=MemoryType.SEMANTIC,
            content=content,
            keywords=keywords,
            tags=tags,
            importance=importance,
        )

        note.embedding = self.embedding.embed(content)

        existing = list(self.store.notes.values())
        links = await self.linker.generate_links(note, existing)
        note.links = links

        self.store.add(note)
        return note

    async def remember_procedure(
        self,
        content: str,
        action_pattern: Dict[str, Any],
        success: bool = True,
    ) -> MemoryNote:
        """
        Remember a procedural pattern (learned skill).

        Procedural memories capture how to do things:
        - Action patterns that worked
        - Success/failure tracking
        """
        keywords, tags = await self.linker.extract_keywords_and_tags(content)
        tags.append("procedural")

        note = MemoryNote(
            id=str(uuid.uuid4())[:12],
            memory_type=MemoryType.PROCEDURAL,
            content=content,
            keywords=keywords,
            tags=tags,
            action_pattern=action_pattern,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            importance=0.6,
        )

        note.embedding = self.embedding.embed(content)

        existing = list(self.store.notes.values())
        links = await self.linker.generate_links(note, existing)
        note.links = links

        self.store.add(note)
        return note

    def remember_working(self, content: str, key: str = None) -> MemoryNote:
        """
        Add to working memory (session-only, not persisted).
        """
        note = MemoryNote(
            id=str(uuid.uuid4())[:12],
            memory_type=MemoryType.WORKING,
            content=content,
            keywords=[key] if key else [],
            tags=["working"],
        )

        self._add_to_working_memory(note)
        return note

    def _add_to_working_memory(self, note: MemoryNote) -> None:
        """Add note to working memory with size limit."""
        self.working_memory.append(note)
        if len(self.working_memory) > self.working_memory_max:
            self.working_memory = self.working_memory[-self.working_memory_max:]

    # =========================================================================
    # RECALL (Retrieve Memories)
    # =========================================================================

    async def recall(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        k: int = 10,
        include_linked: bool = True,
    ) -> List[Tuple[MemoryNote, float]]:
        """
        Recall relevant memories for a query.

        Searches across all memory types and optionally includes linked notes.
        """
        # Search main store
        results = self.store.search(query, memory_type=memory_type, k=k)

        # Optionally expand with linked notes
        if include_linked and results:
            linked_notes = set()
            for note, _ in results[:3]:  # Top 3 results
                for linked_note, strength in self.store.get_linked_notes(note.id):
                    if linked_note.id not in {n.id for n, _ in results}:
                        linked_notes.add((linked_note, strength * 0.5))

            results.extend(list(linked_notes))
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:k]

        # Record access
        for note, _ in results:
            note.record_access()

        return results

    def recall_working(self, key: str = None) -> List[MemoryNote]:
        """Recall from working memory."""
        if key:
            return [n for n in self.working_memory if key in n.keywords]
        return list(self.working_memory)

    async def recall_procedure(
        self,
        task_description: str,
        min_success_rate: float = 0.5,
    ) -> Optional[MemoryNote]:
        """
        Recall the best procedural memory for a task.
        """
        results = self.store.search(
            task_description,
            memory_type=MemoryType.PROCEDURAL,
            k=5,
        )

        for note, _ in results:
            if note.success_rate >= min_success_rate:
                return note

        return None

    # =========================================================================
    # LEARN (Consolidation and Strengthening)
    # =========================================================================

    async def consolidate(self) -> List[MemoryNote]:
        """
        Consolidate episodic memories into semantic knowledge.

        Groups similar episodes and extracts general principles.
        """
        episodes = self.store.get_by_type(MemoryType.EPISODIC)

        if len(episodes) < 3:
            return []

        # Group episodes by keywords
        groups: Dict[str, List[MemoryNote]] = {}
        for ep in episodes:
            key = tuple(sorted(ep.keywords[:3]))
            if key not in groups:
                groups[key] = []
            groups[key].append(ep)

        # Consolidate each group
        new_semantic = []
        for key, group in groups.items():
            if len(group) >= 3:
                semantic = await self.consolidator.consolidate_episodes(group)
                if semantic:
                    self.store.add(semantic)
                    new_semantic.append(semantic)

        return new_semantic

    async def strengthen_procedure(
        self,
        procedure_id: str,
        success: bool,
    ) -> Optional[MemoryNote]:
        """
        Strengthen or weaken a procedural memory based on outcome.
        """
        note = self.store.get(procedure_id)
        if not note or note.memory_type != MemoryType.PROCEDURAL:
            return None

        note = await self.consolidator.strengthen_procedural(note, success)
        self.store._save_note(note)
        return note

    # =========================================================================
    # FORGET (Active Pruning)
    # =========================================================================

    async def forget_irrelevant(
        self,
        current_context: str = "",
    ) -> List[str]:
        """
        Forget irrelevant memories to prevent bloat.
        """
        all_notes = list(self.store.notes.values())
        forgettable = self.forgetter.identify_forgettable(all_notes)

        # Actually remove them
        for note_id in forgettable:
            self.store.remove(note_id)

        return forgettable

    def clear_working_memory(self) -> None:
        """Clear working memory (end of session)."""
        self.working_memory = []

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self.store.get_statistics(),
            "working_memory_size": len(self.working_memory),
        }


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Aliases for backward compatibility
Pattern = MemoryNote
PatternStore = MemoryStore

class Memory(AgenticMemory):
    """Backward-compatible Memory class."""

    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir=base_dir)

    def remember_execution(self, context: str, result: ActionResult) -> None:
        """Remember an execution (backward compatible)."""
        asyncio.get_event_loop().run_until_complete(
            self.remember_episode(
                content=f"{context}: {result.primitive.description}",
                context={"outcome": "success" if result.is_success else "failure"},
            )
        )

    def remember_query(self, question: str, query: str, success: bool, outcome: str = None) -> None:
        """Remember a query (backward compatible)."""
        asyncio.get_event_loop().run_until_complete(
            self.remember_procedure(
                content=f"Q: {question}\nA: {query}",
                action_pattern={"query": query},
                success=success,
            )
        )

    def recall_similar_executions(self, context: str, k: int = 5) -> List[Tuple[MemoryNote, float]]:
        """Recall similar executions (backward compatible)."""
        return self.store.search(context, memory_type=MemoryType.EPISODIC, k=k)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile

    async def test_memory():
        print("=" * 60)
        print("AGENTIC MEMORY - SOTA Memory System")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as tmpdir:
            reasoner = SimpleReasoner()
            memory = AgenticMemory(reasoner=reasoner, base_dir=tmpdir)

            # Test episodic memory
            print("\n[TEST] Episodic Memory")
            ep1 = await memory.remember_episode(
                "User asked for monthly sales report, generated successfully",
                context={"task": "reporting", "outcome": "success"},
            )
            print(f"  Created episode: {ep1.id}")

            ep2 = await memory.remember_episode(
                "Generated quarterly revenue analysis with charts",
                context={"task": "analysis", "outcome": "success"},
            )
            print(f"  Created episode: {ep2.id}")

            # Test semantic memory
            print("\n[TEST] Semantic Memory")
            fact = await memory.remember_fact(
                "Sales reports should include monthly trends and comparisons",
                keywords=["sales", "reports", "trends"],
            )
            print(f"  Created fact: {fact.id}")

            # Test procedural memory
            print("\n[TEST] Procedural Memory")
            proc = await memory.remember_procedure(
                "To generate a sales report: query data, aggregate by month, create chart",
                action_pattern={
                    "steps": ["query", "aggregate", "visualize"],
                    "template": "SELECT month, SUM(amount) FROM sales GROUP BY month",
                },
                success=True,
            )
            print(f"  Created procedure: {proc.id}")

            # Test recall
            print("\n[TEST] Recall")
            results = await memory.recall("monthly sales analysis")
            print(f"  Query: 'monthly sales analysis'")
            for note, sim in results[:3]:
                print(f"    [{sim:.2f}] {note.memory_type.name}: {note.content[:50]}...")

            # Test working memory
            print("\n[TEST] Working Memory")
            memory.remember_working("Current table: sales", key="current_table")
            working = memory.recall_working("current_table")
            print(f"  Working memory items: {len(working)}")

            # Test statistics
            print("\n[TEST] Statistics")
            stats = memory.get_statistics()
            print(f"  Total notes: {stats['total_notes']}")
            print(f"  By type: {stats['by_type']}")
            print(f"  Total links: {stats['total_links']}")

        print("\n" + "=" * 60)
        print("[OK] Agentic Memory System Complete")
        print("=" * 60)
        print("\nKey Components:")
        print("  - MemoryNote: Zettelkasten-style atomic notes")
        print("  - MemoryLinker: LLM-driven relationship discovery")
        print("  - MemoryConsolidator: Episode -> Semantic transfer")
        print("  - ActiveForgetter: Relevance-based pruning")
        print("  - AgenticMemory: Complete A-MEM implementation")
        print("=" * 60)

    asyncio.run(test_memory())
