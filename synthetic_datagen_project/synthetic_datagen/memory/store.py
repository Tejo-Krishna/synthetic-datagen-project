"""
memory/store.py
---------------
MemoryStore -- the exact interface required by the PDF.

Backend priority:
  1. mem0ai      (pip install mem0ai)           -- required by spec
  2. _VectorStore (sentence-transformers + qdrant-client) -- local fallback
  3. _InMemoryStore                             -- keyword fallback, always available

Interface (PDF spec):
    class MemoryStore:
        def add(self, content: str, scope: str, metadata: dict) -> None
        def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]

Scopes:
    "session" -- in-conversation grounding (tool outputs for argument filling)
    "corpus"  -- cross-conversation grounding (planner diversity)

Install:
    pip install mem0ai
    pip install sentence-transformers qdrant-client  # optional fallback
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Fallback in-memory store (used when packages are not installed)
# ---------------------------------------------------------------------------

@dataclass
class _MemoryEntry:
    entry_id: str
    content: str
    scope: str
    metadata: dict
    timestamp: float


class _InMemoryStore:
    """
    Keyword-search fallback store.
    Used when sentence-transformers or qdrant-client are not installed.
    """

    def __init__(self):
        self._entries: list[_MemoryEntry] = []

    def add(self, content: str, scope: str, metadata: dict) -> None:
        self._entries.append(_MemoryEntry(
            entry_id=str(uuid.uuid4()),
            content=content,
            scope=scope,
            metadata=metadata,
            timestamp=time.time(),
        ))

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        query_words = set(query.lower().split())
        scored = []
        for entry in self._entries:
            if entry.scope != scope:
                continue
            overlap = len(query_words & set(entry.content.lower().split()))
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [
            {"id": e.entry_id, "memory": e.content,
             "metadata": e.metadata, "score": s}
            for s, e in scored[:top_k]
        ]

    def clear_scope(self, scope: str) -> None:
        self._entries = [e for e in self._entries if e.scope != scope]

    def count(self, scope: str | None = None) -> int:
        if scope:
            return sum(1 for e in self._entries if e.scope == scope)
        return len(self._entries)


# ---------------------------------------------------------------------------
# mem0ai backend (required by spec)
# ---------------------------------------------------------------------------

class _Mem0Store:
    """
    mem0ai-backed store -- satisfies the hard requirement in the PDF spec.

    Uses scope as user_id to namespace entries:
      scope="session_{conversation_id}" for per-conversation grounding
      scope="corpus" for cross-conversation planner diversity

    Requires: pip install mem0ai
    mem0 may require an OpenAI API key in its default config.
    Falls back automatically in MemoryStore if unavailable.
    """

    def __init__(self):
        from mem0 import Memory
        self._mem = Memory()

    def add(self, content: str, scope: str, metadata: dict) -> None:
        self._mem.add(
            messages=[{"role": "user", "content": content}],
            user_id=scope,
            metadata=metadata,
        )

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        results = self._mem.search(query=query, user_id=scope, limit=top_k)
        # mem0 returns a dict with a "results" key in newer versions,
        # or a plain list in older versions — handle both
        if isinstance(results, dict):
            results = results.get("results", [])
        output = []
        for r in results:
            if isinstance(r, dict):
                output.append({
                    "id":       str(r.get("id", "")),
                    "memory":   r.get("memory", ""),
                    "metadata": r.get("metadata") or {},
                    "score":    float(r.get("score", 0.0)),
                })
        return output[:top_k]


# ---------------------------------------------------------------------------
# Vector store using sentence-transformers + qdrant-client
# ---------------------------------------------------------------------------

class _VectorStore:
    """
    Local vector store -- no API key, no external service required.

    Uses sentence-transformers for embeddings and qdrant-client (in-memory)
    for vector storage and similarity search. Content is stored verbatim
    with no LLM extraction step.

    Requires: pip install sentence-transformers qdrant-client
    """

    MODEL      = "all-MiniLM-L6-v2"   # 22MB, 384 dims
    DIMS       = 384
    COLLECTION = "synthetic_datagen"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._encoder = SentenceTransformer(self.MODEL)
        self._client  = QdrantClient(":memory:")
        self._client.create_collection(
            collection_name=self.COLLECTION,
            vectors_config=VectorParams(size=self.DIMS, distance=Distance.COSINE),
        )
        self._counter = 0

    def add(self, content: str, scope: str, metadata: dict) -> None:
        from qdrant_client.models import PointStruct

        vector = self._encoder.encode(content).tolist()
        self._client.upsert(
            collection_name=self.COLLECTION,
            points=[PointStruct(
                id=self._counter,
                vector=vector,
                payload={"memory": content, "scope": scope, **metadata},
            )],
        )
        self._counter += 1

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        vector = self._encoder.encode(query).tolist()
        scope_filter = Filter(must=[
            FieldCondition(key="scope", match=MatchValue(value=scope))
        ])

        # qdrant-client >= 1.7 uses query_points(); older versions use search()
        try:
            result = self._client.query_points(
                collection_name=self.COLLECTION,
                query=vector,
                query_filter=scope_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = result.points
        except AttributeError:
            hits = self._client.search(
                collection_name=self.COLLECTION,
                query_vector=vector,
                query_filter=scope_filter,
                limit=top_k,
                with_payload=True,
            )

        return [
            {
                "id": str(h.id),
                "memory": h.payload.get("memory", ""),
                "metadata": {k: v for k, v in h.payload.items()
                             if k not in ("memory", "scope")},
                "score": h.score,
            }
            for h in hits
        ]


# ---------------------------------------------------------------------------
# Public MemoryStore -- exact interface from PDF
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    MemoryStore -- stable interface used by all pipeline components.

    Components must depend only on this class, never on the backend directly.

    Scopes:
        "session" -- per-conversation tool output grounding
        "corpus"  -- cross-conversation planner diversity grounding

    Usage:
        store = MemoryStore()
        store.add(
            content=json.dumps(tool_output),
            scope="session",
            metadata={"conversation_id": "c1", "step": 2, "endpoint": "search_flights"},
        )
        results = store.search(query="flight_id", scope="session", top_k=3)

    PDF grounding metric:
        Count a step as grounded whenever search() returns >= 1 result,
        regardless of score threshold.
    """

    def __init__(self, use_mem0: bool = True):
        """
        Args:
            use_mem0: if True, attempt vector store backend (sentence-transformers
                      + qdrant-client). Falls back to keyword store automatically
                      if packages are not installed.
                      Set False to force keyword fallback (useful in tests).
        """
        self._using_vector = False

        if use_mem0:
            # Priority 1: mem0ai (required by spec)
            try:
                self._backend = _Mem0Store()
                self._using_vector = True
                print("[memory] using mem0ai backend")
            except ImportError:
                # Priority 2: sentence-transformers + qdrant-client
                try:
                    self._backend = _VectorStore()
                    self._using_vector = True
                    print("[memory] mem0ai not found -- using vector store fallback. "
                          "Install with: pip install mem0ai")
                except ImportError as e:
                    print(
                        f"[memory] vector store unavailable ({e}) -- "
                        "using in-process keyword fallback. "
                        "Install with: pip install mem0ai"
                    )
                    self._backend = _InMemoryStore()
                except Exception as e:
                    print(f"[memory] vector store init failed ({e}) -- using in-process fallback")
                    self._backend = _InMemoryStore()
            except Exception as e:
                print(f"[memory] mem0ai init failed ({e}) -- using in-process fallback")
                self._backend = _InMemoryStore()
        else:
            self._backend = _InMemoryStore()

    def add(self, content: str, scope: str, metadata: dict) -> None:
        """
        Store a memory entry.

        Args:
            content:  string to store (JSON tool output or summary text)
            scope:    "session" or "corpus"
            metadata: free-form dict (conversation_id, step, endpoint, etc.)
        """
        self._backend.add(content=content, scope=scope, metadata=metadata)

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        """
        Search for relevant memory entries within a scope.

        PDF definition: count as grounded whenever this returns >= 1 result,
        regardless of score threshold.

        Args:
            query:  search query string
            scope:  "session" or "corpus"
            top_k:  max results to return

        Returns:
            List of dicts with keys: id, memory, metadata, score
        """
        try:
            return self._backend.search(query=query, scope=scope, top_k=top_k)
        except Exception as e:
            print(f"[memory] search failed ({e}), returning empty results")
            return []

    @property
    def backend_type(self) -> str:
        """Returns 'vector' or 'in_memory' -- useful for logging and tests."""
        return "vector" if self._using_vector else "in_memory"

    def clear_session(self, conversation_id: str) -> None:
        """Clear session entries for a conversation (fallback backend only)."""
        if isinstance(self._backend, _InMemoryStore):
            self._backend._entries = [
                e for e in self._backend._entries
                if not (e.scope == "session"
                        and e.metadata.get("conversation_id") == conversation_id)
            ]
