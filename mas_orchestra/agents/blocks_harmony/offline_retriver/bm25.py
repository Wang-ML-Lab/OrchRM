"""
BM25 Retriever for Offline Web Search

Uses bm25s (pure Python) instead of Pyserini/Lucene to avoid Java dependency issues.
Index is pre-built and loaded from disk.
"""

import os
import json
import pickle
from typing import Dict, List, Any


def initialize_bm25_retriever(index_path: str) -> Dict[str, Any]:
    """Initialize BM25 retriever using bm25s (pure Python).

    Args:
        index_path: Path to the directory containing the bm25s index
                    (looks for bm25s/ subfolder first, then falls back to index_path itself)

    Returns:
        Dictionary containing retriever metadata and searcher instance
    """
    try:
        import bm25s

        # Support both flat path and bm25s/ subfolder
        bm25s_path = os.path.join(index_path, "bm25s")
        if not os.path.exists(bm25s_path):
            bm25s_path = index_path

        if not os.path.exists(bm25s_path):
            raise FileNotFoundError(f"bm25s index not found at {bm25s_path}")

        retriever = bm25s.BM25.load(bm25s_path, load_corpus=False)

        # Load docids mapping
        docids_path = os.path.join(bm25s_path, "docids.pkl")
        if os.path.exists(docids_path):
            with open(docids_path, "rb") as f:
                docids = pickle.load(f)
        else:
            docids = list(range(retriever.scores.shape[1]))

        print(f"   ✓ BM25s loaded: {len(docids)} documents from {bm25s_path}")
        return {"type": "bm25", "searcher": retriever, "docids": docids}

    except ImportError as e:
        raise ImportError(f"bm25s not installed: {e}. Install with: pip install bm25s")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize BM25: {e}")


async def search_bm25(
    retriever: Dict[str, Any],
    query: str,
    corpus: Dict[str, Any],
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """Perform BM25 search using bm25s.

    Args:
        retriever: BM25 retriever dictionary from initialize_bm25_retriever
        query: Search query string
        corpus: Dictionary mapping docid to document data
        max_results: Maximum number of results to return

    Returns:
        List of search results with docid, title, content, url, and score
    """
    import asyncio
    import bm25s

    searcher = retriever["searcher"]
    docids = retriever["docids"]

    def _search():
        tokens = bm25s.tokenize([query], show_progress=False)
        results, scores = searcher.retrieve(tokens, k=min(max_results, len(docids)))
        return results[0], scores[0]

    hits, hit_scores = await asyncio.to_thread(_search)

    results = []
    for idx, score in zip(hits, hit_scores):
        doc_id = docids[idx]
        if doc_id in corpus:
            doc = corpus[doc_id]
            results.append({
                'docid': doc_id,
                'title': doc.get('title', 'Untitled'),
                'content': doc.get('text', doc.get('contents', '')),
                'url': doc.get('url', ''),
                'score': float(score),
            })

    return results
