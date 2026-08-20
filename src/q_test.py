import sys
from retrieve import search
from bm25_search import bm25_search
from hybrid_search import rrf_fusion
from rerank import rerank

def compare_all(query_text, top_k=5):
    print(f"\nQUERY: {query_text}\n")

    print("--- Dense ---")
    for r in search(query_text, top_k=top_k):
        print(f"Score: {r['score']:.4f} | Title: {r['title']}")

    print("\n--- BM25 ---")
    for r in bm25_search(query_text, top_k=top_k):
        print(f"Score: {r['score']:.4f} | Title: {r['title']}")

    print("\n--- RRF Fusion ---")
    for r in rrf_fusion(query_text, top_k=top_k):
        print(f"{r['rrf_score']:.4f}  {r['title']}")

    print("\n--- Reranked ---")
    for r in rerank(query_text, top_k=top_k):
        print(f"{r['rerank_score']:.4f}  {r['title']}")
    
if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "hybrid retrieval and reranking for RAG systems"
    compare_all(query)
    