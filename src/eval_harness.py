import json
import os
from retrieve import search
from bm25_search import bm25_search
from hybrid_search import rrf_fusion
from rerank import rerank

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
eval_path = os.path.join(project_root, "data", "eval", "eval_queries.json")

with open(eval_path, "r", encoding="utf-8") as f:
    eval_queries = json.load(f)

# print(f"Loaded {len(eval_queries)} eval queries")

def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids)

def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0.0

def evaluate_method(method_fn, k=5):
    from collections import defaultdict
    recalls_by_cat = defaultdict(list)
    rrs_by_cat = defaultdict(list)

    for item in eval_queries:
        query = item["query"]
        relevant_ids = item["relevant_arxiv_ids"]
        category = item["category"]

        if len(relevant_ids) == 0:
            continue

        results = method_fn(query, top_k=k)
        retrieved_ids = [r["arxiv_id"] for r in results]

        recalls_by_cat[category].append(recall_at_k(retrieved_ids, relevant_ids, k))
        rrs_by_cat[category].append(reciprocal_rank(retrieved_ids, relevant_ids))

    return recalls_by_cat, rrs_by_cat

if __name__ == "__main__":
    k = 5
    print(f"Evaluating methods with k={k}...\n")

    methods = {
        "Dense": search,
        "BM25": bm25_search,
        "RRF Fusion": rrf_fusion,
        "Reranked": rerank
    }

    for method_name, method_fn in methods.items():
        recalls_by_cat, rrs_by_cat = evaluate_method(method_fn, k=k)
        print(f"--- {method_name} ---")
        for category in recalls_by_cat:
            avg_recall = sum(recalls_by_cat[category]) / len(recalls_by_cat[category])
            avg_rr = sum(rrs_by_cat[category]) / len(rrs_by_cat[category])
            print(f"Category: {category} | Avg Recall@{k}: {avg_recall:.4f} | Avg MRR: {avg_rr:.4f}")
        print()