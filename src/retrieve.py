import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
embeddings_path = os.path.join(project_root, "data", "processed", "embeddings.parquet") 

df = pd.read_parquet(embeddings_path)
doc_matrix = np.stack(df["embedding"].values).astype(np.float32)

def embed_query(query_text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# print(doc_matrix.shape)  # (num_docs, embedding_dim)

def cosine_similarity(query_vec, doc_matrix):
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    return doc_norms @ query_norm  # matrix multiplication to get cosine similarities

def search(query_text, top_k=5):
    query_vec = embed_query(query_text)
    scores = cosine_similarity(query_vec, doc_matrix)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "arxiv_id": df.iloc[idx]["arxiv_id"],
            "title": df.iloc[idx]["title"],
            "score": scores[idx]
        })
    return results

# results = search("hybrid retrieval and reranking for RAG systems")
# for r in results:
#     print(r["score"], r["title"])


if __name__ == "__main__":
    results = search("hybrid retrieval and reranking for RAG systems")
    for r in results:
        print(r["score"], r["title"])