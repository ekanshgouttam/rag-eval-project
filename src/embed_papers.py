import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
raw_dir = os.path.join(project_root, "data", "raw")

papers = []
for filename in os.listdir(raw_dir):
    filepath = os.path.join(raw_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        paper = json.load(f)
        papers.append(paper)

# print(f"Loaded {len(papers)} papers")
# print(papers[0])
# print(len(papers[0]["title"]))
# print(repr(papers[0]["title"]))  # repr shows exact characters, less prone to terminal wrapping weirdness

texts = []
arxiv_ids = []

for paper in papers:
    title = paper["title"].rstrip(".!?:")  # strip trailing punctuation before appending our own period
    combined = f"{title}. {paper['abstract']}"
    texts.append(combined)
    arxiv_ids.append(paper["arxiv_id"])

# print(texts[0])
# print(repr(texts[0][:100]))  # just the first 100 characters, right around the junction

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

# print(len(response.data))
# print(response.data[0])

embeddings_ordered = [None] * len(texts)
for item in response.data:
    embeddings_ordered[item.index] = item.embedding

embeddings_ordered = np.array(embeddings_ordered, dtype=np.float32)

df = pd.DataFrame({
    "arxiv_id": arxiv_ids,
    "title": [p["title"] for p in papers],
    "abstract": [p["abstract"] for p in papers],
    "embedding": list(embeddings_ordered)
})

processed_dir = os.path.join(project_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
output_path = os.path.join(processed_dir, "embeddings.parquet")
df.to_parquet(output_path, engine="pyarrow")

print(f"Saved {len(df)} rows to {output_path}")

check = pd.read_parquet(output_path)
print(check.shape)
print(check.iloc[0]["arxiv_id"], check.iloc[0]["embedding"].dtype, len(check.iloc[0]["embedding"]))