from retrieve import df

def find_id_by_title(partial_title):
    matches = df[df["title"].str.contains(partial_title, case=False, regex=False)]
    if len(matches) == 0:
        print("No match found")
    elif len(matches) > 1:
        print("Multiple matches:")
        print(matches[["arxiv_id", "title"]])
    else:
        print(matches.iloc[0]["arxiv_id"], "-", matches.iloc[0]["title"])

if __name__ == "__main__":
    find_id_by_title("Blended RAG")
    find_id_by_title("TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance Beyond RAG")
    find_id_by_title("MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation")
    find_id_by_title("MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries")
    find_id_by_title("Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG")
    find_id_by_title("Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents")
    find_id_by_title("PRA-RAG: Provably Robust Aggregation")
    find_id_by_title("Prompt-RAG: Pioneering Vector Embedding-Free Retrieval-Augmented Generation")
    find_id_by_title("Biomedical Literature Q&A System Using Retrieval-Augmented Generation (RAG)")
    find_id_by_title("Toward Optimal Search and Retrieval for RAG")
    find_id_by_title("Progressive Searching for Retrieval in RAG")
    find_id_by_title("Enhancing Technical Documents Retrieval for RAG")
    find_id_by_title("SmartChunk Retrieval")
    find_id_by_title("Optimizing Retrieval for RAG via Reinforcement Learning")
    find_id_by_title("Legal RAG")
    find_id_by_title("MolE-RAG")
    find_id_by_title("LaB-RAG")
    find_id_by_title("Biomedical literature Q&A")
    find_id_by_title("estimator")
    find_id_by_title("regression")
    find_id_by_title("DyG-RAG")
    find_id_by_title("SVD-RAG")