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
    find_id_by_title("Improving Retrieval for RAG based question answering")
    