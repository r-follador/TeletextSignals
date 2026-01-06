from modules.query_utils import fts_search, vector_similarity_search

query = "Theater Chur?"

# --- Example usage FTS Search ---
results = fts_search(query, k=20)

for r in results:
    print(f"--- Teletext: {r['teletext_id']} | Chunk: {r['chunk_id']} | FTS score: {r["fts_score"]} | Vector score: {r["cosine_similarity"]}")
    print(f"Title: {r['title']}")
    print(f"Content: {r['content']}")
    print(f"Publication Date: {r["publication_datetime"].date()}")

print("-"*20)
# --- Example usage Vector similarity Search ---
results = vector_similarity_search(query, k=20)

for r in results:
    print(f"--- Teletext: {r['teletext_id']} | Chunk: {r['chunk_id']} | FTS score: {r["fts_score"]} | Vector score: {r["cosine_similarity"]}")
    print(f"Title: {r['title']}")
    print(f"Content: {r['content']}")
    print(f"Publication Date: {r["publication_datetime"].date()}")
