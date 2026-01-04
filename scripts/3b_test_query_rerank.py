from modules.query_utils import search_rerank

# --- Example usage ---
query = "Brand Chur"
results = search_rerank(query, k=20)

for r in results:
    print(f"--- Teletext: {r['teletext_id']} | Chunk: {r['chunk_id']} | FTS score: {r["fts_score"]} | Vector score: {r["cosine_similarity"]} || Cross Score: {r['cross_score']}")
    print(f"Title: {r['title']}")
    print(f"Content: {r['content']}")
    print(f"Publication Date: {r["publication_datetime"].date()}")
