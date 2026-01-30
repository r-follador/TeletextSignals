from modules.mpnet_similarity import mpnet_nearest_docs_by_teletext_id
example_id = "05b9c3cf-6813-40a3-a06e-612896e740fd"
results = mpnet_nearest_docs_by_teletext_id(example_id, k=10, include_self=True)
print(results)