from modules.langchain_rag import make_teletext_rag

rag = make_teletext_rag(
    model="gemma3:4b-it-qat",
    base_url="http://localhost:11434",
    k=40,
    top_k=5,
    debug=True,
)

answer, sources = rag.ask_with_sources("Wann und weshalb starb Pelé?")
print(answer)
print(sources)
