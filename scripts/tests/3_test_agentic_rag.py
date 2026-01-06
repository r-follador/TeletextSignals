from modules.langchain_agentic_rag import make_teletext_agentic_rag

rag = make_teletext_agentic_rag(
    model="qwen2.5:7b-instruct",
    base_url="http://localhost:11434",
    k=40,
    top_k=5,
    debug=True,
)

answer, sources = rag.ask_with_sources("Welche Staatsoberhäupter haben wann das WEF besucht?")
print(answer)
print(sources)
