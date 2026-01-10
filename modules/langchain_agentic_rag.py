from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from modules.langchain_teletext_retriever import (
    TeletextHybridRetriever,
    documents_to_search_results,
)


def _format_docs_for_context(
    docs: List[Document],
    max_chars: int | None,
    start_index: int = 1,
) -> str:
    sections: list[str] = []
    total = 0

    for idx, d in enumerate(docs, start=start_index):
        md = d.metadata or {}
        title = (md.get("title") or "").strip()
        dt = md.get("publication_datetime") or ""
        teletext_id = md.get("teletext_id") or ""
        text = (d.page_content or "").strip()

        header = f"[{idx}] {title} (Date: {dt}) id={teletext_id}".strip()
        section = header + "\n" + text

        if max_chars is not None and total + len(section) > max_chars:
            remaining = max_chars - total - len(header) - 1
            if remaining <= 0:
                break
            trimmed = text[:remaining].rsplit(" ", 1)[0]
            section = header + "\n" + trimmed

        sections.append(section)
        total += len(section)

        if max_chars is not None and total >= max_chars:
            break

    return "\n\n".join(sections)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()
    return str(content)


@dataclass
class TeletextAgenticRAG:
    """
    Agentic RAG wrapper that lets the model decide when to retrieve.

    Usage:
      rag = TeletextAgenticRAG(model="qwen2.5:7b-instruct", base_url="http://localhost:11434")
      answer, sources = rag.ask_with_sources("Wann und weshalb starb Pele?")
    """

    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    k: int = 20
    top_k: int = 3
    max_context_chars: int = 6000
    temperature: float = 0.0
    debug: bool = False
    score_floor: float = -1.0
    batch_size: int = 32

    system_prompt: str = (
        "You are a helpful assistant that can use the teletext_search tool. "
        "Use the tool when you need facts from swiss news sources about national and international news."
        "Answer using only retrieved context, cite sources like [1], [2]. "
        "If the context does not contain the answer, try a different query."
        "Structure your answer in chronological fashion using the provided dates. Today is January 2026."
    )

    _last_docs: List[Document] = field(default_factory=list, init=False, repr=False)

    def _build_retriever(self) -> TeletextHybridRetriever:
        return TeletextHybridRetriever(
            k=self.k,
            top_k=self.top_k,
            score_floor=self.score_floor,
            batch_size=self.batch_size,
            debug=self.debug,
        )

    def _build_agent(self):
        retriever = self._build_retriever()

        @tool("teletext_search")
        def teletext_search(query: str) -> str:
            """Search swiss and international news articles with semantic queries in german language
            and return context blocks with numbered citations."""
            docs: List[Document] = retriever.invoke(query)
            self._last_docs = docs
            return _format_docs_for_context(docs, self.max_context_chars, start_index=1)

        llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        return create_agent(
            model=llm,
            tools=[teletext_search],
            system_prompt=self.system_prompt,
            debug=self.debug,
        )

    def _extract_answer(self, messages: List[object]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return _content_to_text(msg.content)
        if messages:
            last = messages[-1]
            content = getattr(last, "content", last)
            return _content_to_text(content)
        return ""

    def ask(self, question: str) -> str:
        """Return a complete answer (non-streaming)."""
        self._last_docs.clear()
        agent = self._build_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return self._extract_answer(result["messages"])

    def ask_with_sources(self, question: str):
        """
        Returns (answer, sources_as_SearchResult_like_dicts).

        Sources correspond to the context blocks provided by the last retrieval tool call.
        """
        self._last_docs.clear()
        agent = self._build_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        answer = self._extract_answer(result["messages"])
        docs = self._last_docs
        sources = documents_to_search_results(docs)
        return answer, sources


def make_teletext_agentic_rag(
    *,
    model: str = "gemma3:4b-it-qat",
    base_url: str = "http://localhost:11434",
    k: int = 20,
    top_k: int = 3,
    max_context_chars: int = 6000,
    temperature: float = 0.0,
    debug: bool = False,
    score_floor: float = -1.0,
    batch_size: int = 32,
    system_prompt: Optional[str] = None,
) -> TeletextAgenticRAG:
    rag = TeletextAgenticRAG(
        model=model,
        base_url=base_url,
        k=k,
        top_k=top_k,
        max_context_chars=max_context_chars,
        temperature=temperature,
        debug=debug,
        score_floor=score_floor,
        batch_size=batch_size,
    )
    if system_prompt:
        rag.system_prompt = system_prompt
    return rag
