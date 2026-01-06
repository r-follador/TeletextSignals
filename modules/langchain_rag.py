from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union
from langchain_core.documents import Document
from typing import Iterator, Tuple, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from modules.langchain_teletext_retriever import TeletextHybridRetriever, documents_to_search_results

def _format_docs_for_context(docs, max_chars: int | None) -> str:
    sections: list[str] = []
    total = 0

    for idx, d in enumerate(docs, start=1):
        md = d.metadata or {}
        title = (md.get("title") or "").strip()
        dt = md.get("publication_datetime") or ""
        teletext_id = md.get("teletext_id") or ""
        text = (d.page_content or "").strip()

        header = f"[{idx}] {title} ({dt}) id={teletext_id}".strip()
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


@dataclass
class TeletextRAG:
    """
    Notebook-friendly wrapper that builds a 2-step RAG chain:
      retrieve (your TeletextHybridRetriever) -> single LLM call (Ollama)

    Usage in a notebook:
      rag = TeletextRAG(model="gemma3:4b-it-qat", base_url="http://localhost:11434")
      rag.ask("Wann und weshalb starb Pele?")
      # or streaming:
      for tok in rag.stream("..."):
          print(tok, end="")
    """

    model: str = "gemma3:4b-it-qat"
    base_url: str = "http://localhost:11434"
    k: int = 20
    top_k: int = 3
    max_context_chars: int = 6000
    temperature: float = 0.0
    debug: bool = False
    score_floor: float = -1.0
    batch_size: int = 32

    system_prompt: str = (
        "Answer the question using only the context. "
        "If the context does not contain the answer, say you don't know. "
        "Cite sources like [1], [2] based on the context blocks."
        "Use following HTML tags for readability <b>, <i>, <br>."
    )

    def _build_chain(self):
        retriever = TeletextHybridRetriever(
            k=self.k,
            top_k=self.top_k,
            score_floor=self.score_floor,
            batch_size=self.batch_size,
            debug=self.debug,
        )

        llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
            ]
        )

        chain = (
                {
                    "question": RunnablePassthrough(),
                    "context": retriever
                               | RunnableLambda(lambda docs: _format_docs_for_context(docs, self.max_context_chars)),
                }
                | prompt
                | llm
                | StrOutputParser()
        )
        return chain

    def ask(self, question: str) -> str:
        """Return a complete answer (non-streaming)."""
        chain = self._build_chain()
        return chain.invoke(question)

    def stream(self, question: str) -> Iterator[str]:
        """Yield tokens/chunks as they are generated."""
        chain = self._build_chain()
        yield from chain.stream(question)

    def _build_llm_and_prompt(self):
        llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
            ]
        )
        return llm, prompt

    def _build_retriever(self):
        return TeletextHybridRetriever(
            k=self.k,
            top_k=self.top_k,
            score_floor=self.score_floor,
            batch_size=self.batch_size,
            debug=self.debug,
        )

    def ask_with_sources(self, question: str):
        """
        Returns (answer, sources_as_SearchResult_like_dicts)

        sources are in the same order as the context blocks [1], [2], ...
        """
        retriever = self._build_retriever()
        docs: List[Document] = retriever.invoke(question)

        context = _format_docs_for_context(docs, self.max_context_chars)

        llm, prompt = self._build_llm_and_prompt()
        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({"question": question, "context": context})
        sources = documents_to_search_results(docs)
        return answer, sources

    def stream_with_sources(self, question: str):
        """
        Returns (token_iterator, sources_as_SearchResult_like_dicts)

        The iterator yields tokens. Sources are available immediately for display after streaming.
        """
        retriever = self._build_retriever()
        docs: List[Document] = retriever.invoke(question)
        context = _format_docs_for_context(docs, self.max_context_chars)

        llm, prompt = self._build_llm_and_prompt()
        chain = prompt | llm | StrOutputParser()

        return chain.stream({"question": question, "context": context}), documents_to_search_results(docs)





def make_teletext_rag(
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
) -> TeletextRAG:
    rag = TeletextRAG(
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
