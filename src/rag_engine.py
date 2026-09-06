"""
The RAG engine: ties retrieval and generation together.

Given a user question, it:
  1. Embeds the question and retrieves the most relevant chunks from Chroma.
  2. Builds a prompt that grounds the LLM in ONLY those retrieved chunks.
  3. Calls the Llama model (via Ollama locally, or Groq's free cloud API).
  4. Returns the answer plus the source chunks that supported it.

This "retrieve -> stuff into prompt -> generate" pattern is the classic RAG
approach and is written explicitly here (rather than hidden behind a prebuilt
chain) so the mechanics are easy to read and explain.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import config
from .vector_store import load_vector_store

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using \
ONLY the provided context. Follow these rules strictly:

- Base your answer solely on the context below. Do not use outside knowledge.
- If the context does not contain the answer, say: "I don't have enough \
information in the provided documents to answer that."
- Be concise and accurate. When useful, quote or reference the relevant part.
- Do not invent facts, sources, or numbers.
"""

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResult:
    """An answer plus the source documents that grounded it."""

    answer: str
    sources: list[Document]


def _build_llm() -> BaseChatModel:
    """Instantiate the chat model for the configured backend."""
    backend = config.llm_backend

    if backend == "ollama":
        # Local, fully free. Requires the Ollama app + a pulled model.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=0.1,  # low temperature = factual, grounded answers
        )

    if backend == "groq":
        # Free cloud tier. Requires GROQ_API_KEY.
        if not config.groq_api_key:
            raise ValueError(
                "LLM_BACKEND=groq but GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com/keys "
                "and add it to your .env file."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=0.1,
        )

    raise ValueError(
        f"Unknown LLM_BACKEND '{backend}'. Use 'ollama' or 'groq'."
    )


def _format_context(docs: list[Document]) -> str:
    """Turn retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(blocks)


class RAGChatbot:
    """A small, reusable RAG chatbot object."""

    def __init__(self) -> None:
        self.vector_store = load_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.top_k}
        )
        self.llm = _build_llm()

    def ask(self, question: str) -> RAGResult:
        """Answer a single question with retrieval-augmented generation."""
        # 1. Retrieve relevant chunks.
        docs = self.retriever.invoke(question)

        # 2. Build the grounded prompt.
        context = _format_context(docs)
        user_prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # 3. Generate the answer.
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        # 4. Return answer + sources.
        return RAGResult(answer=answer.strip(), sources=docs)


if __name__ == "__main__":
    # Tiny CLI for quick manual testing without the web UI.
    bot = RAGChatbot()
    print("RAG chatbot ready. Ask a question (Ctrl+C to quit).\n")
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            result = bot.ask(q)
            print(f"\nBot: {result.answer}\n")
            srcs = ", ".join(sorted({d.metadata.get('source', '?')
                                     for d in result.sources}))
            print(f"(sources: {srcs})\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
