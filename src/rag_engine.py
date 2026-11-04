"""
The RAG engine: ties retrieval and generation together.

Given a user question, it:
  0. If there is earlier conversation, rewrites a follow-up ("Who did he
     replace?") into a standalone question so retrieval knows what it's about.
  1. Embeds the question and retrieves the most relevant chunks from Chroma.
  2. Builds a prompt that grounds the LLM in ONLY those retrieved chunks.
  3. Calls the Llama model (via Ollama locally, or Groq's free cloud API).
  4. Returns the answer plus the source chunks that supported it.

This "retrieve -> stuff into prompt -> generate" pattern is the classic RAG
approach and is written explicitly here (rather than hidden behind a prebuilt
chain) so the mechanics are easy to read and explain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import config
from .vector_store import load_vector_store

# Exact sentence the model is told to use when the context has no answer.
# `ask` looks for it to know that no source backed the reply.
REFUSAL = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions using \
ONLY the provided context. Follow these rules strictly:

- Base your answer solely on the context below. Do not use outside knowledge.
- The context is a list of numbered passages. After every claim, cite the \
passage it came from using its number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, reply with exactly this \
sentence and nothing else: "{REFUSAL}"
- Be concise and accurate. When useful, quote or reference the relevant part.
- Do not invent facts, sources, or numbers.
"""

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

REWRITE_PROMPT = """Rewrite the follow-up question so it can be understood \
without the conversation. Replace pronouns and vague references ("he", "it", \
"that film") with what they refer to in the conversation. Use only the \
conversation, never outside knowledge. If the follow-up is already \
self-contained, return it unchanged. Do not answer it.
Reply with only the rewritten question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

# How many earlier messages (user + assistant) to use when rewriting.
MAX_HISTORY = 6


@dataclass
class RAGResult:
    """An answer plus the source documents that grounded it."""

    answer: str
    # Only the passages the answer actually cites, keyed by citation number.
    sources: dict[int, Document]
    # The question actually searched and answered: the user's question,
    # rewritten to stand on its own when there was earlier conversation.
    search_query: str


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


def _cited_sources(answer: str, docs: list[Document]) -> dict[int, Document]:
    """Return the retrieved chunks the answer cites, keyed by their number.

    The retriever always returns `top_k` chunks, relevant or not, so only the
    ones the model referenced count as sources. A refusal has none.
    """
    if REFUSAL.rstrip(".").lower() in answer.lower():
        return {}

    cited: dict[int, Document] = {}
    # Matches [1] as well as grouped citations like [1, 3].
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for number in map(int, re.split(r"\s*,\s*", group)):
            if 1 <= number <= len(docs):
                cited[number] = docs[number - 1]
    return dict(sorted(cited.items()))


class RAGChatbot:
    """A small, reusable RAG chatbot object."""

    def __init__(self) -> None:
        self.vector_store = load_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.top_k}
        )
        self.llm = _build_llm()

    def _rewrite(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up into a standalone question using the conversation."""
        lines = []
        for message in history[-MAX_HISTORY:]:
            speaker = "User" if message["role"] == "user" else "Assistant"
            # Citation markers mean nothing outside the original answer.
            text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", message["content"])
            lines.append(f"{speaker}: {text}")
        prompt = REWRITE_PROMPT.format(history="\n".join(lines), question=question)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        lines = [line.strip() for line in response.content.splitlines() if line.strip()]
        if not lines:
            return question
        # Small models sometimes echo the label or wrap the reply in quotes.
        rewritten = re.sub(r"^standalone question:\s*", "", lines[0], flags=re.I)
        return rewritten.strip('"\'') or question

    def ask(self, question: str, history: list[dict] | None = None) -> RAGResult:
        """Answer a question with retrieval-augmented generation.

        `history` is the earlier conversation, oldest first, as
        {"role": "user" | "assistant", "content": ...} messages. It's used only
        to resolve what a follow-up refers to.
        """
        # 0. Make follow-ups standalone so retrieval can find the right chunks.
        query = self._rewrite(question, history) if history else question

        # 1. Retrieve relevant chunks.
        docs = self.retriever.invoke(query)

        # 2. Build the grounded prompt.
        context = _format_context(docs)
        user_prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        # 3. Generate the answer.
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = answer.strip()

        # 4. Return the answer + only the sources it cites.
        return RAGResult(
            answer=answer,
            sources=_cited_sources(answer, docs),
            search_query=query,
        )


if __name__ == "__main__":
    # Tiny CLI for quick manual testing without the web UI.
    bot = RAGChatbot()
    print("RAG chatbot ready. Ask a question (Ctrl+C to quit).\n")
    history: list[dict] = []
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            result = bot.ask(q, history=history)
            history += [
                {"role": "user", "content": q},
                {"role": "assistant", "content": result.answer},
            ]
            print(f"\nBot: {result.answer}\n")
            srcs = ", ".join(sorted({d.metadata.get('source', '?')
                                     for d in result.sources.values()}))
            if srcs:
                print(f"(sources: {srcs})\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
