"""
Evaluation script for the RAG chatbot.

Runs every case in `questions.json` through the chatbot and checks the result
against simple, deterministic expectations (no LLM judge):

  - "refusal": true   -> the bot must decline and show no sources.
  - "keywords": [...] -> every keyword must appear in the answer.
  - "sources": [...]  -> every listed file must be among the cited sources.
  - "min_sources": n  -> the answer must cite at least n chunks.

A case may also have "history": a list of earlier user questions in the same
conversation, with "question" as the follow-up. The earlier questions are
asked first, so the follow-up sees the bot's real earlier answers.

Run with:
    python -m evals.run_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.rag_engine import REFUSAL, RAGChatbot, RAGResult

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"

def ask(bot: RAGChatbot, case: dict) -> RAGResult:
    """Ask the case's question after playing out any earlier turns."""
    history: list[dict] = []
    for earlier in case.get("history", []):
        reply = bot.ask(earlier, history=history)
        history += [
            {"role": "user", "content": earlier},
            {"role": "assistant", "content": reply.answer},
        ]
    return bot.ask(case["question"], history=history)


def check(case: dict, result: RAGResult) -> list[str]:
    """Return the reasons a case failed (an empty list means it passed)."""
    problems: list[str] = []
    answer = result.answer.lower()
    refused = REFUSAL.rstrip(".").lower() in answer
    cited_files = {d.metadata.get("source", "unknown") for d in result.sources.values()}

    if case.get("refusal"):
        if not refused:
            problems.append("expected a refusal")
        if result.sources:
            problems.append(f"showed sources on a refusal: {sorted(cited_files)}")
        return problems

    if refused:
        problems.append("refused to answer")
    for keyword in case.get("keywords", []):
        if keyword.lower() not in answer:
            problems.append(f"missing keyword '{keyword}'")
    for source in case.get("sources", []):
        if source not in cited_files:
            problems.append(f"did not cite {source}")
    min_sources = case.get("min_sources", 0)
    if len(result.sources) < min_sources:
        problems.append(f"cited {len(result.sources)} chunk(s), expected >= {min_sources}")
    return problems


def main() -> int:
    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    bot = RAGChatbot()
    failed = 0

    for case in cases:
        result = ask(bot, case)
        problems = check(case, result)
        failed += bool(problems)

        print(f"{'FAIL' if problems else 'PASS'}  {case['id']}")
        if problems:
            for turn in case.get("history", []):
                print(f"      earlier:  {turn}")
            print(f"      question: {case['question']}")
            if result.search_query != case["question"]:
                print(f"      rewritten: {result.search_query}")
            print(f"      answer:   {' '.join(result.answer.split())[:200]}")
            for problem in problems:
                print(f"      - {problem}")

    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
