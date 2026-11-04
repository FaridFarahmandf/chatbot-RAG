# 📚 RAG Chatbot — Chat With Your Own Documents

A retrieval-augmented generation (RAG) chatbot that answers questions using
**only your own documents**. It runs on a **free, open-source Llama model** —
locally via [Ollama](https://ollama.com) (no API key, no cost) or on
[Groq](https://groq.com)'s free cloud tier — and grounds every answer in the
documents you provide, showing the source passages behind each response.

Built with Python, LangChain, ChromaDB, sentence-transformers, and Streamlit.

---

## ✨ Features

- **Chat with your documents** — upload PDF, TXT, or Markdown files and ask
  questions about them.
- **100% free LLM** — uses Llama 3.1 (8B) via Ollama locally, or Groq's free
  API. No OpenAI key required.
- **Grounded answers with citations** — the model cites numbered passages
  (`[1]`, `[2]`) and the UI shows only the passages the answer actually cites.
  When the documents don't contain the answer, it says so and shows no sources.
- **Conversation memory** — follow-ups like *"Who did he replace?"* are
  rewritten into standalone questions using the conversation before searching.
- **Multi-topic questions** — a question about two subjects gets a separate
  search for each, so neither is crowded out of the context.
- **Measured with an eval suite** — 13 test cases covering refusals, citations,
  follow-ups and multi-topic questions, with results tracked below.
- **Local, private embeddings** — documents are embedded on your own machine
  with `sentence-transformers`; nothing is sent to a paid embedding API.
- **Simple web UI** — a clean Streamlit interface you can screenshot or demo.

---

## 🏗️ Architecture

```
                ┌─────────────────────────────────────────────┐
                │                Streamlit UI                  │
                │        (chat + upload + rebuild index)       │
                └───────────────────┬─────────────────────────┘
                                    │  question
                                    ▼
   ┌──────────────┐        ┌──────────────────┐        ┌──────────────────┐
   │  Documents   │        │   RAG Engine     │        │   Llama LLM      │
   │ PDF/TXT/MD   │        │                  │        │ (Ollama / Groq)  │
   └──────┬───────┘        │  1. retrieve top-k        └────────▲─────────┘
          │  ingest        │     chunks                         │
          ▼                │  2. build grounded  ───────────────┘
   ┌──────────────┐        │     prompt              grounded answer
   │  Text split  │        │  3. call Llama                     │
   │  + embed     │        │  4. return answer + sources        │
   └──────┬───────┘        └─────────▲────────┘                 │
          │ vectors                  │ relevant chunks           │
          ▼                          │                           ▼
   ┌──────────────┐                  │                   ┌──────────────┐
   │  Chroma      │◄─────────────────┘                   │  answer +    │
   │ vector store │   similarity search                  │  sources     │
   └──────────────┘                                      └──────────────┘
```

**The RAG flow:** documents are split into chunks, embedded into vectors, and
stored in Chroma. At query time the question is embedded, the most similar
chunks are retrieved, and those chunks are inserted into the prompt so the
Llama model answers from your data rather than its own training. Before the
search, follow-ups are made standalone and multi-topic questions get one extra
search per topic.

---

## 📁 Project structure

```
rag-chatbot/
├── app.py                  # Streamlit web UI
├── requirements.txt        # Dependencies
├── .env.example            # Configuration template
├── data/                   # Your documents live here (a sample is included)
│   └── sample_knowledge_base.md
├── evals/
│   ├── questions.json      # Test cases with expected answers and sources
│   └── run_eval.py         # Runs every case and reports pass/fail
└── src/
    ├── config.py           # Loads settings from .env
    ├── ingest.py           # Load -> split -> embed -> store pipeline
    ├── vector_store.py     # Chroma + embedding helpers
    └── rag_engine.py       # Retrieve -> prompt -> generate (the core RAG logic)
```

---

## 🚀 Quick start

### Prerequisites

- Python 3.10+
- One LLM backend:
  - **Option A (recommended, fully free):** [Ollama](https://ollama.com)
    installed locally.
  - **Option B (if your machine is light):** a free
    [Groq API key](https://console.groq.com/keys).

### 1. Install dependencies

```bash
cd rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

**Using Ollama (local, free):** the defaults already point at Ollama. Just
pull the model:

```bash
ollama pull llama3.1:8b
```

Make sure the Ollama app is running.

**Using Groq (free cloud) instead:** edit `.env` and set:

```
LLM_BACKEND=groq
GROQ_API_KEY=your_free_key_here
```

### 3. Build the knowledge base

This embeds the documents in `data/` (a sample is included) into the vector
store:

```bash
python -m src.ingest
```

> The first run downloads the local embedding model (~90 MB). This is a
> one-time download.

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL it prints (usually http://localhost:8501). Ask something like:

- *"What is Acme Robotics' vacation policy?"*
- *"How long is the hardware warranty?"*
- *"What is the maximum payload of the PickBot?"*

Then try a question the documents **don't** cover to see it decline rather
than make something up.

To use your **own** documents: drop files into `data/` (or upload them in the
sidebar) and click **Rebuild index**.

---

## 🧠 How it works (for the technically curious)

1. **Ingestion** (`src/ingest.py`) — documents are loaded and split into
   ~1000-character overlapping chunks with a `RecursiveCharacterTextSplitter`.
2. **Embedding** (`src/vector_store.py`) — each chunk is turned into a vector
   using the local `all-MiniLM-L6-v2` model and stored in a persistent Chroma
   database.
3. **Query rewriting** (`src/rag_engine.py`) — if there's earlier conversation,
   the model rewrites a follow-up into a standalone question ("Who did he
   replace?" → "Who did Michael Carrick replace?"). If the question covers
   several subjects, it's also split into one sub-question per subject.
4. **Retrieval + generation** — the question (and any sub-questions) is
   embedded, the top-k most similar chunks are retrieved, and they're stuffed
   into a grounded prompt sent to Llama as numbered passages. The system prompt
   instructs the model to use only the provided context, cite passage numbers,
   and admit when it doesn't know.
5. **Citation filtering** — the passage numbers in the answer decide which
   sources the UI shows, so uncited chunks never appear as "sources".

Everything is configurable through `.env`: which model, chunk size, overlap,
and how many chunks to retrieve (`TOP_K`).

---

## 📊 Evaluation

`evals/questions.json` holds 13 test cases. Each one states what a correct
reply must contain: required keywords, which files must be cited, how many
passages, or that the bot must refuse. Two-turn cases list an earlier question
in `history`, which is asked first so the follow-up sees the bot's real answer.
Checks are plain string matching, so results are repeatable.

```bash
python -m evals.run_eval
```

Each change was measured with the local `llama3.1:8b` model:

| Change | Eval score |
|---|---|
| Baseline (citations, no memory) | 9/13 |
| + query rewriting for follow-ups | 12/13 |
| + one search per subject for multi-topic questions | 13/13 |

The eval also caught a bad design: a first version that rewrote and split
questions in a single prompt dropped the score to 9–10/13. It over-split simple
questions and reversed the meaning of a follow-up, so the two steps were
separated again.

### Known limitations

- **Small test set.** 13 cases show that each feature works, not how often it
  works on real traffic.
- **Missing facts are caught, invented ones aren't.** Keyword checks confirm
  the right facts appear, but a wrong extra claim in an otherwise correct
  answer still passes.
- **Loose citations.** The 8B model sometimes groups all citation numbers at
  the end of an answer or cites a passage it didn't need.
- **Vague questions on a new topic** can be pulled toward the previous topic,
  e.g. "What stadium do *they* play in?" right after a question about the Joker.
- **Extra latency.** Follow-ups, and questions containing "and" or several
  question marks, each cost one extra model call before the search.

---

## 🔧 Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` (local) or `groq` (free cloud) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per question |

---

## 🛠️ Tech stack

**Python · LangChain · Ollama / Groq (Llama 3.1) · ChromaDB ·
sentence-transformers · Streamlit**

---

## 📝 Resume blurb (feel free to adapt)

> Built a retrieval-augmented generation (RAG) chatbot in Python that answers
> questions over a custom document set using an open-source Llama 3.1 model.
> Implemented the full pipeline — document chunking, local vector embeddings
> (sentence-transformers), semantic retrieval with ChromaDB, and grounded
> prompt construction — plus conversation memory via query rewriting,
> per-topic retrieval for multi-part questions, and citation-filtered sources.
> Built a 13-case evaluation suite and used it to raise the pass rate from
> 9/13 to 13/13.

---

## 📄 License

MIT — free to use, modify, and put on your resume.
