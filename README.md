# RAG PDF Chatbot (No LangChain)

A minimal, from-scratch Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about a PDF document. Built without any RAG framework (no LangChain, no LlamaIndex) — every step of the pipeline is implemented directly so the mechanics are fully visible.

## How it works

```
PDF → extract text → chunk → embed chunks → in-memory index
                                                    ↓
User question → embed question → cosine similarity search → top-k chunks → prompt LLM → answer
```

1. **Extract** raw text from the PDF (`pypdf`)
2. **Chunk** the text into overlapping segments
3. **Embed** each chunk into a vector using a local embedding model (Ollama)
4. **Store** chunks + embeddings in memory (and optionally persist to `index.json`)
5. **Retrieve** the most relevant chunks for a question via cosine similarity
6. **Generate** an answer by feeding the retrieved chunks + question to an LLM

## Requirements

- Python 3.x
- [Ollama](https://ollama.com) installed and running locally (for embeddings)
- A generation model — either:
  - A local Ollama model (e.g. `llama3`), **or**
  - A Gemini API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Python packages

```bash
pip install pypdf requests numpy python-dotenv
# only if using Gemini for generation:
pip install google-generativeai
```

### Ollama models

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3             # generation (skip if using Gemini instead)
```

Verify what's installed at any time:

```bash
ollama list
```

## Setup

1. Clone/open this project and place your PDF in the project folder.
2. If using Gemini for generation, create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```
   (No quotes, no spaces around `=`. Add `.env` to `.gitignore`.)
3. Make sure Ollama is running (`ollama serve`, or launch the desktop app).
4. Open `rag_chatbot.ipynb` and run cells top to bottom, or use **Restart Kernel and Run All Cells** to ensure a clean state.

## Notebook structure

| Cell | Purpose |
|---|---|
| Installs | `pip install` dependencies |
| Config | Sets `OLLAMA_URL`, `EMBEDDING_MODEL`, `GENERATION_MODEL` |
| Extract | Pulls text out of the PDF via `pypdf` |
| Chunk | Splits text into overlapping ~800-character chunks |
| Embed (test) | Verifies a single embedding call works, checks vector dimension |
| Build index | Embeds every chunk (slow step — run once per PDF) |
| Persist | Saves the index to `index.json` so re-embedding isn't needed on restart |
| Retrieval | Cosine similarity + top-k ranking |
| Generation | Sends retrieved context + question to the LLM |
| Chat loop | Interactive `while True` loop for asking questions |

## Usage

```python
# after building or loading the index:
answer = ask_question("What does the document say about X?", index)
print(answer)
```

Or run the interactive loop cell and type questions until you enter `quit`.

## Switching between local (Ollama) and hosted (Gemini) generation

Only the generation step changes — chunking, embedding, and retrieval stay identical either way.

**Ollama (local, free, no internet required):**
```python
res = requests.post(
    f"{OLLAMA_URL}/api/generate",
    json={"model": GENERATION_MODEL, "prompt": prompt, "stream": False},
)
return res.json()["response"]
```

**Gemini (hosted, requires API key and internet):**
```python
response = model.generate_content(prompt)
return response.text
```

> Note: Gemini's free tier has strict rate limits and requires a key generated through Google AI Studio (not a raw Cloud Console key) to get non-zero quota. If you hit a `ResourceExhausted` / `429` error with `limit: 0`, regenerate your key via [aistudio.google.com/apikey](https://aistudio.google.com/apikey) or fall back to Ollama.

## Known limitations / next improvements

- **Chunking is fixed-character, not sentence-aware** — can split mid-sentence. Upgrading to paragraph/sentence-aware chunking would improve retrieval quality.
- **In-memory / JSON index only** — fine for small PDFs; for larger documents or multiple PDFs, consider **pgvector** (Postgres extension) instead of a dedicated vector DB.
- **No re-ranking or hybrid search** — retrieval is pure vector similarity; for longer documents, combining with keyword search may improve results.
- **No source citations in answers** — the model isn't currently prompted to indicate which chunk(s) it used.

## Troubleshooting

- **`NameError` on a config variable** — a cell defining it wasn't run in the current kernel session (e.g. after a restart). Use *Restart Kernel and Run All Cells*.
- **`KeyError: 'response'` from Ollama** — the generation model isn't installed. Check `ollama list` and make sure `GENERATION_MODEL` matches an installed model's exact name (including tag, e.g. `llama3:latest`).
- **`DefaultCredentialsError` from Gemini** — `genai.configure(api_key=...)` wasn't called before `generate_content()`, or the API key/env var wasn't loaded. Verify with `print(os.getenv("GEMINI_API_KEY"))`.
- **`ResourceExhausted` / `429` from Gemini with `limit: 0`** — the key isn't provisioned for free-tier use. Regenerate via AI Studio, or fall back to Ollama.
- **"Processed 0 chunks"** — check `len(text)` and `len(chunks)` individually to see whether extraction or chunking failed; often caused by stale kernel state after edits.