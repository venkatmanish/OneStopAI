# BuddyAi

Runnable v1 of a Streamlit chatbot backed by FastAPI, LangGraph, hybrid RAG,
document versioning, Excel analysis, OCR ingestion, tool calling, and explainable answers.

## Stack

- UI: Streamlit
- API: FastAPI
- Agent: LangGraph
- LLM: Groq first, Ollama fallback
- Embeddings: local SentenceTransformers MiniLM/BGE-compatible interface
- Vector DB: Qdrant
- Metadata DB: PostgreSQL
- Cache-ready: Redis
- PDF/OCR: PyMuPDF, OpenCV, Tesseract, including PDF hyperlink annotations
- PowerPoint: python-pptx for PPTX slides, tables, notes, images, and hyperlinks
- Excel: pandas and DuckDB
- Web/weather: Tavily and OpenWeather free tiers
- Folder connector: Google Drive API service-account scaffold

## Local Setup

This repo targets Python 3.12 because several AI/OCR packages lag latest Python releases.

```bash
cp .env.example .env
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Install services on macOS:

```bash
brew install postgresql@16 redis qdrant tesseract
brew services start postgresql@16
brew services start redis
brew services start qdrant
```

Create a database matching `.env`:

```bash
/opt/homebrew/opt/postgresql@16/bin/createuser -s rag_user || true
/opt/homebrew/opt/postgresql@16/bin/createdb agentic_rag -O rag_user || true
```

Set at least one LLM backend:

```bash
export GROQ_API_KEY=...
```

or run Ollama:

```bash
ollama pull llama3.1
ollama serve
```

## Google Drive Folder Sync

Create a Google Cloud service account with Drive API enabled, download its JSON key,
and share the target Drive folder with the service account email.

Set these in `.env`:

```bash
GOOGLE_DRIVE_FOLDER_ID=...
GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/to/service-account.json
GOOGLE_DRIVE_SHARED_DRIVE_ID= # optional for shared drives
```

The sync endpoint lists supported files in that folder, detects changes using
`md5Checksum`, Drive `version`, and `modifiedTime`, then ingests changed PDFs,
PowerPoint decks, spreadsheets, CSVs, images, and exported Google Docs/Sheets/Slides.

## Run

```bash
./scripts/run_backend.sh
./scripts/run_frontend.sh
```

Open Streamlit at the URL printed by Streamlit, usually `http://localhost:8501`.

## API

- `GET /health`
- `POST /chat`
- `POST /chat/stream`
- `POST /ingest/upload`
- `POST /google-drive/sync`
- `GET /documents`
- `GET /documents/{document_id}/versions`

## Notes

- If Qdrant is unavailable, ingestion still stores chunks in PostgreSQL and retrieval falls back to BM25.
- Google Drive folder sync uses service-account credentials, lists supported folder files, downloads changed content, and sends it through the normal ingestion/versioning pipeline.
- Chart understanding is best-effort OCR and reports lower confidence when evidence is weak.
# OneStopAI
