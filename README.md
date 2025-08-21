# Mini AI Knowledge Assistant (RAG + Tool)

This is a minimal, **one-file** Streamlit app that fulfills your challenge:

- **Frontend**: chat UI + file upload (Streamlit)
- **Backend / AI logic**: chunk text, create embeddings, store in **ChromaDB**, retrieve and call an LLM
- **Tool**: save notes to a local **SQLite** database
- **Integrations**: local DB (SQLite). You can easily swap in Strapi or Postgres later.
- **Infra**: run locally or deploy to Streamlit Cloud / EC2. Config via env vars.

## Quickstart

1) Python 3.10+ and pip installed.
2) Create a virtual env (optional but recommended).  
3) Install deps:
```bash
pip install -r requirements.txt
```
4) Set your API key (Linux/Mac):
```bash
export OPENAI_API_KEY=sk-...
```
   On Windows (Powershell):
```powershell
setx OPENAI_API_KEY "sk-..."
```
5) Run:
```bash
streamlit run app.py
```

## How it works

1. Upload a `.txt` or `.csv` on the sidebar → the app **chunks** and **embeds** the text with `text-embedding-3-small` and stores vectors in **Chroma**.
2. Ask a question → we embed the query, retrieve top-k relevant chunks, and call the chat model (default `gpt-4o-mini`).
3. Save any answer as a **Note** via the sidebar tool (stored in local SQLite `notes.db`).

## Environment variables

- `OPENAI_API_KEY`: **required**
- `CHAT_MODEL`: default `gpt-4o-mini` (change to any available chat model)
- `EMBED_MODEL`: default `text-embedding-3-small`
- `CHROMA_DIR`: default `./chroma_store`
- `DB_PATH`: default `./notes.db`

## Replace SQLite Tool with Strapi (optional)

- Run Strapi locally, create a `Note` content type with fields `title` (text) and `content` (rich text).
- Add a function to POST to Strapi REST API with your token. Replace `save_note()` to call Strapi instead of SQLite.

## Deployment

### Streamlit Cloud (fastest)
- Push this repo to GitHub.
- In Streamlit Cloud, create an app pointing to `app.py`.
- Add `OPENAI_API_KEY` as a secret.

### AWS EC2
- SSH to EC2, install Python and dependencies, set env vars.
- Run `streamlit run app.py --server.port 80 --server.address 0.0.0.0` and open the public DNS.

### Docker (optional)
Create a `Dockerfile`:
```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
ENV OPENAI_API_KEY=changeme
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build & run:
```bash
docker build -t mini-rag .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... mini-rag
```

## Notes

- This is intentionally minimal (no LangChain) so you can see the moving parts.
- Swap `Chroma` for Pinecone/Weaviate if you prefer.
- For production, add auth, rate-limiting, and proper error handling.