# 🧠 Mini RAG Knowledge Assistant (Streamlit + Chroma + OpenAI)

Upload a `.txt` or `.csv`, embed it into **ChromaDB**, and ask questions using an OpenAI model (RAG).
Includes a simple **Notes tool** (SQLite) and a **Clear KB** button.

## Features
- Upload `.txt` / `.csv`
- Ingest to Chroma vector store
- Ask questions with retrieved context (top-k slider)
- Save answers as notes (SQLite)
- Clear KB button to reset vectors
- Active-source filtering: answers come from the last ingested file

## Quickstart

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Set your OpenAI key (Windows PowerShell)
setx OPENAI_API_KEY "sk-REPLACE_ME"
# open a new terminal window after setx

# 3) Run the app
py -m streamlit run app.py
# Open http://localhost:8501
