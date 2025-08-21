
import os
import io
import sqlite3
import streamlit as st
import pandas as pd
from typing import List
from openai import OpenAI
import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "kb01"  # nombre válido para Chroma (>=3 caracteres)


# -------------- Config --------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")  # change if you like
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store_v2")  # usa un store nuevo
DB_PATH = os.getenv("DB_PATH", "./notes.db")

# -------------- Helpers --------------
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Very simple character-based chunker.
    """
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
        if i < 0:
            break
    return chunks

def read_upload(file_bytes: bytes, filename: str) -> str:
    """
    Devuelve el contenido como texto unificado.
    - Si es CSV: lo lee con pandas y lo vuelve CSV (texto).
    - Si es TXT: decodifica bytes -> string.
    """
    from io import BytesIO
    import pandas as pd

    if filename.lower().endswith(".csv"):
        # Intentar varios encodings comunes en Windows
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(BytesIO(file_bytes), encoding=enc, engine="python")
                return df.astype(str).to_csv(index=False)
            except UnicodeDecodeError:
                continue
        # Último recurso: ignorar caracteres problemáticos
        df = pd.read_csv(BytesIO(file_bytes), encoding="latin-1", engine="python", on_bad_lines="skip")
        return df.astype(str).to_csv(index=False)
    else:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("latin-1", errors="ignore")

def ensure_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()

def save_note(title: str, content: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
    con.commit()
    con.close()

def list_notes():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, title, created_at FROM notes ORDER BY created_at DESC")
    rows = cur.fetchall()
    con.close()
    return rows

# -------------- Vector Store --------------
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    col = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return col

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ingest_file_to_chroma(client: OpenAI, file_bytes, file_name: str):
    text = read_upload(file_bytes, file_name)
    chunks = chunk_text(text)
    embeddings = embed_texts(client, chunks)
    col = get_chroma_collection()  # <- SIEMPRE kb01
    ids = [f"{file_name}-{i}" for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=[{"source": file_name}] * len(chunks))
    return len(chunks)

def retrieve_context(client: OpenAI, query: str, top_k: int = 4) -> List[str]:
    col = get_chroma_collection()  # <- SIEMPRE kb01
    q_emb = embed_texts(client, [query])[0]
    res = col.query(query_embeddings=[q_emb], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    return docs

def call_llm(client: OpenAI, question: str, context_chunks: List[str]) -> str:
    system = "You are a helpful assistant that answers using only the provided context when relevant. If the context is insufficient, say so and ask for more info."
    context_text = "\n\n".join([f"[Context #{i+1}]\n{c}" for i,c in enumerate(context_chunks)])
    user = f"Question: {question}\n\nUse the following context if it helps:\n{context_text}"
    resp = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content

# -------------- Streamlit UI --------------
st.set_page_config(page_title="Mini RAG Assistant")
st.title("Mini AI Knowledge Assistant (RAG + Tool)")

# 👇 Añade esta línea justo después del st.title()
st.caption(f"Vector store: {CHROMA_DIR} | Collection: {COLLECTION_NAME}")


if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in your environment to use this app. See README.md")
else:
    client = OpenAI()

# Sidebar: Upload & Notes
with st.sidebar:
    st.header("📂 Knowledge Base")
    up = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
    if st.button("Ingest file to KB") and up and OPENAI_API_KEY:
        bytes_data = up.read()
        n_chunks = ingest_file_to_chroma(client, bytes_data, up.name)
        st.success(f"Ingested {n_chunks} chunks from {up.name}")

    st.divider()
    st.header("🛠️ Tool: Notes")
    ensure_db()
    with st.form("save_note_form"):
        note_title = st.text_input("Title", value="Untitled")
        note_content = st.text_area("Content")
        submitted = st.form_submit_button("Save Note")
        if submitted:
            save_note(note_title, note_content)
            st.success("Note saved.")

    st.caption("This satisfies the 'custom tool' requirement by saving notes to a local SQLite DB.")

    st.divider()
    st.subheader("📒 Your Notes")
    rows = list_notes()
    if rows:
        for rid, title, ts in rows[:10]:
            st.write(f"• #{rid} — **{title}** ({ts})")
    else:
        st.write("_No notes yet._")

# Main chat
st.subheader("💬 Chat with your Knowledge Base")
question = st.text_input("Ask a question about your uploaded content:")
top_k = st.slider("Retrieve top-k context", 1, 8, 4)

if st.button("Ask") and question and OPENAI_API_KEY:
    with st.spinner("Thinking..."):
        ctx = retrieve_context(client, question, top_k=top_k)
        answer = call_llm(client, question, ctx)

        # Show the context (for debugging)
        st.markdown("### Retrieved Context")
        st.write(ctx)

        # Show the final answer
        st.markdown("### Answer")
        st.write(answer)

        # convenience: prefill note content with answer
        st.session_state["prefill_note"] = answer
        st.info("Tip: You can save this answer as a note from the sidebar.")

st.caption("Powered by OpenAI + ChromaDB. Store: local persistence. Tool: SQLite notes.")
