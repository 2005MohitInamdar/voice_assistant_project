import os
import json
from typing import List


try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import FastEmbedEmbeddings

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        # Don't require HF_TOKEN when running locally with the fallback.
        # We'll still try to use the real embeddings when a token exists.
        HF_TOKEN = None

    VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

    if HF_TOKEN:
        embeddings = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    else:
        embeddings = None

    vector_db = Chroma(
        collection_name="assistant_memory",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    def add_to_memory(user_text: str, assistant_text: str):
        text = f"User: {user_text}\nAssistant: {assistant_text}"
        vector_db.add_texts(
            texts=[text],
            metadatas=[{"user": user_text, "assistant": assistant_text}]
        )
        vector_db.persist()
        print(f"Memory saved: '{user_text[:30]}...'")

    def retrieve_memory(query: str, k: int = 3):
        results = vector_db.similarity_search(query, k=k)
        if not results:
            return ""
        memories = "\n\n".join([doc.page_content for doc in results])
        return f"Previous conversations:\n{memories}\n\n"

    def clear_memory():
        vector_db.delete_collection()
        print("All memory cleared!")

except Exception:
    print("⚠️  langchain_community not available, using lightweight local JSON memory fallback.")

    _MEMORY_FILE = os.path.join(os.path.dirname(__file__), "_memory.json")
    if not os.path.exists(_MEMORY_FILE):
        with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

    def _load_memory() -> List[dict]:
        try:
            with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_memory(mem: List[dict]):
        with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)

    def add_to_memory(user_text: str, assistant_text: str):
        mem = _load_memory()
        text = f"User: {user_text}\nAssistant: {assistant_text}"
        mem.append({"user": user_text, "assistant": assistant_text, "text": text})
        _save_memory(mem)
        print(f"Memory saved (fallback): '{user_text[:30]}...'")

    def retrieve_memory(query: str, k: int = 3):
        mem = _load_memory()
        if not mem:
            return ""
        # Naive retrieval: return the last `k` interactions
        last = mem[-k:]
        memories = "\n\n".join([item["text"] for item in last])
        return f"Previous conversations:\n{memories}\n\n"

    def clear_memory():
        _save_memory([])
        print("All memory cleared (fallback)!")
