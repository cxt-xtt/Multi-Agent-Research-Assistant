import chromadb
import os
from typing import List, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")

class KnowledgeBase:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=DATA_DIR)

    def get_collection(self, user_id: str):
        safe_name = f"kb_{user_id.replace('-','_').replace(' ','_')}"
        return self.client.get_or_create_collection(name=safe_name)

    def add_doc(self, user_id: str, doc_id: str, content: str, metadata: dict = None):
        col = self.get_collection(user_id)
        col.add(documents=[content], metadatas=[metadata or {}], ids=[doc_id])

    def search(self, user_id: str, query: str, top_k: int = 3) -> List[str]:
        col = self.get_collection(user_id)
        results = col.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            name = meta.get("file_name", f"片段{i+1}")
            parts.append(f"[知识库·{name}]\n{doc}")
        return parts

kb = KnowledgeBase()
