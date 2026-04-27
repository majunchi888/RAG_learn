import uuid
from typing import List
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
llm = ChatDeepSeek(api_key=deepseek_api_key, model="deepseek-chat", temperature=0.1, max_tokens=1000)

def simple_summary(text: str) -> str:
    return llm.invoke("请把{text}进行总结提炼").content

class SummaryVectorStore:
    def __init__(self):
      self.model = SentenceTransformer("shibing624/text2vec-base-chinese")
      self.index = None

      self.docstore = {}
      self.index_to_docid = {}

    def add_documents(self, documents: List[str]):
      summaries = []
      doc_ids = []

      for doc in docs:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        summary = simple_summary(doc)

        summaries.append(summary)#总结
        doc_ids.append(doc_id)#id

        self.docstore[doc_id] = doc#对应原文

      embeddings =  self.model.encode(summaries).astype("float32")
      dim = embeddings.shape[1]

      self.index = faiss.IndexFlatL2(dim)
      self.index.add(embeddings)

      for i, doc_id in enumerate(doc_ids):
        self.index_to_docid[i] = doc_id

    def query(self, query: str, top_k = 3):
      query_vec = self.model.encode([query]).astype("float32")    

      D, I = self.index.search(query_vec, top_k)

      results = []
      for idx in I[0]:
        doc_id = self.index_to_docid[idx]
        original_doc = self.docstore[doc_id]

        results.append({
            "doc_id": doc_id,
            "content": original_doc
        })
      return results    

docs = [
    "Python is a programming language used for AI and backend development.",
    "FAISS is a library for efficient similarity search of vectors.",
    "RAG combines retrieval with large language models.",
    "Transformers are neural networks used in NLP tasks."
]

store = SummaryVectorStore()
store.add_documents(docs)

query = "What is python?"

results = store.query(query)

for r in results:
    print(r["doc_id"], "->", r["content"])    