import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorstore:
  def __init__(self,persist_dir: str = "faiss_store", embedding_model: str = "shibing624/text2vec-base-chinese", chunk_size: int = 200, chunk_overlap: int = 20):
    self.persist_dir = persist_dir
    os.makedirs(self.persist_dir, exist_ok=True)
    self.index = None
    self.metadata = []
    self.embedding_model = embedding_model
    self.model = SentenceTransformer(embedding_model)
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    print(f"[Debug] Using model: {embedding_model} for vectorizing and searching")

  def build_from_documents(self, documents: List[Any]):  
    print(f"[Debug] Building vector store from {len(documents)} documents")
    emb_pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    chunks = emb_pipeline.chunk_documents(documents)
    embeddings = emb_pipeline.embeded_chunks(chunks)
    metadata = [{"text": chunk.page_content} for chunk in chunks]
    self.add_embeddings(np.array(embeddings).astype('float32'), metadata)
    self.save()
    print(f"[Info] Vector store built and saved to: {self.persist_dir}")

  def add_embeddings(self, embeddings: np.ndarray, metadata: List[Any] = None):
    dim = embeddings.shape[1]
    if self.index is None:  
      self.index = faiss.IndexFlatL2(dim)
    self.index.add(embeddings)
    if metadata:
      self.metadata.extend(metadata)
    print(f"[Debug] Added {embeddings.shape[0]} embeddings to vector store")  
    
  def save(self):
    faiss_path = os.path.join(self.persist_dir, "faiss_index")
    meta_path = os.path.join(self.persist_dir, "metadata.pkl")
    faiss.write_index(self.index, faiss_path)
    with open(meta_path, "wb") as f:
      pickle.dump(self.metadata, f)
    print(f"[Info] Saved Faiss index and metadata to: {self.persist_dir}") 

  def load(self):
    faiss_path = os.path.join(self.persist_dir, "faiss_index")
    meta_path = os.path.join(self.persist_dir, "metadata.pkl")
    self.index = faiss.read_index(faiss_path)
    with open(meta_path, "rb") as f:
      self.metadata = pickle.load(f)
    print(f"[Info] Loaded Faiss index and metadata from: {self.persist_dir}")

  def search(self, query_embedding: np.ndarray,top_k: int = 5):
    D, I = self.index.search(query_embedding, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
      meta = self.metadata[idx] if idx < len(self.metadata) else None
      results.append({"index": idx, "distance": dist, "metadata": meta})
    return results  

  def query(self, query_text: str, top_k: int = 3):
    
    print(f"[Debug] Querying vector store with: {query_text}")
    query_embedding = self.model.encode([query_text]).astype('float32')
    return self.search(query_embedding, top_k=top_k)
