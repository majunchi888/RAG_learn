from typing import List , Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
  def __init__(self, model_name: str = "shibing624/text2vec-base-chinese", chunk_size: int = 200, chunk_overlap: int = 20):
    self.model = SentenceTransformer(model_name)
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    print(f"[Debug] Using model: {model_name}") 

  def chunk_documents(self, documents: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
    length_function=len,
    separators = ["\n\n", "\n", " ", ""]
    )  
    chunks = splitter.split_documents(documents)
    print(f"[Debug] Chunked {len(documents)} documents into {len(chunks)} chunks")
    return chunks

  def embeded_chunks(self, chunks: List[Any]) -> np.ndarray:
    embeddings = self.model.encode([chunk.page_content for chunk in chunks],show_progress_bar=True)  
    print(f"[Debug] Generated embeddings for {len(chunks)} chunks, shape: {embeddings.shape}")
    return embeddings