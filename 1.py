import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # text splitter
from sentence_transformers import SentenceTransformer # model enable 
import faiss
import pickle
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv()

# 1 document loaders 
def document_loaders(data_path: str ) -> List[Any]:
  file_path = Path(data_path).resolve()
  if not file_path.exists():
    raise print(f"not found {file_path}")

  documents = []  

#PDF loader
  pdf_paths = list(file_path.glob("**/*.pdf")) #不转list的话是个generator生成器
  
  for pdf_path in pdf_paths:
    try:
     pdf_loader = PyPDFLoader(str(pdf_path)) # pdf_path 是 pathlib.Path 对象
     loaded = pdf_loader.load()
     documents.extend(loaded)
     print(f"loaded {pdf_path}")

    except Exception as e:
        print(f"Fail to load PDF from {pdf_path} : {e}")

  return documents  


# 2 chunk and embedding
class EmbeddingPipeline:
  def __init__(self, model_name = "shibing624/text2vec-base-chinese", chunk_size: int = 100, chunk_overslap: int = 20):
    self.model = SentenceTransformer(model_name)
    self.chunk_size = chunk_size
    self.chunk_overslap = chunk_overslap
    print(f"[Debug] Using model: {model_name}")

  def chunk_document(self, documents: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
                 chunk_size=self.chunk_size,
                 chunk_overlap=self.chunk_overslap,
                 length_function=len,
                 separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    # print(chunks)
    print(f"[Debug] Chunked {len(documents)} documents into {len(chunks)} chunks")
    return chunks

  def embeded_chunks(self, chunks : List[Any]) -> np.ndarray:
    embeddings =  self.model.encode([chunk.page_content for chunk in chunks] ,show_progress_bar=True ) # document由page_content和metadata组成
    print(f"[Debug] Generated embeddings for {len(chunks)} chunks, shape: {embeddings.shape}")
    return embeddings


class FaissVectorStore:
  def __init__(self, persist_dir: str = "faiss_store1", embedding_model: str = "shibing624/text2vec-base-chinese", chunk_size: int = 200, chunk_overslap: int = 20):
    self.persist_dir = persist_dir
    os.makedirs(persist_dir, exist_ok=True)
    self.model = SentenceTransformer(embedding_model)
    self.embedding_model = embedding_model
    self.chunk_size = chunk_size
    self.chunk_overslap = chunk_overslap
    self.index = None
    self.metadata = []
    print(f"[Debug] Using model: {embedding_model} for vectorizing and searching")

  def build_documents(self, documents: List[Any]):
    print(f"[Debug] Building vector store from {len(documents)} documents")
    emb_pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overslap=self.chunk_overslap)

    chunks = emb_pipeline.chunk_document(documents)
    embeddings = emb_pipeline.embeded_chunks(chunks)
    metadata = [{"text" : chunk.page_content} for chunk in chunks]
    self.add_embeddings(np.array(embeddings), metadata)
    self.save()
    print(f"[Info] Vector store built and saved to: {self.persist_dir}")

  def add_embeddings(self, embeddings: np.array, metadata: List[Any] = None):
      dim = embeddings.shape[1]
      if self.index is None:
        self.index = faiss.IndexFlatL2(dim)  # L2 distance 创建空的索引 ，add , search, remove
      self.index.add(embeddings)  
      
      if metadata:
        self.metadata.extend(metadata)
      print(f"[Debug] Added {embeddings.shape[0]} embeddings to vector store")  

  def save(self):
    faiss_path = os.path.join(self.persist_dir, "faiss_index")  # 字符串拼接成路径
    metadata_path = os.path.join(self.persist_dir, "matedata_pkl")

    faiss.write_index(self.index, faiss_path) # faiss的方法write_index 将 索引写入 

    with open(metadata_path, "wb") as f: # 以二进制写入 with 写完了自动关闭 ，f是文件对象
      pickle.dump(self.metadata, f) # pickle.dump 将metadata 序列化并写入文件
    print(f"[Info] Saved Faiss index and metadata to: {self.persist_dir}") 

  def load(self):  
    faiss_path = os.path.join(self.persist_dir, "faiss_index")
    metadata_path = os.path.join(self.persist_dir, "matedata_pkl")

    self.index = faiss.read_index(faiss_path) # faiss的方法read_index 读取索引
    with open(metadata_path, "rb") as f:
      self.metadata = pickle.load(f) # pickle.load 反序列化
    print(f"[Info] Loaded Faiss index and metadata from: {self.persist_dir}")

  def search(self, query_embedding: np.ndarray, top_k: int = 3):
    D,I = self.index.search(query_embedding, top_k) # 返回距离和索引 
    results = []
    for idx, distance in zip(I[0],D[0]): # 遍历索引和距离 
      meta = self.metadata[idx] if idx < len(self.metadata) else None
      results.append({"index": idx, "distance": distance, "metadata": meta})
      return results

  def query(self, query_text: str, top_k: int = 3):
    query_embedding = self.model.encode([query_text]).astype('float32')
    return self.search(query_embedding, top_k = top_k)


class RAGSearch:
  def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "shibing624/text2vec-base-chinese", chunk_size: int = 200, chunk_overlap: int = 20, llm_model: str = "deepseek-chat"):
    self.vectorstore = FaissVectorStore(persist_dir,embedding_model)
    faiss_path = os.path.join(self.vectorstore.persist_dir, "faiss_index")
    meta_path = os.path.join(self.vectorstore.persist_dir, "metadata.pkl")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
      docs = document_loaders("data")
      self.vectorstore.build_documents(docs) 
    else:  
      self.vectorstore.load()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    self.llm = ChatDeepSeek(model=llm_model, api_key=deepseek_api_key)
    print(f"[Debug] Using model: {llm_model} for generating responses")

  def search_and_summarize(self, query: str, top_k: int = 3):
    results = self.vectorstore.query(query, top_k=top_k)
    texts = [r["metadata"].get("text", "") for r in results if r["metadata"]] # r["metadata"]是一个字典，get("text", "")是一个字符串
    context = "\n".join(texts)
    if not context:
      return "无法找到相关内容"
    prompt = f"请根据以下内容回答问题:\n{context}\n问题是:{query}"  
    response = self.llm.invoke(prompt)
    return response.content  


if __name__ == "__main__":
  docs = document_loaders("data")
  store = FaissVectorStore("faiss_store")
  store.build_documents(docs)
  store.load()

  rag_search = RAGSearch()
  query = "介绍马俊驰"
  summary = rag_search.search_and_summarize(query, top_k=3)
  print("Summary:", summary)


# doc = document_loaders("data")

# embedding_document = EmbeddingPipeline()
# embedding_document.chunk_document(doc)
