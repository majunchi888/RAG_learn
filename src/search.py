import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorstore
from langchain_deepseek import ChatDeepSeek

load_dotenv()

class RAGSearch:
  def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "shibing624/text2vec-base-chinese", chunk_size: int = 200, chunk_overlap: int = 20, llm_model: str = "deepseek-chat"):
    self.vectorstore = FaissVectorstore(persist_dir,embedding_model)
    
    faiss_path = os.path.join(self.vectorstore.persist_dir, "faiss_index")
    meta_path = os.path.join(self.vectorstore.persist_dir, "metadata.pkl")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
      from data_loader import load_all_documents
      docs = load_all_documents("data")
      self.vectorstore.build_from_documents(docs)
    else:  
      self.vectorstore.load()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    self.llm = ChatDeepSeek(model=llm_model, api_key=deepseek_api_key)
    print(f"[Debug] Using model: {llm_model} for generating responses")

  def search_and_summarize(self, query: str, top_k: int = 3) -> str:
    results = self.vectorstore.query(query, top_k=top_k)
    texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
    context = "\n".join(texts)
    if not context:
      return "无法找到相关内容"
    prompt = f"请根据以下内容回答问题:\n{context}\n问题是:{query}"  
    response = self.llm.invoke(prompt)
    return response.content


