from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorstore
from src.search import RAGSearch

if __name__ == "__main__":
  # docs = load_all_documents("data")
  store = FaissVectorstore("faiss_store")
  # store.build_from_documents(docs)
  store.load()

  rag_search = RAGSearch()
  query = "介绍马俊驰"
  summary = rag_search.search_and_summarize(query, top_k=3)
  print("Summary:", summary)
  
  # print(chunkvectors)
  