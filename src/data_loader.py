from pathlib import Path
from typing import Any, List
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader



def load_all_documents(data_dir: str) -> list[Any]:
  """load all supported from data directory and concert to Langchain document strcture"""

  file_path = Path(data_dir).resolve()
  if not file_path.exists():
    raise FileNotFoundError(f"Data directory not found: {file_path}")
  print(f"[Debug] Loading data from: {file_path}")
  documents = []

  # PDFloader
  pdf_files = list(file_path.glob("**/*.pdf"))
  print(f"[Debug] Found {len(pdf_files)} PDF files. PDF files: {[str(f) for f in pdf_files]}")
  for pdf_file in pdf_files:
    try:
      print(f"[Debug] Loading PDF: {pdf_file}")
      loader = PyPDFLoader(str(pdf_file))
      loaded = loader.load()
      documents.extend(loaded)

    except Exception as e:
      print(f"[Error] Failed to load PDF {pdf_file}: {e}")

  # TextLoader
    text_files = list(file_path.glob("**/*.txt"))
    print(f"[Debug] Found {len(text_files)} text files. Text files: {[str(f) for f in text_files]}")
    for text_file in text_files:
      try:
        print(f"[Debug] Loading text: {text_file}")
        loader = TextLoader(str(text_file),encoding="utf-8")
        loaded = loader.load()
        documents.extend(loaded)

      except Exception as e:
        print(f"[Error] Failed to load text {text_file}: {e}")

  # csvLoader

  return documents


