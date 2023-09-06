import torch
from constants import EMBEDDING_MODEL
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from typing import List


def get_embedding_retriever(docs: List[Document]):
    device = "cuda" if torch.cuda.isavailable() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": device}
    )
    embeddings_retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    return embeddings_retriever
