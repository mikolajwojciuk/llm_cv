import torch
from constants import EMBEDDING_MODEL

from langchain.embeddings import HuggingFaceEmbeddings


def get_embedding_model(model_id: str = EMBEDDING_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_id, model_kwargs={"device": device}
    )

    return embedding_model
