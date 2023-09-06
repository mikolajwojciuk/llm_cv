from constants import LLM
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.document import Document
from src.utils.embeddings_db import get_embedding_retriever
from typing import List


def _get_model(model_id: str = LLM):
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", offload_folder="."
    ).eval()

    return tokenizer, model


def get_generation_pipeline(
    docs: List[Document],
    model_id: str = LLM,
    temperature: float = 0.1,
    repetition_penalty: float = 1.05,
):
    tokenizer, model = _get_model(model_id=model_id)
    embeddings_retriever = get_embedding_retriever(docs)

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task="text-generation",
        temperature=temperature,  # generation parameter resposible for output sampling
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=repetition_penalty,  # penalty for repeating tokens
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=generate_text, model_id=model_id)

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        embeddings_retriever,
        return_source_documents=True,
        max_tokens_limit=3500,
    )

    return chain
