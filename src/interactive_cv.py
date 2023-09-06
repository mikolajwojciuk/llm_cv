# noqa: E501
from src.utils.model import get_tokenizer_and_model
from src.utils.data_loading import load_csv, load_pdf, load_web_data
from src.utils.embeddings import get_embedding_model
from typing import List
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
import transformers
from langchain.vectorstores import FAISS
from src.utils.constants import LLM, EMBEDDING_MODEL


class InteractiveCv:
    def __init__(self, llm_id: str = LLM, embedding_model_id: str = EMBEDDING_MODEL):
        self.tokenizer, self.model = get_tokenizer_and_model(model_id=llm_id)
        self.embedding_model = get_embedding_model(model_id=embedding_model_id)
        self.llm_id = llm_id
        self.embedding_model_id = embedding_model_id
        self.documents = []
        self.chat_history = []

    def add_csv_file(self, file_path: str):
        self.documents += load_csv(file_path)

    def add_pdf_file(self, file_path: str):
        self.documents += load_pdf(file_path)

    def add_web_data(self, links: List[str]):
        self.documents += load_web_data(links)

    def __call__(self, question: str):
        if not self.chain:
            self.init_chain()

        if question:
            result = self.chain(
                {"question": question, "chat_history": self.chat_history}
            )["answer"].lstrip()
            self.chat_history.append((question, result))
            return result

        return "Sorry, I do not know the answer."

    def init_chain(self):
        embeddings_retriever = FAISS.from_documents(
            self.documents, self.embedding_model
        ).as_retriever()

        pipeline = self._get_text_generation_pipeline()

        llm = HuggingFacePipeline(pipeline=pipeline, model_id=self.llm_id)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm,
            embeddings_retriever,
            return_source_documents=True,
            max_tokens_limit=3500,
        )

    def _get_text_generation_pipeline(self):
        text_generation_pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task="text-generation",
            temperature=0.1,  # generation parameter resposible for output sampling
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.05,  # penalty for repeating tokens
            do_sample=True,
        )
        return text_generation_pipeline

    def _preprocess_documents(
        self, chunk_size: int = 500, chunk_overlap: int = 25
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.processed_documents = text_splitter.split_documents(self.documents)
