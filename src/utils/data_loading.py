from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import WebBaseLoader
from langchain.schema.document import Document
from typing import List


def load_pdf(file_path: str) -> List[Document]:
    pdf_data = PyPDFLoader(file_path).load()
    return pdf_data


def load_csv(file_path: str) -> List[Document]:
    csv_data = CSVLoader(file_path).load()
    return csv_data


def load_web_data(links: List[str]) -> List[Document]:
    web_data = WebBaseLoader(links).load()
    return web_data
