import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from samples.fullApp.utility.consts import vsDir


def save_pdf(filename, pdf):
    filebytes = pdf.read()
    with open(filename, "wb") as f:
        f.write(filebytes)


def vectorize_doc(embeddings, pdf_path, index_name=vsDir) -> None:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents=documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_name)


def load_vectorized(embeddings, index_name=vsDir):
    if os.path.isdir(index_name):
        return FAISS.load_local(index_name, embeddings)


if __name__ == "__main__":
    pdf_path = "./papers/react.pdf"
    vectorize_doc(pdf_path)
    vs = load_vectorized()
    print(vs)
