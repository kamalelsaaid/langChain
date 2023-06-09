import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import VectorDBQA, OpenAI


if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("./samples/qa_from_txt/mediumblog.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    db = Chroma.from_documents(texts, embeddings)

    # query = "What did the president say about Ketanji Brown Jackson"

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=db, return_source_documents=False
    )
    query = "What is a vector DB? Give me a 15 word answer for a begginner"
    # docs = db.similarity_search(query)
    # print(docs)
    # docs = db.similarity_search_with_score(query)
    # print(docs)
    result = qa({"query": query})
    print(result)
