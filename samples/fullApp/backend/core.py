import os
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


INDEX_NAME = "langchain-doc-index"


def run_llm(vectoreStore, query: str, chat_history: List[Dict[str, Any]] = []):
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vectoreStore.as_retriever(), return_source_documents=False
    )
    return qa({"question": query, "chat_history": chat_history})
