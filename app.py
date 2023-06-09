from typing import Set
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain.embeddings import OpenAIEmbeddings
from samples.fullApp.backend.core import run_llm
from samples.fullApp.utility.vectorStore import vectorize_doc, load_vectorized, save_pdf
import streamlit as st
from streamlit_chat import message


def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])


def main():
    st.set_page_config(page_title="Ask PDF")
    st.header("Ask your PDF...")
    if (
        "chat_answers_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
    ):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []

    pdf = st.file_uploader("Upload your PDF:", type="pdf")

    if pdf is not None:
        save_pdf("./localVS/myReAct.pdf", pdf)
        embeddings = get_embeddings()
        vectorize_doc(embeddings, "./localVS/myReAct.pdf")
        # remove saved pdf

    prompt = st.text_input(
        "Prompt", placeholder="Enter your message here..."
    ) or st.button("Submit")
    if prompt:
        with st.spinner("Generating response.."):
            embeddings = get_embeddings()
            vectoreStore = load_vectorized(embeddings)
            generated_response = run_llm(
                vectoreStore,
                query=prompt,
                chat_history=st.session_state["chat_history"],
            )

            formatted_response = f"{generated_response['answer']}"
            st.session_state.chat_history.append((prompt, generated_response["answer"]))
            st.session_state.user_prompt_history.append(prompt)
            st.session_state.chat_answers_history.append(formatted_response)

    if "chat_answers_history" in st.session_state:
        if st.session_state["chat_answers_history"]:
            for generated_response, user_query in zip(
                st.session_state["chat_answers_history"],
                st.session_state["user_prompt_history"],
            ):
                message(
                    user_query,
                    is_user=True,
                )
                message(generated_response)


if __name__ == "__main__":
    main()
