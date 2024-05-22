import glob

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as pc
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
load_dotenv(find_dotenv())

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
doc_db = Pinecone.from_documents('', embeddings, index_name = 'kameltrainvectors')


def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result


def main():
    st.title("Question and Answering App powered by LLM and Pinecone")

    text_input = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)


if __name__ == "__main__":
    main()








