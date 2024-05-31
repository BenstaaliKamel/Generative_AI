import glob
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader

load_dotenv(find_dotenv())

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Automatically detect all files in the folder according to types

def extract_files_from_folder(folder_path, file_type):
    return list(Path(folder_path).rglob(file_type))

def process_files(files, loader_class, text_splitter, embeddings, index_name, namespace):
    for file_path in files:
        loader = loader_class(str(file_path))
        doc_file = loader.load()
        doc_file_split = text_splitter.split_documents(doc_file)
        Pinecone.from_documents(doc_file_split, embeddings, index_name=index_name, namespace=namespace)

@st.cache_resource
def load_split_each_file(index_name='portailassociations'):

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Extract files
    pdf_files = extract_files_from_folder('data/', '*.pdf')
    csv_files = extract_files_from_folder('data/', '*.csv')
    xlsx_files = extract_files_from_folder('data/', '*.xlsx')
    txt_files = extract_files_from_folder('data/', '**/*.txt')

    # Process text files
    loader = DirectoryLoader('data/', glob='**/*.txt', show_progress=True)
    doc_file = loader.load()
    doc_file_split = text_splitter.split_documents(doc_file)
    doc_db = Pinecone.from_documents(doc_file_split, embeddings, index_name=index_name, namespace='txt')

    # Process other file types
    process_files(pdf_files, PyPDFLoader, text_splitter, embeddings, index_name, 'pdf')
    process_files(csv_files, CSVLoader, text_splitter, embeddings, index_name, 'csv')
    process_files(xlsx_files, UnstructuredExcelLoader, text_splitter, embeddings, index_name, 'excel')

    return doc_db

# Initializing the LLM model
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
doc_db = load_split_each_file()  # Will be the Pinecone index filled with embeddings

#Function to answer with gpt and the data
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
