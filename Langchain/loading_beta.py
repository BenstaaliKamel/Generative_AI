import glob

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
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


def extract_text_from_pdfs(pdf_folder):
    """
    Automatically detect PDF files in a folder, create a PyPDFLoader for each, and extract the text.

    Args:
        pdf_folder (str): The folder path containing PDF files.

    Returns:
        pdf_files
    """
    # Automatically detect all PDF files in the folder
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    print(pdf_files)
    return pdf_files

def extract_text_from_csv(csv_folder):
    """
    Automatically detect CSV files in a folder.

    Args:
        csv_folder (str): The folder path containing CSV files.

    Returns:
        csv_files (list): A list of paths to the detected CSV files.
    """
    # Automatically detect all CSV files in the folder
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    print(csv_files)
    return csv_files

def extract_text_from_xlsx(xlsx_folder):
    """
    Automatically detect XLSX files in a folder.

    Args:
        xlsx_folder (str): The folder path containing XLSX files.

    Returns:
        xlsx_files (list): A list of paths to the detected XLSX files.
    """
    # Automatically detect all XLSX files in the folder
    xlsx_files = glob.glob(os.path.join(xlsx_folder, "*.xlsx"))
    print(xlsx_files)
    return xlsx_files

@st.cache_resource
def load_split_each_file():
    """
    Automatically does the preprocess for each file , the preprocess will be about creatinga loader , spliting into chunks, adding in pinecone index

    Args:
        none

    Returns:
        doc_db: the last pinecone intialization in the loup so we use it for retrieval
    """
# Loop through each PDF file path and create a loader
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    pdf_files = extract_text_from_pdfs('data/')
    csv_files = extract_text_from_csv('data/')
    xlsx_files = extract_text_from_xlsx('data/')
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        doc_file = loader.load()
        doc_file_split = text_splitter.split_documents(doc_file)
        doc_db = Pinecone.from_documents(doc_file_split, embeddings, index_name = 'kameltrainvectors' , namespace = 'pdf')
    for csv_file in csv_files:
        loader = CSVLoader(file_path=csv_file)
        doc_file = loader.load()
        doc_file_split = text_splitter.split_documents(doc_file)
        doc_db = Pinecone.from_documents(doc_file_split, embeddings, index_name='kameltrainvectors', namespace = 'csv')
    for xlsx_file in xlsx_files:
        loader = UnstructuredExcelLoader(xlsx_file)
        doc_file = loader.load()
        doc_file_split = text_splitter.split_documents(doc_file)
        doc_db = Pinecone.from_documents(doc_file_split, embeddings, index_name = 'kameltrainvectors', namespace = 'excel')

    return doc_db


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
doc_db = load_split_each_file() #will be the pinecone index filled with embeddings


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








