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




def doc_preprocessing():
    loader = PyPDFLoader("TICADI - Spécifications - 2024-04-10 - WorkFlowy.pdf")
    #loader = DirectoryLoader(
    #    'data/',
    #    glob='**/*.pdf',  # only the PDFs
    #    show_progress=True
    #)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split


@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    #pc = Pinecone()
    #index = pc.index('kameltrain')
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(docs_split, embeddings, index_name = 'kameltrainvectors')
    return doc_db


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
doc_db = embedding_db() #will be the pinecone index filled with embeddings


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


# Function to simulate the chatbot response
def chatbot_response(input_text):
    # Simple echo response for demonstration purposes
    return "Chatbot: " + input_text

# Define the Streamlit app
def main():
    st.title("Chatbot Messaging App")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Load existing conversation history from file if available
    conversation_file_path = "conversation_history.txt"
    try:
        with open(conversation_file_path, "r") as file:
            st.session_state.conversation_history = file.readlines()
    except FileNotFoundError:
        pass

    # Display conversation history
    for item in st.session_state.conversation_history:
        if "user" in item:
            item_display = item.replace("user: ","")
            st.info(item_display)
        elif "bot" in item:
            item_answer_display = item.replace("bot: Chatbot:", "")
            st.success(item_answer_display)

    # User input box
    user_input = st.chat_input("Ask something")

    # Check if user input is not empty and Enter key is pressed
    if user_input:
        # Get chatbot response
        bot_response = chatbot_response(retrieval_answer(user_input))

        # Add user's message to the conversation history
        st.session_state.conversation_history.append("user: " + user_input)

        # Add chatbot's response to the conversation history
        st.session_state.conversation_history.append("bot: " + bot_response)

        # Display the latest user input and chatbot response above the input field
        st.info(user_input)
        st.success(bot_response)

        # Update conversation history file
        with open(conversation_file_path, "a") as file:
            file.write("user: " + user_input + "\n")
            file.write("bot: " + bot_response + "\n")

# Run the Streamlit app
if __name__ == "__main__":
    main()
