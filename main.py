import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import tempfile
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import torch
from langchain.embeddings.openai import OpenAIEmbeddings  # Corrected import
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration 
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI instead of ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import openai
import streamlit as st
import requests
import base64
from PyPDF2 import PdfReader
import numpy as np
from PIL import Image
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISIONCHAT_API_KEY = os.getenv("visionchat_api")

# Initialize OpenAI client
# client = openai.Completion.create(
#     base_url=API_BASE_URL,
#     api_key=API_KEY
# )

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def Chat_with_document_using_RAG():
    st.title("Chat with Document using RAG")
    st.write("Chat with your uploaded documents!")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        st.success("Document uploaded successfully!")
        with st.spinner("Processing the document..."):
            with open("uploaded_document.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Load the document using PyPDFLoader
            loader = PyPDFLoader("uploaded_document.pdf")
            documents = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text_chunks = text_splitter.split_documents(documents)

            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
           
            vector_store = FAISS.from_documents(text_chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})

            # Set up memory
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Load the conversational model (using ChatOpenAI)
            model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.5)

            # Create the retrieval chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=model,
                retriever=retriever,
                memory=memory
            )

        st.write("You can now ask questions about your document!")
        user_input = st.text_input("Ask a question about the document:")
        if st.button("Get Answer"):
            if user_input.strip():
                with st.spinner("Generating answer..."):
                    # Generate the response using the chain
                    result = chain.invoke({"question": user_input, "chat_history": []})
                    st.write("**Answer:**", result["answer"])
            else:
                st.error("Please enter a question.")


#LLMS
def Meeting_Notes_Summarizer_using_LLMS():
    """
    This function combines the entire process of summarizing meeting notes into a single function.
    It integrates loading the model, preprocessing input, generating a summary, and displaying the result via Streamlit.
    """

    # Step 2: Load Pre-trained Model and Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Step 3: Preprocess Input Text
    def preprocess_input(text):
        """
        Prepares the input text for the model by encoding it.
        The input text is prepended with "summarize: " to specify the task.
        """
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        return inputs

    # Step 4: Generate Summary using T5
    def generate_summary(text):
        """
        Generates a summary for the input text using the T5 model.
        The summary is decoded and split into bullet points.
        """
        # Preprocess input text
        inputs = preprocess_input(text)

        # Generate summary using the model
        summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Split the summary into bullet points
        summary_points = summary.split(". ")
        return summary_points

    # Step 5: Initialize Streamlit UI for Summarizing Meeting Notes
    st.title("Meeting Notes Summarizer")
    st.write("This web app summarizes your meeting notes into concise bullet points.")

    # Step 6: Input Area for Meeting Notes
    user_input = st.text_area("Enter Meeting Notes:")

    # Button to trigger summarization
    if st.button("Summarize Meeting Notes"):
        if user_input:
            # Display the input text first
            st.write("### Original Meeting Notes:")
            st.write(user_input)

            # Generate the summary and display it
            summary_points = generate_summary(user_input)
            st.write("### Summarized Meeting Notes:")

            # Display the bullet points
            for i, point in enumerate(summary_points, 1):
                st.write(f"• {point}")
        else:
            st.write("Please enter meeting notes to summarize.")

# Main Streamlit app
def main():
    # Sidebar configuration
    st.sidebar.title("Choose a Functionality")
    st.sidebar.write("Select an option below to explore different features of the app:")
    
    # Sidebar options
    options = [
        "Chat with Document using RAG",
        "Meeting Notes Summarizer",
    ]
    choice = st.sidebar.radio("Select an option:", options)
    if choice == "Chat with Document using RAG":
        Chat_with_document_using_RAG()

    elif choice == "Meeting Notes Summarizer":
        Meeting_Notes_Summarizer_using_LLMS()

if __name__ == "__main__":
    main()
