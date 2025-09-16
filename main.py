import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

# --- Modern LangChain Imports ---
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# CHANGED: Import the Gemini chat model instead of Groq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env file
load_dotenv()
# CHANGED: Load the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Streamlit UI Setup ---
st.title("RockyBot: News Research Tool 📈 (Gemini API)")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# --- LLM Initialization ---
# CHANGED: Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True # Helps with compatibility for some chains
)

# --- Check for Session State Initialization ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if process_url_clicked:
    # Remove empty URLs
    urls = [url for url in urls if url.strip()]
    if not urls:
        st.sidebar.error("Please provide at least one URL.")
    else:
        try:
            # Step 1: Load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading... Started ✅✅✅")
            data = loader.load()

            # Step 2: Split text
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=400,
                chunk_overlap=100
            )
            main_placeholder.text("Text Splitting... Started ✅✅✅")
            docs = text_splitter.split_documents(data)

            # Step 3: Create embeddings & FAISS vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Store... Building ✅✅✅")
            time.sleep(2)

            # Save retriever to session state
            st.session_state.retriever = vectorstore.as_retriever()
            main_placeholder.success("URLs Processed! You can now ask questions.")
        
        except Exception as e:
            main_placeholder.error(f"An error occurred: {e}. Please check your URLs.")

# --- The Modern RAG Chain (LCEL) ---
prompt = ChatPromptTemplate.from_template("""
Answer the user's question based ONLY on the following context.
If the answer is not in the context, say you don't know.
Provide a detailed answer and list the sources used.

Context:
{context}

Question:
{input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)

if st.session_state.retriever:
    retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

    query = st.text_input("Ask a Question: ")
    if query:
        result = retrieval_chain.invoke({"input": query})

        # Display answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources
        sources = result.get("context", [])
        if sources:
            st.subheader("Sources:")
            # Create a unique set of sources to avoid duplicates
            unique_sources = set(doc.metadata.get('source', 'Unknown') for doc in sources)
            for source in unique_sources:
                st.write(f"- {source}")
else:
    st.info("Please process URLs to start asking questions.")