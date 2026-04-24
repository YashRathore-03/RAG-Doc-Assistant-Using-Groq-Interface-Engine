import streamlit as st
import os
import time
from langchain_groq import ChatGroq
print("Import successful")
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Session state initialization
if "vector" not in st.session_state:
    with st.spinner("Loading documents..."):
        st.session_state.embeddings = OllamaEmbeddings(model="phi3")  # Specify model
        st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
        st.session_state.docs = st.session_state.loader.load()
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq Demo")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.1
)

prompt_template = ChatPromptTemplate.from_template("""
Answer the question based on the following context:
{context}

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vector.as_retriever(search_kwargs={"k": 4})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your question here")

if prompt:
    start = time.process_time()
    with st.spinner("Thinking..."):
        response = retrieval_chain.invoke({"input": prompt})  # Note: use "input" key
    st.write(response['answer'])
    st.info(f"Response time: {time.process_time() - start:.2f}s")
    
    with st.expander("Source Documents"):
        for i, doc in enumerate(response["context"]):
            st.write(f"**Document {i+1}:**")
            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            st.write("---")