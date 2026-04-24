# RAG-Doc-Assistant-Using-Groq-Interface-Engine

A **Retrieval-Augmented Generation (RAG)** based chatbot that allows users to query web-based documents in real-time using **LangChain**, **Groq LLM**, **Ollama embeddings**, and **FAISS vector database**, all wrapped in a **Streamlit UI**.

---

## 🚀 Features

- 🔍 Context-aware question answering from web documents  
- ⚡ Ultra-fast inference using Groq LLM  
- 🧠 Semantic search using vector embeddings (FAISS)  
- 🌐 Web document ingestion  
- 💬 Interactive UI with Streamlit  
- 📑 Source document transparency  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- LangChain  
- Groq API  
- Ollama (Embeddings)  
- FAISS (Vector DB)  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/docuquery-ai.git
cd docuquery-ai
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
source venv/bin/activate
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key_here

```

### 5️⃣ Install & Run Ollama (for embeddings)

Download Ollama:
👉 https://ollama.com/

Pull embedding model:
```bash
ollama pull phi3
```

### 6️⃣ Run the Application

```bash
streamlit run app.py
```
