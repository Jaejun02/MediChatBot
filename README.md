# 🏥 Medical Symptom Analyzer with ChromaDB and OpenAI GPT

This project is a medical chatbot that leverages **OpenAI's GPT model** and **ChromaDB** for retrieval-augmented generation (RAG) to analyze patient symptoms and suggest potential diseases. It consists of two main components:

1. **Medical ChromaDB Builder**: Processes medical symptom data and creates a ChromaDB vector database for efficient similarity searches.
2. **Medical Symptom Analysis API**: A **FastAPI**-based chatbot that integrates OpenAI's GPT model with ChromaDB to provide medical insights.

---

## ✨ Key Features

✔ **Data Processing & Vector Storage**: Cleans medical text, extracts symptoms, and stores embeddings in ChromaDB.\
✔ **AI-Powered Symptom Analysis**: Uses OpenAI's GPT model to analyze symptoms and provide diagnostic suggestions.\
✔ **Efficient Similarity Search**: Leverages ChromaDB for retrieving relevant medical information.\
✔ **FastAPI Backend**: Provides RESTful API endpoints for querying symptoms.\
✔ **Session-Based Conversation**: Maintains chat history for a better user experience.\
✔ **CORS Support**: Enables cross-origin requests for front-end integration.

---

## 🚀 Installation

### Prerequisites

Ensure you have **Python 3.8+** installed.

### Setup Steps

1️⃣ Clone the repository:

```sh
$ git clone https://github.com/Jaejun02/MediChatBot
$ cd MedicalChatBot
```

2️⃣ Create a **Conda environment** and install dependencies:

```sh
$ conda env create -f environment.yml
$ conda activate medichatbot
```

3️⃣ Set up environment variables by creating a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
PERSIST_DIR=./chroma_db
```

4️⃣ Build the **ChromaDB Vector Store**:

```sh
$ python create_db.py
```

*Note: Running `build_chroma_db.py` requires a dedicated GPU. If you do not have one, you can try out the medical chatbot using the pre-processed `chroma_db` file (though it is only processed with partial dataset).*

---

## 🎯 Usage & API Endpoints

### Run the FastAPI Server

Start the API server with:

```sh
$ uvicorn medichatbot:app --reload
```

Access the API documentation at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc UI: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Endpoints

#### Health Check

**GET /health**\
Returns server status and model details.

```json
{
  "status": "active",
  "model": "gpt-4o-mini",
  "sessions": 3
}
```

#### Chat with the Medical Assistant

**POST /chat**\
Send a message to analyze symptoms.

##### Request:

```json
{
  "message": "I have a fever and headache",
  "session_id": "optional-uuid"
}
```

##### Response:

```json
{
  "response": "Based on your symptoms, you might have flu or a common cold.",
  "session_id": "generated-uuid"
}
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

**Disclaimer:** This tool is for research purposes only. It is not intended for actual medical diagnosis or treatment advice.

---
📌 **Author:** Jaejun Shim  
📆 **Date:** January 29th, 2025
