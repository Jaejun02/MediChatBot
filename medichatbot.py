"""
Medical Symptom Analysis API using FastAPI.

This API provides a chatbot interface for medical symptom analysis using OpenAI's GPT model
and ChromaDB vector database for context-aware medical information retrieval.

Key Components:
- FastAPI backend with REST endpoints.
- OpenAI GPT-4 model integration for natural language processing.
- ChromaDB vector database for medical knowledge retrieval.
- Session-based conversation management.
- Tool calling pattern for context-aware responses.

Technologies Used:
- FastAPI for web server implementation.
- Pydantic for data validation.
- OpenAI API for AI-powered symptom analysis.
- LangChain Chroma for vector database management.
- UUID for session management.

Features:
- Session-based conversation history persistence.
- Contextual symptom analysis using vector similarity search.
- Strict medical protocol enforcement through system prompts.
- Health monitoring endpoint for service status.
- CORS configuration for cross-origin compatibility.
"""

import os
import json
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API key
PERSIST_DIR = os.getenv("PERSIST_DIR")  # Directory for ChromaDB persistence
MODEL = "gpt-4o-mini"  # OpenAI model to use
SYSTEM_MESSAGE = """
You are a helpful medical assistant that analyzes symptoms and provides potential diagnoses.
Always follow these rules:
1. Use get_medi_info tool for medical information
2. Never guess diagnoses without tool data
3. Limit responses to 3 potential conditions
4. Request more symptoms if uncertain (max 2 times)
5. Stay polite and professional
"""

# Global application state management
app_state = {
    "openai_client": None,  # OpenAI client instance
    "embeddings": None,     # OpenAI embeddings model
    "vector_db": None,      # ChromaDB vector database
    "sessions": {}          # Active user sessions storage
}

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Medical Symptom Analyzer API",
    description="API for analyzing medical symptoms with AI-powered diagnosis suggestions",
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@medicalai.com"
    },
    license_info={
        "name": "MIT License"
    }
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request model for the chat endpoint.
    
    Attributes:
        message: User's input message containing symptoms.
        session_id: Optional UUID for continuing existing conversation.
    """
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for the chat endpoint.
    
    Attributes:
        response: AI-generated analysis of symptoms.
        session_id: UUID for maintaining conversation context.
    """
    response: str
    session_id: str


class HealthCheckResponse(BaseModel):
    """Health check response model.
    
    Attributes:
        status: Service status (active/inactive).
        model: Currently loaded AI model.
        sessions: Number of active sessions.
    """
    status: str
    model: str
    sessions: int


@app.on_event("startup")
async def initialize_services():
    """Initialize core application services on startup.
    
    Initializes:
    - OpenAI client for GPT interactions.
    - OpenAI embeddings model for vector database.
    - ChromaDB vector database connection.
    
    Raises:
        RuntimeError: If any service fails to initialize.
    """
    try:
        app_state["openai_client"] = OpenAI(api_key=OPENAI_API_KEY)
        app_state["embeddings"] = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        app_state["vector_db"] = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=app_state["embeddings"]
        )
    except Exception as e:
        raise RuntimeError(f"Service initialization failed: {str(e)}") from e


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    """Redirect root endpoint to API documentation.
    
    Returns:
        RedirectResponse: Redirects to /docs endpoint.
    """
    return RedirectResponse(url="/docs")


def get_medi_info(user_input: str, threshold: float = 0.3) -> str:
    """
    Retrieve medical information from vector database using symptom similarity search.
    
    Args:
        user_input: Comma-separated list of symptoms.
        threshold: Maximum similarity score threshold (lower is more similar).
    
    Returns:
        str: Formatted context string containing disease information.
    
    Example:
        >>> get_medi_info("headache,fever")
        "Disease: Influenza...\n\nDisease: Common Cold..."
    """
    # Perform similarity search with score filtering
    results = app_state["vector_db"].similarity_search_with_score(user_input, k=5)
    filtered = [doc for doc, score in results if score < threshold]
    
    # Format results into context string
    context_items = []
    for doc in filtered:
        context_items.append(
            f"Disease: {doc.metadata.get('label', 'N/A')}\n"
            f"Key Symptoms: {doc.page_content}\n"
            f"Description: {doc.metadata.get('clean_description', 'N/A')}"
        )
    return "\n\n".join(context_items)


# OpenAI tool definition for medical information retrieval
MEDI_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": "get_medi_info",
        "description": "Retrieve medical information for symptom analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "Comma-separated symptoms (e.g., 'headache,fever')"
                }
            },
            "required": ["user_input"],
            "additionalProperties": False
        }
    }
}


def handle_tool_call(tool_call) -> Dict:
    """
    Process OpenAI tool call and generate formatted response.
    
    Args:
        tool_call: OpenAI tool call object from API response.
    
    Returns:
        Dict: Formatted tool response for chat history.
    
    Raises:
        JSONDecodeError: If tool arguments are invalid JSON.
    """
    args = json.loads(tool_call.function.arguments)
    context = get_medi_info(args["user_input"])
    
    return {
        "role": "tool",
        "content": json.dumps({"medical_context": context}),
        "tool_call_id": tool_call.id
    }


def process_chat(user_input: str, chat_history: List[Dict]) -> str:
    """
    Process chat message through OpenAI API with tool handling.
    
    Args:
        user_input: User's symptom description.
        chat_history: Conversation history for context maintenance.
    
    Returns:
        str: Assistant's formatted response text.
    
    Process Flow:
        1. Add user message to history.
        2. Initial API call with tool definition.
        3. Handle tool call if required.
        4. Follow-up API call with tool response.
        5. Return final assistant message.
    """
    chat_history.append({"role": "user", "content": user_input})
    
    # Initial API call with tool specification
    response = app_state["openai_client"].chat.completions.create(
        model=MODEL,
        messages=chat_history,
        tools=[MEDI_INFO_TOOL]
    )
    
    message = response.choices[0].message
    
    # Handle tool call if required
    if message.tool_calls:
        chat_history.append(message)
        tool_response = handle_tool_call(message.tool_calls[0])
        chat_history.append(tool_response)
        
        # Follow-up API call with complete context
        response = app_state["openai_client"].chat.completions.create(
            model=MODEL,
            messages=chat_history
        )
    
    # Extract and store final message
    final_message = response.choices[0].message
    chat_history.append(final_message)
    
    return final_message.content


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for symptom analysis.
    
    Args:
        request: ChatRequest containing message and optional session_id.
    
    Returns:
        ChatResponse: Contains AI response and session ID.
    
    Session Management:
        - Generates new UUID for new sessions.
        - Maintains conversation history per session.
        - Stores history in application state.
    
    Raises:
        HTTPException: 500 error if processing fails.
    """
    try:
        # Session initialization or retrieval
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in app_state["sessions"]:
            app_state["sessions"][session_id] = [
                {"role": "system", "content": SYSTEM_MESSAGE}
            ]
            
        history = app_state["sessions"][session_id]
        response_text = process_chat(request.message, history)
        
        return {
            "response": response_text,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Service health check endpoint.
    
    Returns:
        HealthCheckResponse: Contains service status, model info, and session count.
    """
    return {
        "status": "active",
        "model": MODEL,
        "sessions": len(app_state["sessions"])
    }