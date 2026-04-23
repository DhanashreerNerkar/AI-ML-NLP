from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None


class DocumentPage(BaseModel):
    page_number: int
    text: str


class DocumentData(BaseModel):
    id: int
    filename: str
    pages: List[DocumentPage]
    total_pages: int


class ChatRequest(BaseModel):
    question: str
    documents: List[DocumentData]  # Documents sent from client
    description: str  # Collection description
    chat_history: Optional[List[ChatMessage]] = []
    model: Optional[str] = "gpt-5-mini"


class ChatResponse(BaseModel):
    answer: str
    selected_documents: List[str]
    relevant_pages_count: int
    total_cost: Optional[float] = 0.0


class UploadResponse(BaseModel):
    documents: List[DocumentData]  # Return processed documents to client
    message: str


class UpdateDescriptionRequest(BaseModel):
    description: str


class AddDocumentsResponse(BaseModel):
    documents: List[DocumentData]  # Return all documents including new ones
    message: str
    new_documents_count: int


class SessionData(BaseModel):
    session_id: str
    description: str
    documents: List[Dict[str, Any]]
    created_at: datetime
    total_session_cost: Optional[float] = 0.0
