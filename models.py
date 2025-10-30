# models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    REPORT = "report"
    EVALUATION = "evaluation"
    BASELINE = "baseline"
    ENDLINE = "endline"
    TRAINING = "training"
    MANUAL = "manual"
    GUIDE = "guide"
    PRESENTATION = "presentation"
    PROPOSAL = "proposal"
    BUDGET = "budget"
    AGREEMENT = "agreement"

class ProjectStatus(str, Enum):
    APPROVED = "approved"
    AWARDED = "awarded"
    REJECTED = "rejected"
    PENDING = "pending"
    COMPLETED = "completed"

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation context")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters (e.g., country, status)")
    max_results: Optional[int] = Field(default=5, description="Maximum number of results to return")
    
class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")

class DocumentData(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    file_path: Optional[str] = Field(default=None, description="Original file path")
    country: Optional[str] = Field(default=None, description="Country associated with document")
    project_id: Optional[str] = Field(default=None, description="Unique project identifier")
    project_name: Optional[str] = Field(default=None, description="Project name")
    project_type: Optional[str] = Field(default=None, description="Type of project")
    value_chain: Optional[str] = Field(default=None, description="Value chain (e.g., livestock, crops)")
    document_type: Optional[DocumentType] = Field(default=None, description="Document type")
    status: Optional[ProjectStatus] = Field(default=None, description="Project status (approved/rejected/etc)")
    year: Optional[int] = Field(default=None, description="Document year")
    author: Optional[str] = Field(default=None, description="Document author")
    created_at: Optional[str] = Field(default=None, description="Document creation date")

class DocumentIngestRequest(BaseModel):
    documents: List[DocumentData] = Field(..., description="List of documents to ingest")

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    file_path: Optional[str]
    country: Optional[str]
    status: Optional[str]
    project_type: Optional[str]
    document_type: Optional[str]
    year: Optional[int]
    created_at: Optional[str]
