from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator, cast
import json, uuid, time, asyncio
import logging
from datetime import datetime

from models import QueryRequest, QueryResponse, DocumentIngestRequest, DocumentData
from chroma_manager import ChromaManager
from rag_engine import RAGEngine
from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Org Knowledge Management API",
    description="RAG API for Org's Knowledge Base with ChromaDB",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3080", "http://127.0.0.1:3080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
chroma_manager: Optional[ChromaManager] = None
rag_engine: Optional[RAGEngine] = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine and ChromaDB on startup."""
    global chroma_manager, rag_engine
    logger.info("Starting up Org Knowledge Management API...")
    
    # Initialize ChromaDB manager
    chroma_manager = ChromaManager(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        collection_name=settings.CHROMA_COLLECTION_NAME
    )
    await chroma_manager.initialize()
    
    # Initialize RAG engine
    rag_engine = RAGEngine(chroma_manager, settings.OPENAI_API_KEY)
    await rag_engine.initialize()
    
    logger.info("API startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")
    if chroma_manager:
        await chroma_manager.close()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Org Knowledge Management API is running!",
        "version": "2.0.0",
        "vector_store": "ChromaDB"
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        if chroma_manager is None:
            raise HTTPException(status_code=503, detail="ChromaDB manager not initialized")
        
        chroma_status = await chroma_manager.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "chromadb": chroma_status,
            "rag_engine": rag_engine is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Main query endpoint for RAG functionality."""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process the query through RAG engine
        response = await rag_engine.query(
            query=request.query,
            conversation_history=request.conversation_history,
            filters=request.filters,
            max_results=request.max_results or 5
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/ingest/documents")
async def ingest_documents(request: DocumentIngestRequest):
    """Ingest new documents into the knowledge base."""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        logger.info(f"Starting document ingestion for {len(request.documents)} documents...")
        
        results = await rag_engine.ingest_documents(request.documents)
        
        return {
            "message": "Document ingestion completed",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), metadata: Optional[str] = None):
    """Ingest a single file upload."""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Read file content
        content = await file.read()
        
        # Parse metadata if provided
        file_metadata = {}
        if metadata:
            file_metadata = json.loads(metadata)
        
        # Basic text extraction (for simple cases)
        if file.content_type and "text" in file.content_type:
            text_content = content.decode('utf-8')
        else:
            raise HTTPException(
                status_code=400,
                detail="Use the comprehensive ingest script for Word/Excel/PDF files"
            )
        
        # Create document object
        doc_data = DocumentData(
            title=file.filename or "untitled",
            content=text_content,
            metadata=file_metadata,
            file_path=file.filename
        )
        
        # Ingest the document
        result = await rag_engine.ingest_documents([doc_data])
        
        return {
            "message": f"File '{file.filename}' ingested successfully",
            "document_id": result[0]["document_id"] if result else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")

@app.get("/documents")
async def list_documents(
    offset: int = 0,
    limit: int = 20,
    country: Optional[str] = None,
    status: Optional[str] = None,
    document_type: Optional[str] = None,
    year: Optional[int] = None
):
    """List documents in the knowledge base."""
    try:
        if chroma_manager is None:
            raise HTTPException(status_code=503, detail="ChromaDB manager not initialized")
        
        filters = {}
        if country:
            filters["country"] = country
        if status:
            filters["status"] = status
        if document_type:
            filters["document_type"] = document_type
        if year:
            filters["year"] = year
        
        documents = await chroma_manager.get_documents(
            offset=offset,
            limit=limit,
            filters=filters
        )
        
        return {
            "documents": documents,
            "offset": offset,
            "limit": limit,
            "filters": filters
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document listing failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get knowledge base statistics."""
    try:
        if chroma_manager is None:
            raise HTTPException(status_code=503, detail="ChromaDB manager not initialized")
        
        stats = await chroma_manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")

# --- OpenAI-compatible shim for LibreChat ---

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "org-knowledge-assistant",
                "object": "model",
                "created": 0,
                "owned_by": "org-rag-backend",
            }
        ],
    }

def _now_i():
    return int(time.time())

def _openai_completion(payload: str, model: str):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": _now_i(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": payload},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

def _chunk(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 1500
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest):
    if req.model != "org-knowledge-assistant":
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    # Latest user message
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message provided")
    user_query = user_msgs[-1]

    # Call RAG engine
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    engine = cast(RAGEngine, rag_engine)
    rag_resp = await engine.query(
        query=user_query,
        conversation_history=[m.dict() for m in req.messages[:-1]] or None,
        filters=None,
        max_results=5,
    )
    full_text = rag_resp.answer or "I couldn't find relevant info in the knowledge base."

    # Non-streaming path
    if not req.stream:
        return JSONResponse(_openai_completion(full_text, req.model))

    # Streaming path (SSE)
    async def streamer() -> AsyncGenerator[bytes, None]:
        preamble = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": _now_i(),
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }
        yield _chunk(preamble).encode("utf-8")

        # Stream in chunks
        step = 800
        for i in range(0, len(full_text), step):
            part = full_text[i:i+step]
            chunk = {
                "id": preamble["id"],
                "object": "chat.completion.chunk",
                "created": _now_i(),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
            }
            yield _chunk(chunk).encode("utf-8")
            await asyncio.sleep(0)

        # Final chunk
        done = {
            "id": preamble["id"],
            "object": "chat.completion.chunk",
            "created": _now_i(),
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield _chunk(done).encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
