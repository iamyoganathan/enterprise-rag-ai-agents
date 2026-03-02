"""
Documents API Endpoints
Handle document upload, processing, and management.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid

from src.ingestion import IngestionPipeline, ingest_file
from src.embeddings import IndexingPipeline
from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


# Request/Response models
class DocumentResponse(BaseModel):
    """Document processing response."""
    id: str
    filename: str
    status: str
    chunks: int
    message: str


class DocumentListResponse(BaseModel):
    """List of documents response."""
    documents: List[Dict[str, Any]]
    total: int


class DeleteResponse(BaseModel):
    """Delete operation response."""
    status: str
    message: str


# Background task for processing
async def process_document_task(file_path: Path, doc_id: str):
    """Background task to process document."""
    try:
        logger.info(f"Processing document: {file_path.name}")
        
        # Ingest document
        ingestion = IngestionPipeline(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        result = ingestion.process_file(file_path)
        
        # Index chunks
        indexing = IndexingPipeline(collection_name="documents")
        index_result = indexing.index_document(
            result["chunks"],
            {
                "document_id": doc_id,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "category": result["metadata"].get("category", "unknown")
            }
        )
        
        logger.info(
            f"Document processed: {file_path.name}, "
            f"{index_result['chunks_indexed']} chunks indexed"
        )
        
    except Exception as e:
        logger.error(f"Failed to process document: {str(e)}")


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a document.
    
    Supports: PDF, DOCX, TXT, MD
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Save file
        upload_dir = Path("data") / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{doc_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file.filename} ({file_path})")
        
        # Add background task for processing
        background_tasks.add_task(process_document_task, file_path, doc_id)
        
        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            status="processing",
            chunks=0,
            message="Document uploaded and queued for processing"
        )
    
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-multiple")
async def upload_multiple_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload multiple documents at once."""
    results = []
    
    for file in files:
        try:
            response = await upload_document(background_tasks, file)
            results.append({
                "filename": file.filename,
                "status": "success",
                "id": response.id
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "uploaded": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }


@router.get("/list")
async def list_documents(
    limit: int = 100,
    offset: int = 0
):
    """List all indexed documents."""
    try:
        from src.embeddings import get_vector_store
        
        vector_store = get_vector_store()
        
        # Get documents with metadata
        results = vector_store.get_documents(limit=limit, offset=offset)
        
        # Extract unique documents by file_name
        documents_dict = {}
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            file_name = metadata.get("file_name", "unknown")
            
            if file_name not in documents_dict:
                documents_dict[file_name] = {
                    "file_name": file_name,
                    "document_id": metadata.get("document_id", "unknown"),
                    "category": metadata.get("category", "unknown"),
                    "chunks": 0
                }
            
            documents_dict[file_name]["chunks"] += 1
        
        documents = list(documents_dict.values())
        
        return {
            "documents": documents,
            "total": len(documents),
            "total_chunks": len(results["ids"])
        }
    
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get details of a specific document."""
    try:
        from src.embeddings import get_vector_store
        
        vector_store = get_vector_store()
        
        # Get chunks for this document
        results = vector_store.get_documents(
            where={"document_id": document_id}
        )
        
        if not results["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Build response
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        
        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "file_name": results["metadatas"][0].get("file_name", "unknown"),
            "category": results["metadatas"][0].get("category", "unknown"),
            "chunks_data": chunks
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    try:
        from src.embeddings import get_vector_store
        
        vector_store = get_vector_store()
        
        # Check if document exists
        results = vector_store.get_documents(
            where={"document_id": document_id},
            limit=1
        )
        
        if not results["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete all chunks
        vector_store.delete_documents(where={"document_id": document_id})
        
        logger.info(f"Deleted document: {document_id}")
        
        return {
            "status": "success",
            "message": f"Document {document_id} deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/")
async def delete_all_documents():
    """Delete all documents from the system."""
    try:
        from src.embeddings import get_vector_store
        
        vector_store = get_vector_store()
        count_before = vector_store.count()
        
        vector_store.reset()
        
        logger.warning("All documents deleted from system")
        
        return {
            "status": "success",
            "message": f"Deleted {count_before} document chunks"
        }
    
    except Exception as e:
        logger.error(f"Failed to delete all documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
