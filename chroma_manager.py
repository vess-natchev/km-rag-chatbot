import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manager for ChromaDB vector database operations."""
    
    def __init__(self, host: str, port: int, collection_name: str):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client: Optional[chromadb.HttpClient] = None
        self.collection: Optional[Collection] = None
    
    async def initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create HTTP client to connect to ChromaDB server
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            if self.client is None:
                raise RuntimeError("Failed to create ChromaDB client")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"ChromaDB initialized: {self.host}:{self.port}")
            logger.info(f"Collection '{self.collection_name}' ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def close(self):
        """Close ChromaDB connection (HTTP client doesn't need explicit close)."""
        logger.info("ChromaDB connection closed")

    async def health_check(self) -> Dict[str, Any]:
        """Check ChromaDB health and return status."""
        try:
            if self.collection is None:
                raise RuntimeError("ChromaDB collection not initialized")
            
            count = self.collection.count()
            return {
                "status": "healthy",
                "document_count": count,
                "collection": self.collection_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def insert_document(self, document_id: str, chunks: List[Dict[str, Any]]):
        """
        Insert document chunks with embeddings into ChromaDB.
        
        Args:
            document_id: Unique document identifier
            chunks: List of dicts with 'text', 'embedding', 'metadata', 'chunk_index'
        """
        try:
            if self.collection is None:
                raise RuntimeError("ChromaDB collection not initialized")
            
            if not chunks:
                logger.warning(f"No chunks to insert for document {document_id}")
                return
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                chunk_id = f"{document_id}_chunk_{chunk['chunk_index']}"
                ids.append(chunk_id)
                embeddings.append(chunk['embedding'])
                documents.append(chunk['text'])
                
                # Flatten metadata for ChromaDB
                metadata = {
                    'document_id': document_id,
                    'chunk_index': chunk['chunk_index'],
                    **chunk.get('metadata', {})
                }
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error inserting document {document_id}: {e}")
            raise

    async def similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on embeddings.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            filters: Metadata filters (e.g., {"country": "Kenya", "status": "approved"})
        
        Returns:
            List of matching chunks with metadata and similarity scores
        """
        try:
            if self.collection is None:
                raise RuntimeError("ChromaDB collection not initialized")
            
            # Build where clause from filters
            where = None
            if filters:
                # ChromaDB uses where clause for metadata filtering
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Convert distance to similarity (cosine distance â†’ cosine similarity)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # ChromaDB returns cosine distance
                    
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': float(similarity),
                        'chunk_index': results['metadatas'][0][i].get('chunk_index', 0)
                    })
            
            logger.debug(f"Similarity search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise

    async def full_text_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text search on documents using pre-generated embeddings.
        Note: ChromaDB doesn't have native full-text search like PostgreSQL,
        so this is essentially another similarity search to provide additional results.
        
        Args:
            query_embedding: Pre-generated query embedding vector
            limit: Maximum number of results
            filters: Metadata filters
        
        Returns:
            List of matching documents with metadata
        """
        try:
            if self.collection is None:
                raise RuntimeError("ChromaDB collection not initialized")
            
            # Build where clause
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            # Use pre-generated embeddings for search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    relevance = 1 - distance
                    
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'relevance_score': float(relevance)
                    })
            
            logger.debug(f"Full-text search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in full-text search: {e}")
            raise

    async def get_documents(
        self,
        offset: int = 0,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get paginated list of documents.
        
        Args:
            offset: Number of documents to skip
            limit: Maximum number of documents to return
            filters: Metadata filters
        
        Returns:
            List of document metadata
        """
        try:
            if self.collection is None:
                raise RuntimeError("ChromaDB collection not initialized")
            
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            # Get documents (ChromaDB doesn't have native pagination, so we fetch more and slice)
            results = self.collection.get(
                where=where,
                limit=offset + limit,
                include=["metadatas"]
            )
            
            # Extract unique documents (by document_id)
            documents_dict = {}
            if results and results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    document_id = metadata.get('document_id', doc_id)
                    
                    if document_id not in documents_dict:
                        documents_dict[document_id] = {
                            'id': document_id,
                            'title': metadata.get('title', 'Unknown'),
                            'file_path': metadata.get('file_path'),
                            'country': metadata.get('country'),
                            'status': metadata.get('status'),
                            'project_type': metadata.get('project_type'),
                            'document_type': metadata.get('document_type'),
                            'year': metadata.get('year'),
                            'created_at': metadata.get('created_at'),
                        }
            
            # Apply pagination
            documents = list(documents_dict.values())
            return documents[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            raise

    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        try:
            if self.collection is None:
                raise RuntimeError("ChromaDB collection not initialized")
            
            # Get all metadata
            all_results = self.collection.get(
                include=["metadatas"]
            )
            
            total_chunks = len(all_results['ids']) if all_results['ids'] else 0
            
            # Count unique documents
            unique_docs = set()
            countries = {}
            doc_types = {}
            years = {}
            statuses = {}
            
            if all_results and all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    doc_id = metadata.get('document_id')
                    if doc_id:
                        unique_docs.add(doc_id)
                    
                    # Count by country
                    country = metadata.get('country')
                    if country:
                        countries[country] = countries.get(country, 0) + 1
                    
                    # Count by document type
                    doc_type = metadata.get('document_type')
                    if doc_type:
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    # Count by year
                    year = metadata.get('year')
                    if year:
                        years[year] = years.get(year, 0) + 1
                    
                    # Count by status
                    status = metadata.get('status')
                    if status:
                        statuses[status] = statuses.get(status, 0) + 1
            
            return {
                'total_documents': len(unique_docs),
                'total_chunks': total_chunks,
                'countries': [{'name': k, 'count': v} for k, v in sorted(countries.items())],
                'document_types': [{'type': k, 'count': v} for k, v in sorted(doc_types.items())],
                'years': [{'year': k, 'count': v} for k, v in sorted(years.items(), reverse=True)],
                'statuses': [{'status': k, 'count': v} for k, v in sorted(statuses.items())],
                'collection_name': self.collection_name,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise

    async def delete_document(self, document_id: str):
        """Delete all chunks for a document."""
        try:
            if self.collection is None:
                raise RuntimeError("ChromaDB collection not initialized")
            
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results and results['ids']:
                # Delete by IDs
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.warning(f"No chunks found for document {document_id}")
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise

    async def reset_collection(self):
        """Reset the entire collection (delete all data)."""
        try:
            if self.client is None:
                raise RuntimeError("ChromaDB client not initialized")
            
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' reset")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise