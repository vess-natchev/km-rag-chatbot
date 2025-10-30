import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import openai
import tiktoken
import numpy as np
from datetime import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize

# Import Unstructured library for semantic chunking
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, CompositeElement, Title, NarrativeText, Text

from chroma_manager import ChromaManager
from models import DocumentData, QueryResponse
from config import settings

logger = logging.getLogger(__name__)

# Configure detailed logging for RAG operations
rag_logger = logging.getLogger(__name__ + '.rag_operations')
rag_logger.setLevel(logging.DEBUG)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class RAGEngine:
    def __init__(self, chroma_manager: ChromaManager, openai_api_key: str):
        self.chroma_manager = chroma_manager
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL)
        
        # Initialize reranker if enabled
        self.reranker = None
        if settings.USE_RERANKER:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(settings.RERANKER_MODEL)
                logger.info(f"Reranker initialized: {settings.RERANKER_MODEL}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Reranking disabled.")
                logger.warning("Install with: pip install sentence-transformers")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
        
    async def initialize(self):
        """Initialize the RAG engine."""
        logger.info("Initializing RAG engine...")
        logger.info(f"Chunking strategy: {settings.CHUNKING_STRATEGY}")
        logger.info(f"Max chunk size: {settings.MAX_CHUNK_SIZE} characters")
        logger.info(f"Chunk overlap: {settings.CHUNK_OVERLAP} characters")
        
        # Test OpenAI connection
        try:
            await self.openai_client.models.list()
            logger.info("OpenAI connection successful")
        except Exception as e:
            logger.error(f"OpenAI connection failed: {e}")
            raise
        
        logger.info("RAG engine initialized successfully")

    def _parse_document_structure(self, text: str) -> List[Element]:
        """
        Parse document text into structured elements (Titles, Paragraphs).
        This creates a document structure that semantic chunking can work with.
        """
        elements = []
        element_id = 0
        
        # Split text into lines
        lines = text.split('\n')
        
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # Empty line - finish current paragraph if any
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    elements.append(
                        NarrativeText(
                            text=para_text,
                            element_id=str(element_id)
                        )
                    )
                    element_id += 1
                    current_paragraph = []
                continue
            
            # Heuristic: Detect headings/titles
            # Titles are typically:
            # - Short (< 100 chars)
            # - End without punctuation (or with colon)
            # - May be in ALL CAPS or Title Case
            # - May start with numbers (1., 2.3, etc.)
            is_title = False
            
            # Check for numbered headings (1., 1.1, etc.)
            if re.match(r'^\d+\.(\d+\.)*\s+', line):
                is_title = True
            # Check for ALL CAPS (at least 3 words)
            elif line.isupper() and len(line.split()) >= 2:
                is_title = True
            # Check for short lines ending with colon
            elif len(line) < 100 and line.endswith(':'):
                is_title = True
            # Check for Title Case (most words capitalized) and short
            elif len(line) < 100 and not line.endswith('.') and not line.endswith(','):
                words = line.split()
                if words and len(words) <= 15:
                    capitalized = sum(1 for w in words if w and w[0].isupper())
                    if capitalized / len(words) > 0.6:  # 60% capitalized
                        is_title = True
            
            if is_title:
                # Finish current paragraph
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    elements.append(
                        NarrativeText(
                            text=para_text,
                            element_id=str(element_id)
                        )
                    )
                    element_id += 1
                    current_paragraph = []
                
                # Add title
                elements.append(
                    Title(
                        text=line,
                        element_id=str(element_id)
                    )
                )
                element_id += 1
            else:
                # Add to current paragraph
                current_paragraph.append(line)
        
        # Add final paragraph if any
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            elements.append(
                NarrativeText(
                    text=para_text,
                    element_id=str(element_id)
                )
            )
        
        return elements

    def chunk_text_semantic(
        self,
        text: str,
        max_characters: Optional[int] = None,
        overlap: Optional[int] = None,
        combine_under_n_chars: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        Chunk text using semantic/structure-aware strategy.
        
        This method:
        1. Parses document structure (titles, paragraphs)
        2. Uses Unstructured's chunk_by_title for semantic chunking
        3. Preserves document structure and context
        4. Returns chunks with indices
        
        Args:
            text: Document text to chunk
            max_characters: Maximum chunk size in characters
            overlap: Character overlap between chunks
            combine_under_n_chars: Combine sections smaller than this
            
        Returns:
            List of (chunk_text, chunk_index) tuples
        """
        max_characters = max_characters or settings.MAX_CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP
        combine_under_n_chars = combine_under_n_chars or settings.COMBINE_TEXT_UNDER_N_CHARS
        
        try:
            # Parse document into structured elements
            elements = self._parse_document_structure(text)
            
            if not elements:
                logger.warning("No elements parsed from document")
                return []
            
            # Apply semantic chunking using Unstructured's chunk_by_title
            chunked_elements = chunk_by_title(
                elements=elements,
                max_characters=max_characters,
                new_after_n_chars=max_characters - overlap,  # Soft limit for new chunks
                combine_text_under_n_chars=combine_under_n_chars,
                multipage_sections=settings.MULTIPAGE_SECTIONS,
                overlap=overlap,
                overlap_all=False  # Only overlap split chunks, not semantic boundaries
            )
            
            # Convert chunked elements to (text, index) tuples
            chunks = []
            for idx, element in enumerate(chunked_elements):
                chunk_text = element.text if hasattr(element, 'text') else str(element)
                if chunk_text.strip():
                    chunks.append((chunk_text.strip(), idx))
            
            logger.debug(f"Semantic chunking produced {len(chunks)} chunks from {len(elements)} elements")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            # Fallback to simple paragraph-based chunking
            logger.warning("Falling back to paragraph-based chunking")
            return self._fallback_chunk_text(text, max_characters, overlap)
    
    def _fallback_chunk_text(
        self,
        text: str,
        max_characters: int,
        overlap: int
    ) -> List[Tuple[str, int]]:
        """
        Fallback chunking method using paragraph boundaries.
        Used when semantic chunking fails.
        """
        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max, start new chunk
            if current_chunk and len(current_chunk) + len(para) + 1 > max_characters:
                chunks.append((current_chunk.strip(), chunk_index))
                chunk_index += 1
                
                # Add overlap from previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk.strip(), chunk_index))
        
        return chunks

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.openai_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=batch,
                    dimensions=settings.EMBEDDING_DIMENSION
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def ingest_documents(self, documents: List[DocumentData]) -> List[Dict[str, Any]]:
        """Ingest documents into the knowledge base using semantic chunking."""
        results = []
        
        for document in documents:
            try:
                logger.info(f"Processing document: {document.title}")
                
                # Generate document ID
                import uuid
                doc_id = str(uuid.uuid4())
                
                # Chunk the document content using semantic chunking
                chunks = self.chunk_text_semantic(document.content)
                logger.info(f"Created {len(chunks)} semantic chunks for document {document.title}")
                
                if not chunks:
                    logger.warning(f"No chunks created for document {document.title}")
                    results.append({
                        'document_id': doc_id,
                        'title': document.title,
                        'status': 'warning',
                        'message': 'No chunks created',
                        'chunks': 0,
                        'embeddings': 0
                    })
                    continue
                
                # Filter out empty chunks
                valid_chunks = [(text, idx) for text, idx in chunks if text.strip()]
                valid_chunk_texts = [text for text, idx in valid_chunks]
                
                if not valid_chunk_texts:
                    logger.warning(f"No valid chunks after filtering for document {document.title}")
                    results.append({
                        'document_id': doc_id,
                        'title': document.title,
                        'status': 'warning',
                        'message': 'No valid chunks after filtering',
                        'chunks': len(chunks),
                        'embeddings': 0
                    })
                    continue
                
                # Generate embeddings for valid chunks
                embeddings = await self.generate_embeddings(valid_chunk_texts)
                
                # Prepare chunk data with metadata
                chunk_data = []
                for (chunk_text, chunk_index), embedding in zip(valid_chunks, embeddings):
                    chunk_data.append({
                        'text': chunk_text,
                        'embedding': embedding,
                        'chunk_index': chunk_index,
                        'metadata': {
                            'title': document.title,
                            'file_path': document.file_path,
                            'country': document.country,
                            'project_id': document.project_id,
                            'project_name': document.project_name,
                            'project_type': document.project_type,
                            'value_chain': document.value_chain,
                            'document_type': document.document_type,
                            'status': document.status,  # CRITICAL: approved vs rejected
                            'year': document.year,
                            'author': document.author,
                            'created_at': document.created_at or datetime.utcnow().isoformat(),
                            'chunking_strategy': settings.CHUNKING_STRATEGY
                        }
                    })
                
                # Insert into ChromaDB
                await self.chroma_manager.insert_document(doc_id, chunk_data)
                
                results.append({
                    'document_id': doc_id,
                    'title': document.title,
                    'status': 'success',
                    'chunks': len(chunks),
                    'valid_chunks': len(valid_chunks),
                    'embeddings': len(embeddings)
                })
                
                logger.info(f"Successfully processed document: {document.title}")
                
            except Exception as e:
                logger.error(f"Error processing document {document.title}: {e}")
                results.append({
                    'title': document.title,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder model.
        
        Args:
            query: The user's query
            results: List of search results with 'content' field
            
        Returns:
            Reranked list of results with updated scores
        """
        if not self.reranker or not results:
            return results
        
        try:
            # Prepare query-document pairs for the cross-encoder
            pairs = [[query, result['content']] for result in results]
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Add rerank scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
            
            # Sort by rerank score (descending)
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            
            rag_logger.debug(f"Reranked {len(results)} results")
            top_scores = [f"{r['rerank_score']:.4f}" for r in reranked[:3]]
            rag_logger.debug(f"Top rerank scores: {top_scores}")
            
            return reranked
            
        except Exception as e:
            rag_logger.error(f"Error during reranking: {e}")
            return results  # Return original order if reranking fails

    async def query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 5
    ) -> QueryResponse:
        """Process a query using RAG."""
        
        try:
            rag_logger.info("="*80)
            rag_logger.info(f"RAG QUERY STARTED: {query[:100]}...")
            rag_logger.info("="*80)
            
            if filters:
                rag_logger.info(f"Applied Filters: {filters}")
            
            # Calculate how many candidates to retrieve
            retrieval_limit = max_results * settings.VECTOR_RERANK_MULTIPLIER if settings.USE_RERANKER else max_results
            
            # Generate embedding for the query
            rag_logger.info("Step 1: Generating query embedding...")
            query_embeddings = await self.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            rag_logger.info(f"âœ“ Query embedding generated (dimension: {len(query_embedding)})")
            
            # Perform similarity search
            rag_logger.info(f"\nStep 2: Performing similarity search (retrieving {retrieval_limit} candidates)...")
            similar_chunks = await self.chroma_manager.similarity_search(
                query_embedding=query_embedding,
                limit=retrieval_limit,
                filters=filters
            )
            rag_logger.info(f"âœ“ Similarity search returned {len(similar_chunks)} chunks")
            
            if similar_chunks:
                rag_logger.debug("\nRETRIEVED CHUNKS FROM VECTOR SEARCH:")
                for i, chunk in enumerate(similar_chunks[:5], 1):  # Log top 5
                    rag_logger.debug(f"\nChunk {i}:")
                    rag_logger.debug(f"  Similarity: {chunk.get('similarity', 0.0):.4f}")
                    rag_logger.debug(f"  Title: {chunk['metadata'].get('title', 'Unknown')}")
                    rag_logger.debug(f"  Status: {chunk['metadata'].get('status', 'N/A')}")
                    rag_logger.debug(f"  Country: {chunk['metadata'].get('country', 'N/A')}")
                    rag_logger.debug(f"  Content: {chunk['content'][:200]}...")
            
            # Also perform full-text search using the same query embedding
            rag_logger.info(f"\nStep 3: Performing full-text search...")
            fts_results = await self.chroma_manager.full_text_search(
                query_embedding=query_embedding,
                limit=max_results,
                filters=filters
            )
            rag_logger.info(f"âœ“ Full-text search returned {len(fts_results)} documents")
            
            # Combine and deduplicate results
            rag_logger.info(f"\nStep 4: Combining search results...")
            all_sources = self._combine_search_results(similar_chunks, fts_results)
            rag_logger.info(f"âœ“ Combined to {len(all_sources)} unique sources")
            
            # Rerank if enabled and we have more results than needed
            if settings.USE_RERANKER and self.reranker and len(all_sources) > max_results:
                rag_logger.info(f"\nStep 5: Reranking {len(all_sources)} candidates with cross-encoder...")
                all_sources = self._rerank_results(query, all_sources)
                rag_logger.info(f"âœ“ Reranking completed")
                
                if all_sources:
                    rag_logger.debug("\nTOP RESULTS AFTER RERANKING:")
                    for i, source in enumerate(all_sources[:3], 1):
                        rag_logger.debug(f"\nResult {i}:")
                        rag_logger.debug(f"  Rerank Score: {source.get('rerank_score', 0.0):.4f}")
                        rag_logger.debug(f"  Original Similarity: {source.get('similarity', 0.0):.4f}")
                        rag_logger.debug(f"  Title: {source['metadata'].get('title', 'Unknown')}")
                        rag_logger.debug(f"  Content: {source['content'][:200]}...")
            
            # Select top sources (after reranking if enabled)
            top_sources = all_sources[:max_results]
            rag_logger.info(f"\nâœ“ Selected top {len(top_sources)} sources for LLM context")
            
            if not top_sources:
                rag_logger.warning("\nâš ï¸ NO SOURCES FOUND - Returning default response")
                return QueryResponse(
                    answer="I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your query or contact the Org team for more specific information.",
                    sources=[],
                    metadata={
                        'query': query,
                        'search_type': 'no_results',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
            
            # Generate contextual answer
            rag_logger.info(f"\nStep 6: Generating answer with LLM...")
            answer = await self._generate_answer(query, top_sources, conversation_history)
            rag_logger.info(f"âœ“ Answer generated (length: {len(answer)} characters)")
            
            # Format sources for response
            formatted_sources = []
            for source in top_sources:
                formatted_source = {
                    'content': source['content'][:300] + "..." if len(source['content']) > 300 else source['content'],
                    'metadata': source['metadata'],
                    'similarity': source.get('similarity', 0.0),
                    'relevance_score': source.get('relevance_score', 0.0)
                }
                if 'rerank_score' in source:
                    formatted_source['rerank_score'] = source['rerank_score']
                formatted_sources.append(formatted_source)
            
            rag_logger.info("\n" + "="*80)
            rag_logger.info("RAG QUERY COMPLETED SUCCESSFULLY")
            rag_logger.info("="*80 + "\n")
            
            return QueryResponse(
                answer=answer,
                sources=formatted_sources,
                metadata={
                    'query': query,
                    'total_sources': len(all_sources),
                    'used_sources': len(top_sources),
                    'search_types_used': ['similarity', 'full_text'],
                    'reranking_enabled': settings.USE_RERANKER and self.reranker is not None,
                    'chunking_strategy': settings.CHUNKING_STRATEGY,
                    'filters_applied': filters,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            rag_logger.error(f"âœ— Error processing query: {e}", exc_info=True)
            raise

    def _combine_search_results(
        self,
        similarity_results: List[Dict[str, Any]],
        fts_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine and deduplicate search results."""
        
        combined = []
        seen_content = set()
        
        # Add similarity results first (usually higher quality)
        for result in similarity_results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined.append(result)
        
        # Add unique FTS results
        for result in fts_results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                # Add missing fields for consistency
                result['similarity'] = 0.0
                combined.append(result)
        
        return combined

    async def _generate_answer(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate an answer using the retrieved sources."""
        
        # Prepare context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            metadata = source['metadata']
            status = metadata.get('status', 'unknown')
            country = metadata.get('country', 'Unknown')
            title = metadata.get('title', 'Unknown')
            
            context_part = f"Source {i} ({title} - {country} - Status: {status}):\n{source['content']}\n"
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        # Prepare conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-3:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_parts.append(f"{role}: {content}")
            history_text = "\n".join(history_parts)
        
        # Create system prompt with status-aware instructions
        system_prompt = """You are an AI assistant for this nonprofit organization. Your role is to help staff by providing accurate, contextual answers based on the organization's knowledge base.

CRITICAL: Pay close attention to project status in the sources:
- APPROVED/AWARDED projects are successful projects that were funded and implemented
- REJECTED projects are proposals that were not approved for funding
- When answering about "successful" or "approved" projects, ONLY use sources marked as "approved" or "awarded"
- When answering about "rejected" or "unsuccessful" proposals, ONLY use sources marked as "rejected"

Key guidelines:
1. Always base your answers on the provided context sources
2. Be specific and cite relevant details from the sources
3. Pay attention to project status (approved vs rejected) when answering
4. If the answer isn't clearly supported by the sources, say so
5. Focus on practical, actionable information
6. Mention specific countries, projects, or metrics when available
7. Use professional but accessible language
8. Structure longer answers with clear sections or bullet points

Context from knowledge base:
{context}

{history_section}

Answer the user's question based on the provided context. If you cannot answer based on the context, explain what information is available and suggest how to get more specific help."""
        
        if history_text:
            history_section = f"Previous conversation:\n{history_text}\n"
        else:
            history_section = ""
        
        formatted_system_prompt = system_prompt.format(
            context=context,
            history_section=history_section
        )
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Generate response
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            rag_logger.error(f"âœ— Error generating answer: {e}", exc_info=True)
            return f"I encountered an error while generating the response. Please try again or contact support. Error: {str(e)}"