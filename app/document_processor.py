import os
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import hashlib
import time
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Use a smaller model for better performance
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []
        self.sources = []
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set")
            self.groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
            
            # Set the model directly to one we know is available
            self.preferred_model = "llama3-70b-8192"
            logger.info(f"Using model: {self.preferred_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise
        self.is_initialized_flag = False
        # Add memory limits
        self.max_chunks = 1000  # Limit total chunks
        self.max_chunk_size = 500  # Smaller chunk size
        self.chunk_overlap = 100  # Reduced overlap
        self.query_cache = {}
        self.last_cleanup = time.time()
        self.cache_ttl = 3600  # 1 hour cache TTL

    def _test_groq_connection(self):
        """Test the Groq API connection and get available models."""
        try:
            # List available models
            models = self.groq_client.models.list()
            available_models = [model.id for model in models.data]
            logger.info(f"Available Groq models: {available_models}")
            
            # Check if our preferred model is available
            self.preferred_model = "mixtral-8x7b-32768"  # Default model
            if self.preferred_model not in available_models:
                # Try alternative models
                alternative_models = ["llama2-70b-4096", "gemma-7b-it"]
                for model in alternative_models:
                    if model in available_models:
                        self.preferred_model = model
                        break
                else:
                    raise ValueError(f"No supported models available. Available models: {available_models}")
            
            logger.info(f"Using model: {self.preferred_model}")
        except Exception as e:
            logger.error(f"Error testing Groq connection: {str(e)}")
            raise

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        if current_time - self.last_cleanup > 300:  # Cleanup every 5 minutes
            expired_keys = [k for k, v in self.query_cache.items() 
                          if current_time - v['timestamp'] > self.cache_ttl]
            for k in expired_keys:
                del self.query_cache[k]
            self.last_cleanup = current_time

    def process_documents(self, directory: Path) -> None:
        """Process all documents in the given directory."""
        try:
            all_texts = []
            all_sources = []
            processed_files = set()

            for file_path in directory.glob("*"):
                if len(all_texts) >= self.max_chunks:
                    break

                file_hash = self._get_file_hash(file_path)
                if file_hash in processed_files:
                    continue

                try:
                    if file_path.suffix.lower() == '.pdf':
                        text = self._read_pdf(file_path)
                    elif file_path.suffix.lower() == '.docx':
                        text = self._read_docx(file_path)
                    elif file_path.suffix.lower() == '.txt':
                        text = self._read_txt(file_path)
                    elif file_path.suffix.lower() == '.csv':
                        text = self._read_csv(file_path)
                    else:
                        continue

                    chunks = self._split_text(text)
                    remaining_slots = self.max_chunks - len(all_texts)
                    chunks = chunks[:remaining_slots]
                    all_texts.extend(chunks)
                    all_sources.extend([str(file_path)] * len(chunks))
                    processed_files.add(file_hash)

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue

            if all_texts:
                self.texts = all_texts
                self.sources = all_sources
                self._build_index(all_texts)
                self.is_initialized_flag = True
                self.query_cache.clear()  # Clear cache when new documents are processed

        except Exception as e:
            print(f"Error in process_documents: {str(e)}")
            raise

    def _build_index(self, texts: List[str]) -> None:
        """Build FAISS index in batches with progress tracking."""
        try:
            batch_size = 100
            embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_embeddings = self.model.encode(batch)
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error encoding batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                    continue

            if embeddings:
                dimension = len(embeddings[0])
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(np.array(embeddings).astype('float32'))
        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for caching."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return str(file_path)

    def _split_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split text into chunks with overlap."""
        chunk_size = chunk_size or self.max_chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length and len(chunks) < self.max_chunks:
            end = min(start + chunk_size, text_length)
            if end < text_length:
                # Try to find a good breaking point
                break_point = text.rfind('.', start, end)
                if break_point != -1:
                    end = break_point + 1
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap

        return chunks

    def _read_pdf(self, file_path: Path) -> str:
        """Read text from a PDF file with memory optimization."""
        text = ""
        doc = fitz.open(file_path)
        # Limit number of pages to process
        max_pages = 50
        for page_num, page in enumerate(doc):
            if page_num >= max_pages:
                break
            text += page.get_text()
        doc.close()
        return text

    def _read_docx(self, file_path: Path) -> str:
        """Read text from a DOCX file with memory optimization."""
        doc = Document(file_path)
        # Limit number of paragraphs to process
        max_paragraphs = 1000
        return "\n".join([p.text for p in doc.paragraphs[:max_paragraphs]])

    def _read_txt(self, file_path: Path) -> str:
        """Read text from a TXT file with memory optimization."""
        max_size = 10 * 1024 * 1024  # 10MB limit
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read(max_size)

    def _read_csv(self, file_path: Path) -> str:
        """Read text from a CSV file with memory optimization."""
        # Read only first 1000 rows
        df = pd.read_csv(file_path, nrows=1000)
        return df.to_string()

    def query(self, query: str, mode: str = "simple") -> Dict[str, Any]:
        """Query the processed documents using Groq with caching and retries."""
        if not self.is_initialized():
            return {
                "answer": "No documents have been processed yet. Please upload documents first.",
                "sources": [],
                "references": []
            }

        # Clean up expired cache entries
        self._cleanup_cache()

        # Check cache
        cache_key = f"{query}_{mode}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]['response']

        try:
            # Get relevant chunks using FAISS
            query_embedding = self.model.encode([query])[0].reshape(1, -1)
            k = 3 if mode == "advanced" else 1
            D, I = self.index.search(query_embedding.astype('float32'), k)

            relevant_texts = [self.texts[i] for i in I[0]]
            relevant_sources = [self.sources[i] for i in I[0]]

            # Prepare context for Groq with length limit
            context = "\n\n".join(relevant_texts)
            context = context[:8000]

            # Retry logic for Groq API
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting Groq API call (attempt {attempt + 1}/{max_retries}) with model {self.preferred_model}")
                    response = self.groq_client.chat.completions.create(
                        model=self.preferred_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say 'I cannot find the answer in the provided documents.'"
                            },
                            {
                                "role": "user",
                                "content": f"Context: {context}\n\nQuestion: {query}"
                            }
                        ],
                        temperature=0.7,
                        max_tokens=500,
                        timeout=30
                    )

                    answer = response.choices[0].message.content
                    result = {
                        "answer": answer,
                        "sources": relevant_sources if mode == "advanced" else [],
                        "references": relevant_texts if mode == "advanced" else []
                    }

                    # Cache the result
                    self.query_cache[cache_key] = {
                        'response': result,
                        'timestamp': time.time()
                    }

                    logger.info("Successfully processed query")
                    return result

                except Exception as e:
                    logger.error(f"Error in Groq API call (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)  # Wait before retry

        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while processing your question. Please try again with a simpler query.",
                "sources": [],
                "references": []
            }

    def is_initialized(self) -> bool:
        """Check if the processor has been initialized with documents."""
        return self.is_initialized_flag 