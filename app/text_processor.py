from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        self.vector_index = None
        self.documents = []

    def process_text(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process text content into chunks and generate embeddings"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Generate embeddings for each chunk
        embeddings = self.model.encode(chunks)
        
        # Store chunks with their embeddings and metadata
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                "chunk_id": f"{metadata.get('doc_id', 'unknown')}_{i}",
                "content": chunk,
                "embedding": embedding.tolist(),
                "metadata": metadata
            })
        
        return processed_chunks

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from document chunks"""
        if not chunks:
            return
        
        # Extract embeddings
        embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        
        # Create and train index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings)
        
        # Store document metadata
        self.documents = chunks

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks using semantic similarity"""
        if not self.vector_index:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in FAISS index
        distances, indices = self.vector_index.search(
            np.array([query_embedding], dtype=np.float32), 
            k
        )
        
        # Return relevant chunks
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result["score"] = float(distance)
                results.append(result)
        
        return results 