"""
Simple Vector Database for Chat Context and Embeddings
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorDB:
    """Simple in-memory vector database for chat context"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize vector database"""
        self.data_dir = data_dir
        self.vectors = {}
        self.metadata = {}
        self.embeddings = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Try to load existing data
        self._load_data()
        
        logger.info("Vector DB initialized")
    
    def add_vector(self, key: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Add vector with metadata"""
        self.vectors[key] = vector
        self.metadata[key] = metadata or {}
        self.metadata[key]['timestamp'] = datetime.now().isoformat()
        
        # Save to disk
        self._save_data()
    
    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors"""
        if not self.vectors:
            return []
        
        similarities = []
        
        for key, vector in self.vectors.items():
            # Simple cosine similarity (placeholder)
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((key, similarity, self.metadata.get(key, {})))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_vector(self, key: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get vector and metadata by key"""
        if key in self.vectors:
            return self.vectors[key], self.metadata.get(key, {})
        return None
    
    def delete_vector(self, key: str) -> bool:
        """Delete vector by key"""
        if key in self.vectors:
            del self.vectors[key]
            if key in self.metadata:
                del self.metadata[key]
            self._save_data()
            return True
        return False
    
    def list_vectors(self) -> List[str]:
        """List all vector keys"""
        return list(self.vectors.keys())
    
    def clear(self):
        """Clear all vectors"""
        self.vectors.clear()
        self.metadata.clear()
        self._save_data()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _save_data(self):
        """Save data to disk"""
        try:
            data = {
                'vectors': self.vectors,
                'metadata': self.metadata
            }
            
            file_path = os.path.join(self.data_dir, 'vector_db.json')
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving vector DB data: {e}")
    
    def _load_data(self):
        """Load data from disk"""
        try:
            file_path = os.path.join(self.data_dir, 'vector_db.json')
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.vectors = data.get('vectors', {})
                self.metadata = data.get('metadata', {})
                
                logger.info(f"Loaded {len(self.vectors)} vectors from disk")
            
        except Exception as e:
            logger.error(f"Error loading vector DB data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_vectors': len(self.vectors),
            'data_dir': self.data_dir,
            'latest_update': max([meta.get('timestamp', '') for meta in self.metadata.values()]) if self.metadata else None
        }