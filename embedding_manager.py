# embedding_manager.py
# Purpose: Manages text embeddings and similarity calculations
# What it does: Loads pre-trained model, creates embeddings, computes cosine similarity
# Responsibilities: Model loading, text → vector conversion, similarity ranking
# Input: Text strings and descriptor lists
# Output: Similarity scores between text and descriptors

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import EMBEDDING_SETTINGS, DESCRIPTOR_COLUMNS

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Handles text embeddings and similarity calculations for mapping text to scent descriptors
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize EmbeddingManager
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model_name = model_name or EMBEDDING_SETTINGS['model_name']
        self.model = None
        self.descriptor_embeddings = None
        self.descriptor_names = DESCRIPTOR_COLUMNS
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def precompute_descriptor_embeddings(self):
        """
        Precompute embeddings for all scent descriptors for efficiency
        """
        if self.model is None:
            self.load_model()
        
        logger.info("Precomputing embeddings for scent descriptors...")
        
        # Create embeddings for all descriptors
        self.descriptor_embeddings = self.model.encode(
            self.descriptor_names,
            show_progress_bar=True
        )
        
        logger.info(f"Precomputed embeddings for {len(self.descriptor_names)} descriptors")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for input text
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            self.load_model()
        
        return self.model.encode([text])[0]
    
    def find_similar_descriptors(self, text: str, 
                                top_k: Optional[int] = None,
                                threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Find scent descriptors most similar to input text
        
        Args:
            text: Input text (e.g., "I want to smell like Elsa")
            top_k: Number of top descriptors to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (descriptor, similarity_score) tuples, sorted by similarity
        """
        if self.descriptor_embeddings is None:
            self.precompute_descriptor_embeddings()
        
        # Get embedding for input text
        text_embedding = self.get_text_embedding(text)
        
        # Calculate similarities
        similarities = cosine_similarity(
            [text_embedding], 
            self.descriptor_embeddings
        )[0]
        
        # Create (descriptor, similarity) pairs
        descriptor_similarities = list(zip(self.descriptor_names, similarities))
        
        # Sort by similarity (highest first)
        descriptor_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply threshold if specified
        if threshold is not None:
            descriptor_similarities = [
                (desc, sim) for desc, sim in descriptor_similarities
                if sim >= threshold
            ]
        
        # Apply top_k if specified
        if top_k is not None:
            descriptor_similarities = descriptor_similarities[:top_k]
        
        return descriptor_similarities
    
    def map_text_to_descriptors(self, text: str) -> Dict[str, float]:
        """
        Map input text to scent descriptors with similarity scores
        
        Args:
            text: Input text description
            
        Returns:
            Dict mapping descriptor names to similarity scores
        """
        top_k = EMBEDDING_SETTINGS['top_descriptors_count']
        threshold = EMBEDDING_SETTINGS['similarity_threshold']
        
        similar_descriptors = self.find_similar_descriptors(
            text, top_k=top_k, threshold=threshold
        )
        
        return dict(similar_descriptors)
    
    def explain_mapping(self, text: str, top_k: int = 10) -> str:
        """
        Generate human-readable explanation of text to descriptor mapping
        
        Args:
            text: Input text
            top_k: Number of top descriptors to explain
            
        Returns:
            Formatted explanation string
        """
        similar_descriptors = self.find_similar_descriptors(text, top_k=top_k)
        
        explanation = f"Text: '{text}'\n"
        explanation += "Most similar scent descriptors:\n"
        
        for i, (descriptor, similarity) in enumerate(similar_descriptors, 1):
            explanation += f"{i:2d}. {descriptor:<15} (similarity: {similarity:.3f})\n"
        
        return explanation
    
    def batch_process_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Process multiple texts at once for efficiency
        
        Args:
            texts: List of input texts
            
        Returns:
            List of descriptor mappings for each text
        """
        if self.descriptor_embeddings is None:
            self.precompute_descriptor_embeddings()
        
        # Get embeddings for all texts
        text_embeddings = self.model.encode(texts, show_progress_bar=True)
        
        results = []
        top_k = EMBEDDING_SETTINGS['top_descriptors_count']
        threshold = EMBEDDING_SETTINGS['similarity_threshold']
        
        for text_embedding in text_embeddings:
            # Calculate similarities
            similarities = cosine_similarity(
                [text_embedding], 
                self.descriptor_embeddings
            )[0]
            
            # Create descriptor mapping
            descriptor_similarities = list(zip(self.descriptor_names, similarities))
            descriptor_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply filters
            filtered = [
                (desc, sim) for desc, sim in descriptor_similarities[:top_k]
                if sim >= threshold
            ]
            
            results.append(dict(filtered))
        
        return results
    
    def find_descriptor_clusters(self, threshold: float = 0.8) -> Dict[str, List[str]]:
        """
        Find clusters of similar descriptors
        
        Args:
            threshold: Similarity threshold for clustering
            
        Returns:
            Dict mapping cluster representative to list of similar descriptors
        """
        if self.descriptor_embeddings is None:
            self.precompute_descriptor_embeddings()
        
        # Calculate pairwise similarities between descriptors
        similarities = cosine_similarity(self.descriptor_embeddings)
        
        clusters = {}
        used_descriptors = set()
        
        for i, descriptor in enumerate(self.descriptor_names):
            if descriptor in used_descriptors:
                continue
            
            # Find all descriptors similar to this one
            similar_indices = np.where(similarities[i] >= threshold)[0]
            similar_descriptors = [
                self.descriptor_names[j] for j in similar_indices
                if self.descriptor_names[j] not in used_descriptors
            ]
            
            if len(similar_descriptors) > 1:
                clusters[descriptor] = similar_descriptors
                used_descriptors.update(similar_descriptors)
        
        return clusters
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dict with model information
        """
        if self.model is None:
            return {"model_loaded": False}
        
        return {
            "model_loaded": True,
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'Unknown'),
            "descriptors_precomputed": self.descriptor_embeddings is not None,
            "num_descriptors": len(self.descriptor_names)
        }