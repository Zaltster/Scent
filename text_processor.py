# text_processor.py
# Purpose: Converts text descriptions into scent descriptors
# What it does: Uses embeddings to find similarity between input text and your 138 descriptors
# Responsibilities: Load embedding model, compute similarities, rank descriptors
# Input: "Elsa" or "Disney princess"
# Output: ["fresh", "clean", "powdery", "ethereal"] with similarity scores

import logging
from typing import Dict, List, Tuple, Optional
from embedding_manager import EmbeddingManager
from config import EMBEDDING_SETTINGS, DESCRIPTOR_COLUMNS

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Converts natural language text into scent descriptor mappings using AI embeddings
    """
    
    def __init__(self):
        """Initialize TextProcessor with embedding manager"""
        self.embedding_manager = EmbeddingManager()
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the embedding model and precompute descriptor embeddings"""
        if not self.is_initialized:
            logger.info("Initializing TextProcessor...")
            self.embedding_manager.load_model()
            self.embedding_manager.precompute_descriptor_embeddings()
            self.is_initialized = True
            logger.info("TextProcessor initialized successfully")
    
    def process_text(self, text: str) -> Dict[str, float]:
        """
        Main method: Convert text to scent descriptors with similarity scores
        
        Args:
            text: Input text (e.g., "I want to smell like Elsa")
            
        Returns:
            Dict mapping descriptor names to similarity scores
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Processing text: '{text}'")
        
        # Clean and preprocess text
        processed_text = self._preprocess_text(text)
        
        # Get descriptor mappings using embeddings
        descriptor_scores = self.embedding_manager.map_text_to_descriptors(processed_text)
        
        logger.info(f"Found {len(descriptor_scores)} relevant descriptors")
        
        return descriptor_scores
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess input text for better embedding results
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove common prefixes that don't help with scent mapping
        prefixes_to_remove = [
            "i want to smell like",
            "make me smell like", 
            "i want a fragrance that smells like",
            "create a scent like",
            "perfume that smells like",
            "fragrance inspired by"
        ]
        
        text_lower = text.lower()
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        return text
    
    def get_detailed_analysis(self, text: str, top_k: int = 15) -> Dict:
        """
        Get detailed analysis of text to descriptor mapping
        
        Args:
            text: Input text
            top_k: Number of top descriptors to analyze
            
        Returns:
            Dict with detailed analysis
        """
        if not self.is_initialized:
            self.initialize()
        
        processed_text = self._preprocess_text(text)
        
        # Get more descriptors for analysis
        all_similarities = self.embedding_manager.find_similar_descriptors(
            processed_text, top_k=top_k, threshold=0.1
        )
        
        # Separate selected vs. not selected descriptors
        threshold = EMBEDDING_SETTINGS['similarity_threshold']
        max_descriptors = EMBEDDING_SETTINGS['top_descriptors_count']
        
        selected = []
        candidates = []
        rejected = []
        
        for desc, score in all_similarities:
            if len(selected) < max_descriptors and score >= threshold:
                selected.append((desc, score))
            elif score >= threshold:
                candidates.append((desc, score))
            else:
                rejected.append((desc, score))
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'selected_descriptors': dict(selected),
            'candidate_descriptors': dict(candidates),
            'rejected_descriptors': dict(rejected[:10]),  # Top 10 rejected
            'analysis': {
                'num_selected': len(selected),
                'avg_similarity': sum(score for _, score in selected) / len(selected) if selected else 0,
                'max_similarity': max(score for _, score in selected) if selected else 0,
                'min_similarity': min(score for _, score in selected) if selected else 0
            }
        }
    
    def explain_selection(self, text: str) -> str:
        """
        Generate human-readable explanation of descriptor selection
        
        Args:
            text: Input text
            
        Returns:
            Formatted explanation string
        """
        analysis = self.get_detailed_analysis(text)
        
        explanation = f"Analysis for: '{text}'\n"
        explanation += f"Processed as: '{analysis['processed_text']}'\n\n"
        
        explanation += "Selected Descriptors:\n"
        for desc, score in analysis['selected_descriptors'].items():
            explanation += f"  • {desc:<15} (similarity: {score:.3f})\n"
        
        if analysis['candidate_descriptors']:
            explanation += f"\nOther Strong Candidates (above threshold but not selected):\n"
            for desc, score in list(analysis['candidate_descriptors'].items())[:5]:
                explanation += f"  • {desc:<15} (similarity: {score:.3f})\n"
        
        explanation += f"\nSummary:\n"
        explanation += f"  • Selected {analysis['analysis']['num_selected']} descriptors\n"
        explanation += f"  • Average similarity: {analysis['analysis']['avg_similarity']:.3f}\n"
        explanation += f"  • Strongest match: {analysis['analysis']['max_similarity']:.3f}\n"
        
        return explanation
    
    def suggest_alternatives(self, text: str) -> List[str]:
        """
        Suggest alternative ways to phrase the input for better results
        
        Args:
            text: Input text
            
        Returns:
            List of suggested alternative phrasings
        """
        suggestions = []
        
        # Analyze current results
        analysis = self.get_detailed_analysis(text)
        avg_similarity = analysis['analysis']['avg_similarity']
        
        # If similarity is low, suggest more specific terms
        if avg_similarity < 0.4:
            suggestions.append("Try being more specific about scent qualities (e.g., 'fresh and clean' instead of just a name)")
            suggestions.append("Add descriptive words like 'floral', 'woody', 'citrus', 'sweet', or 'spicy'")
            suggestions.append("Describe the mood or feeling you want (e.g., 'romantic and elegant')")
        
        # If too generic, suggest refinement
        selected_descriptors = list(analysis['selected_descriptors'].keys())
        if 'sweet' in selected_descriptors and len(selected_descriptors) == 1:
            suggestions.append("'Sweet' is very broad - try adding what kind of sweet (fruity, vanilla, honey, etc.)")
        
        # Suggest specific improvements based on content
        text_lower = text.lower()
        if any(word in text_lower for word in ['character', 'person', 'disney']):
            suggestions.append("Instead of character names, describe their personality or visual aesthetics")
            suggestions.append("Think about what scents match their story or environment")
        
        return suggestions
    
    def batch_process(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Process multiple texts efficiently
        
        Args:
            texts: List of input texts
            
        Returns:
            List of descriptor mappings for each text
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Batch processing {len(texts)} texts")
        
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Use embedding manager's batch processing
        results = self.embedding_manager.batch_process_texts(processed_texts)
        
        return results
    
    def get_descriptor_usage_stats(self, texts: List[str]) -> Dict:
        """
        Analyze which descriptors are most commonly selected across multiple texts
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            Dict with usage statistics
        """
        results = self.batch_process(texts)
        
        # Count descriptor usage
        descriptor_counts = {}
        descriptor_scores = {}
        
        for result in results:
            for desc, score in result.items():
                descriptor_counts[desc] = descriptor_counts.get(desc, 0) + 1
                if desc not in descriptor_scores:
                    descriptor_scores[desc] = []
                descriptor_scores[desc].append(score)
        
        # Calculate statistics
        stats = {}
        for desc in descriptor_counts:
            scores = descriptor_scores[desc]
            stats[desc] = {
                'usage_count': descriptor_counts[desc],
                'usage_percentage': descriptor_counts[desc] / len(texts) * 100,
                'avg_similarity': sum(scores) / len(scores),
                'max_similarity': max(scores),
                'min_similarity': min(scores)
            }
        
        # Sort by usage count
        sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1]['usage_count'], reverse=True))
        
        return {
            'total_texts': len(texts),
            'descriptor_stats': sorted_stats,
            'most_common': list(sorted_stats.keys())[:10],
            'summary': {
                'unique_descriptors_used': len(descriptor_counts),
                'avg_descriptors_per_text': sum(len(r) for r in results) / len(results)
            }
        }