# fragrance_generator.py
# Purpose: Main orchestrator class that ties everything together
# What it does: Takes text input → returns complete fragrance formula
# Responsibilities: Coordinates all other modules, manages the full pipeline flow
# Input: "I want to smell like Elsa"
# Output: Complete formula with molecules and percentages

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from data_loader import DataLoader
from text_processor import TextProcessor
from molecule_selector import MoleculeSelector
from note_classifier import NoteClassifier
from formula_composer import FormulaComposer
from config import DEBUG, LOG_LEVEL

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FragranceGenerator:
    """
    Main pipeline class for generating fragrance formulas from text descriptions
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize FragranceGenerator
        
        Args:
            dataset_path: Path to molecule dataset CSV (uses config default if None)
        """
        logger.info("Initializing FragranceGenerator pipeline...")
        
        # Initialize all components
        self.data_loader = DataLoader(dataset_path)
        self.text_processor = TextProcessor()
        self.molecule_selector = None  # Will be initialized after data loading
        self.note_classifier = NoteClassifier()
        self.formula_composer = FormulaComposer()
        
        # Pipeline state
        self.is_initialized = False
        self.dataset_loaded = False
        
    def initialize(self):
        """Initialize the entire pipeline"""
        if self.is_initialized:
            logger.info("Pipeline already initialized")
            return
        
        try:
            logger.info("Loading dataset...")
            self.data_loader.load_dataset()
            self.dataset_loaded = True
            
            logger.info("Initializing text processor...")
            self.text_processor.initialize()
            
            logger.info("Initializing molecule selector...")
            self.molecule_selector = MoleculeSelector(self.data_loader)
            
            self.is_initialized = True
            logger.info("FragranceGenerator pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def generate_fragrance(self, text_input: str, 
                         fragrance_type: str = 'balanced',
                         custom_proportions: Optional[Dict[str, float]] = None,
                         include_alternatives: bool = False) -> Dict:
        """
        Main method: Generate complete fragrance formula from text input
        
        Args:
            text_input: Text description (e.g., "I want to smell like Elsa")
            fragrance_type: Type of fragrance ('balanced', 'fresh', 'oriental', 'floral')
            custom_proportions: Custom note proportions (overrides fragrance_type)
            include_alternatives: Whether to include alternative formulas
            
        Returns:
            Dict with complete fragrance generation results
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Generating fragrance for: '{text_input}'")
        
        try:
            # Step 1: Convert text to scent descriptors
            logger.info("Step 1: Processing text input...")
            descriptor_scores = self.text_processor.process_text(text_input)
            
            if not descriptor_scores:
                raise ValueError("No relevant scent descriptors found for input text")
            
            # Step 2: Find matching molecules
            logger.info("Step 2: Selecting molecules...")
            candidate_molecules = self.molecule_selector.select_molecules(descriptor_scores)
            
            if len(candidate_molecules) == 0:
                raise ValueError("No molecules found matching the selected descriptors")
            
            # Step 3: Classify molecules by note position
            logger.info("Step 3: Classifying note positions...")
            classified_molecules = self.note_classifier.classify_molecules(candidate_molecules)
            
            # Step 4: Compose final formula
            logger.info("Step 4: Composing formula...")
            formula = self.formula_composer.compose_formula(
                classified_molecules, 
                fragrance_type, 
                custom_proportions
            )
            
            # Generate comprehensive results
            results = self._compile_results(
                text_input, 
                descriptor_scores, 
                candidate_molecules, 
                classified_molecules, 
                formula,
                fragrance_type
            )
            
            # Add alternatives if requested
            if include_alternatives:
                logger.info("Generating alternative formulas...")
                alternatives = self.formula_composer.create_alternative_formulas(classified_molecules)
                results['alternatives'] = alternatives
            
            logger.info("Fragrance generation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error generating fragrance: {e}")
            raise
    
    def _compile_results(self, text_input: str, 
                        descriptor_scores: Dict[str, float],
                        candidate_molecules: pd.DataFrame,
                        classified_molecules: pd.DataFrame,
                        formula: Dict,
                        fragrance_type: str) -> Dict:
        """
        Compile comprehensive results from all pipeline steps
        
        Args:
            text_input: Original text input
            descriptor_scores: Selected descriptors and scores
            candidate_molecules: Candidate molecules before classification
            classified_molecules: Molecules after note classification
            formula: Final composed formula
            fragrance_type: Type of fragrance
            
        Returns:
            Comprehensive results dictionary
        """
        # Get detailed analysis from each step
        text_analysis = self.text_processor.get_detailed_analysis(text_input)
        selection_summary = self.molecule_selector.get_selection_summary(candidate_molecules, descriptor_scores)
        classification_summary = self.note_classifier.get_classification_summary(classified_molecules)
        
        results = {
            # Input information
            'input': {
                'text': text_input,
                'fragrance_type': fragrance_type,
                'processing_timestamp': pd.Timestamp.now().isoformat()
            },
            
            # Step-by-step analysis
            'text_analysis': text_analysis,
            'molecule_selection': {
                'summary': selection_summary,
                'selected_molecules': len(candidate_molecules),
                'molecules_by_descriptor': {
                    desc: len(self.data_loader.get_molecules_by_descriptor(desc))
                    for desc in descriptor_scores.keys()
                }
            },
            'note_classification': {
                'summary': classification_summary,
                'note_distribution': {
                    note: len(classified_molecules[classified_molecules['note_position'] == note])
                    for note in ['top', 'middle', 'base']
                }
            },
            
            # Final formula
            'formula': formula,
            'formula_display': self.formula_composer.format_formula_display(formula),
            
            # Quality metrics
            'quality_assessment': self._assess_formula_quality(formula, text_analysis),
            
            # Recommendations
            'recommendations': self._generate_recommendations(
                text_analysis, selection_summary, classification_summary, formula
            )
        }
        
        return results
    
    def _assess_formula_quality(self, formula: Dict, text_analysis: Dict) -> Dict:
        """
        Assess the quality of the generated formula
        
        Args:
            formula: Generated formula
            text_analysis: Text processing analysis
            
        Returns:
            Quality assessment dictionary
        """
        quality_scores = {}
        
        # Descriptor coverage score
        formula_descriptors = set()
        for component in formula['components']:
            formula_descriptors.update(component.get('matched_descriptors', []))
        
        selected_descriptors = set(text_analysis['selected_descriptors'].keys())
        coverage_score = len(formula_descriptors & selected_descriptors) / len(selected_descriptors) if selected_descriptors else 0
        quality_scores['descriptor_coverage'] = coverage_score * 100
        
        # Note balance score
        note_percentages = formula['metadata']['note_distribution']['percentages']
        ideal_balance = {'top': 30, 'middle': 50, 'base': 20}  # Ideal balanced profile
        
        balance_deviations = []
        for note, ideal_pct in ideal_balance.items():
            actual_pct = note_percentages.get(note, 0)
            deviation = abs(actual_pct - ideal_pct) / ideal_pct if ideal_pct > 0 else 0
            balance_deviations.append(deviation)
        
        balance_score = max(0, 100 - (sum(balance_deviations) / len(balance_deviations) * 100))
        quality_scores['note_balance'] = balance_score
        
        # Complexity score (based on number of components and descriptor diversity)
        num_components = formula['num_components']
        ideal_components = 8  # Ideal number of components
        complexity_score = max(0, 100 - abs(num_components - ideal_components) * 10)
        quality_scores['complexity'] = complexity_score
        
        # Relevance score (average relevance of selected molecules)
        avg_relevance = formula['metadata']['quality_metrics']['avg_relevance_score']
        relevance_score = min(100, avg_relevance * 100)  # Assuming relevance scores are 0-1
        quality_scores['relevance'] = relevance_score
        
        # Overall quality (weighted average)
        weights = {'descriptor_coverage': 0.3, 'note_balance': 0.25, 'complexity': 0.2, 'relevance': 0.25}
        overall_score = sum(weights[metric] * score for metric, score in quality_scores.items())
        quality_scores['overall'] = overall_score
        
        # Quality rating
        if overall_score >= 80:
            rating = 'Excellent'
        elif overall_score >= 65:
            rating = 'Good'
        elif overall_score >= 50:
            rating = 'Fair'
        else:
            rating = 'Needs Improvement'
        
        return {
            'scores': quality_scores,
            'overall_rating': rating,
            'overall_score': overall_score
        }
    
    def _generate_recommendations(self, text_analysis: Dict, 
                                selection_summary: Dict,
                                classification_summary: Dict,
                                formula: Dict) -> List[str]:
        """
        Generate recommendations for improving the formula
        
        Args:
            text_analysis: Text processing analysis
            selection_summary: Molecule selection summary
            classification_summary: Note classification summary
            formula: Final formula
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Text processing recommendations
        if text_analysis['analysis']['avg_similarity'] < 0.5:
            recommendations.append("Consider using more specific scent-related terms in your input for better results")
        
        # Molecule selection recommendations
        for issue in selection_summary.get('issues', []):
            recommendations.append(f"Selection: {issue}")
        
        # Formula composition recommendations
        for warning in formula['metadata'].get('warnings', []):
            recommendations.append(f"Formula: {warning}")
        
        # Note balance recommendations
        note_percentages = formula['metadata']['note_distribution']['percentages']
        if note_percentages.get('top', 0) < 15:
            recommendations.append("Consider adding more fresh/citrus elements for better opening impact")
        if note_percentages.get('base', 0) < 10:
            recommendations.append("Consider adding woody/musky elements for better longevity")
        
        # Complexity recommendations
        num_components = formula['num_components']
        if num_components < 4:
            recommendations.append("Formula could benefit from more components for greater complexity")
        elif num_components > 12:
            recommendations.append("Formula might be simplified by reducing the number of components")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def generate_batch(self, text_inputs: List[str], 
                      fragrance_type: str = 'balanced') -> List[Dict]:
        """
        Generate fragrance formulas for multiple text inputs
        
        Args:
            text_inputs: List of text descriptions
            fragrance_type: Type of fragrance for all formulas
            
        Returns:
            List of result dictionaries
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Generating batch of {len(text_inputs)} fragrances")
        
        results = []
        for i, text_input in enumerate(text_inputs):
            try:
                logger.info(f"Processing batch item {i+1}/{len(text_inputs)}: {text_input}")
                result = self.generate_fragrance(text_input, fragrance_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch item {i+1}: {e}")
                # Add error result
                results.append({
                    'input': {'text': text_input, 'fragrance_type': fragrance_type},
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def get_pipeline_status(self) -> Dict:
        """
        Get status information about the pipeline
        
        Returns:
            Dict with pipeline status
        """
        status = {
            'initialized': self.is_initialized,
            'dataset_loaded': self.dataset_loaded,
            'components': {
                'data_loader': self.data_loader is not None,
                'text_processor': self.text_processor is not None,
                'molecule_selector': self.molecule_selector is not None,
                'note_classifier': self.note_classifier is not None,
                'formula_composer': self.formula_composer is not None
            }
        }
        
        if self.dataset_loaded:
            status['dataset_stats'] = self.data_loader.get_dataset_stats()
            status['embedding_model'] = self.text_processor.embedding_manager.get_model_info()
        
        return status
    
    def validate_input(self, text_input: str) -> Tuple[bool, str]:
        """
        Validate input text before processing
        
        Args:
            text_input: Input text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text_input or not text_input.strip():
            return False, "Input text cannot be empty"
        
        if len(text_input.strip()) < 3:
            return False, "Input text too short - please provide more description"
        
        if len(text_input) > 500:
            return False, "Input text too long - please limit to 500 characters"
        
        return True, "Valid input"