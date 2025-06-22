# evaluator.py
# Purpose: Validates pipeline output quality
# What it does: Checks if generated formulas make chemical/perfumery sense
# Responsibilities: Descriptor consistency, proportion validation, chemical compatibility
# Input: Generated formulas
# Output: Quality scores and validation reports

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from molecular_utils import MolecularUtils
from config import VALIDATION, PROPORTION_PROFILES

logger = logging.getLogger(__name__)

class FormulaEvaluator:
    """
    Evaluates and validates generated fragrance formulas
    """
    
    def __init__(self):
        """Initialize FormulaEvaluator"""
        self.molecular_utils = MolecularUtils()
        
    def evaluate_formula(self, formula: Dict, original_text: str = "") -> Dict:
        """
        Main evaluation method for a complete formula
        
        Args:
            formula: Generated formula dictionary
            original_text: Original input text for context
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating formula with {formula.get('num_components', 0)} components")
        
        evaluation = {
            'formula_id': formula.get('formula_id', 'unknown'),
            'original_text': original_text,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'scores': {},
            'validations': {},
            'recommendations': [],
            'overall_rating': 'Unknown'
        }
        
        # Core evaluations
        evaluation['scores'] = self._calculate_quality_scores(formula)
        evaluation['validations'] = self._run_validation_checks(formula)
        evaluation['recommendations'] = self._generate_evaluation_recommendations(formula)
        evaluation['overall_rating'] = self._calculate_overall_rating(evaluation['scores'])
        
        return evaluation
    
    def _calculate_quality_scores(self, formula: Dict) -> Dict[str, float]:
        """
        Calculate various quality scores for the formula
        
        Args:
            formula: Formula dictionary
            
        Returns:
            Dict with quality scores (0-100 scale)
        """
        scores = {}
        components = formula.get('components', [])
        
        if not components:
            return {score_type: 0 for score_type in ['chemical_validity', 'note_balance', 'complexity', 'diversity']}
        
        # 1. Chemical Validity Score
        scores['chemical_validity'] = self._score_chemical_validity(components)
        
        # 2. Note Balance Score
        scores['note_balance'] = self._score_note_balance(formula)
        
        # 3. Complexity Score
        scores['complexity'] = self._score_complexity(components)
        
        # 4. Diversity Score
        scores['diversity'] = self._score_molecular_diversity(components)
        
        # 5. Realism Score (based on actual perfumery practices)
        scores['realism'] = self._score_perfumery_realism(components)
        
        # 6. Descriptor Consistency Score
        scores['descriptor_consistency'] = self._score_descriptor_consistency(components)
        
        return scores
    
    def _score_chemical_validity(self, components: List[Dict]) -> float:
        """Score based on chemical validity and compatibility"""
        if not components:
            return 0
        
        valid_count = 0
        total_count = len(components)
        
        # Check individual molecule validity
        for component in components:
            smiles = component.get('smiles', '')
            is_valid, _ = self.molecular_utils.validate_smiles(smiles)
            if is_valid:
                valid_count += 1
        
        validity_score = (valid_count / total_count) * 100
        
        # Check for chemical incompatibilities
        smiles_list = [c.get('smiles', '') for c in components]
        compatibility_warnings = self.molecular_utils.check_chemical_compatibility(smiles_list)
        
        # Deduct points for incompatibilities
        total_warnings = sum(len(warnings) for warnings in compatibility_warnings.values())
        compatibility_penalty = min(30, total_warnings * 5)  # Max 30 point penalty
        
        return max(0, validity_score - compatibility_penalty)
    
    def _score_note_balance(self, formula: Dict) -> float:
        """Score based on note position balance"""
        note_percentages = formula.get('metadata', {}).get('note_distribution', {}).get('percentages', {})
        
        if not note_percentages:
            return 0
        
        # Ideal ranges for different fragrance types
        fragrance_type = formula.get('fragrance_type', 'balanced')
        ideal_ranges = {
            'balanced': {'top': (25, 35), 'middle': (45, 55), 'base': (15, 25)},
            'fresh': {'top': (35, 45), 'middle': (40, 50), 'base': (10, 20)},
            'oriental': {'top': (15, 25), 'middle': (35, 45), 'base': (35, 45)},
            'floral': {'top': (20, 30), 'middle': (50, 60), 'base': (15, 25)}
        }
        
        ranges = ideal_ranges.get(fragrance_type, ideal_ranges['balanced'])
        
        score = 100
        for note, (min_pct, max_pct) in ranges.items():
            actual_pct = note_percentages.get(note, 0)
            
            if actual_pct < min_pct:
                penalty = (min_pct - actual_pct) * 2
            elif actual_pct > max_pct:
                penalty = (actual_pct - max_pct) * 2
            else:
                penalty = 0
            
            score -= penalty
        
        return max(0, score)
    
    def _score_complexity(self, components: List[Dict]) -> float:
        """Score based on formula complexity"""
        num_components = len(components)
        
        # Ideal range: 6-10 components
        if 6 <= num_components <= 10:
            return 100
        elif 4 <= num_components <= 12:
            return 80
        elif 3 <= num_components <= 15:
            return 60
        elif num_components >= 2:
            return 40
        else:
            return 0
    
    def _score_molecular_diversity(self, components: List[Dict]) -> float:
        """Score based on molecular diversity"""
        if len(components) < 2:
            return 0
        
        # Check molecular weight diversity
        mw_values = [c.get('molecular_weight', 0) for c in components if c.get('molecular_weight')]
        
        if len(mw_values) < 2:
            return 50  # Neutral score if we can't assess diversity
        
        mw_range = max(mw_values) - min(mw_values)
        mw_diversity_score = min(100, (mw_range / 300) * 100)  # 300 is good MW range
        
        # Check descriptor diversity
        all_descriptors = set()
        for component in components:
            descriptors = component.get('matched_descriptors', [])
            all_descriptors.update(descriptors)
        
        descriptor_diversity_score = min(100, len(all_descriptors) * 10)  # 10+ descriptors = 100 score
        
        # Average the two diversity measures
        return (mw_diversity_score + descriptor_diversity_score) / 2
    
    def _score_perfumery_realism(self, components: List[Dict]) -> float:
        """Score based on realistic perfumery practices"""
        score = 100
        
        # Check for realistic percentages
        for component in components:
            percentage = component.get('percentage', 0)
            
            # Very high percentages are unrealistic for most molecules
            if percentage > 40:
                score -= 20
            elif percentage > 25:
                score -= 10
            
            # Very low percentages might be ineffective
            if percentage < 0.5:
                score -= 5
        
        # Check for balanced note representation
        note_counts = {}
        for component in components:
            note = component.get('note_position', 'unknown')
            note_counts[note] = note_counts.get(note, 0) + 1
        
        # Penalize if missing any note category
        for note in ['top', 'middle', 'base']:
            if note_counts.get(note, 0) == 0:
                score -= 15
        
        return max(0, score)
    
    def _score_descriptor_consistency(self, components: List[Dict]) -> float:
        """Score based on descriptor consistency within the formula"""
        all_descriptors = []
        for component in components:
            descriptors = component.get('matched_descriptors', [])
            all_descriptors.extend(descriptors)
        
        if not all_descriptors:
            return 0
        
        # Count descriptor frequencies
        descriptor_counts = {}
        for desc in all_descriptors:
            descriptor_counts[desc] = descriptor_counts.get(desc, 0) + 1
        
        # Check for good coverage of key descriptor categories
        key_categories = {
            'fresh': ['fresh', 'clean', 'citrus', 'ozone', 'aldehydic'],
            'floral': ['floral', 'rose', 'jasmin', 'lavender', 'violet'],
            'woody': ['woody', 'cedar', 'sandalwood', 'pine'],
            'spicy': ['spicy', 'cinnamon', 'clove', 'pepper'],
            'sweet': ['sweet', 'vanilla', 'honey', 'caramellic']
        }
        
        covered_categories = 0
        for category, descriptors in key_categories.items():
            if any(desc in descriptor_counts for desc in descriptors):
                covered_categories += 1
        
        consistency_score = (covered_categories / len(key_categories)) * 100
        
        return consistency_score
    
    def _run_validation_checks(self, formula: Dict) -> Dict:
        """Run various validation checks on the formula"""
        validations = {
            'structure_validation': self._validate_formula_structure(formula),
            'percentage_validation': self._validate_percentages(formula),
            'chemical_validation': self._validate_chemistry(formula),
            'perfumery_validation': self._validate_perfumery_rules(formula)
        }
        
        return validations
    
    def _validate_formula_structure(self, formula: Dict) -> Dict:
        """Validate basic formula structure"""
        validation = {'passed': True, 'issues': []}
        
        components = formula.get('components', [])
        
        # Check minimum/maximum components
        min_components = VALIDATION['min_formula_components']
        max_components = VALIDATION['max_formula_components']
        
        if len(components) < min_components:
            validation['passed'] = False
            validation['issues'].append(f"Too few components: {len(components)} < {min_components}")
        
        if len(components) > max_components:
            validation['passed'] = False
            validation['issues'].append(f"Too many components: {len(components)} > {max_components}")
        
        # Check for required fields
        for i, component in enumerate(components):
            required_fields = ['smiles', 'percentage', 'note_position']
            for field in required_fields:
                if field not in component or component[field] is None:
                    validation['issues'].append(f"Component {i+1} missing required field: {field}")
                    validation['passed'] = False
        
        return validation
    
    def _validate_percentages(self, formula: Dict) -> Dict:
        """Validate percentage calculations"""
        validation = {'passed': True, 'issues': []}
        
        components = formula.get('components', [])
        if not components:
            return validation
        
        total_percentage = sum(c.get('percentage', 0) for c in components)
        
        # Check total percentage
        if total_percentage < 95:
            validation['passed'] = False
            validation['issues'].append(f"Total percentage too low: {total_percentage:.1f}%")
        
        if total_percentage > 105:
            validation['passed'] = False
            validation['issues'].append(f"Total percentage too high: {total_percentage:.1f}%")
        
        # Check individual percentages
        for i, component in enumerate(components):
            percentage = component.get('percentage', 0)
            
            if percentage <= 0:
                validation['issues'].append(f"Component {i+1} has zero or negative percentage")
                validation['passed'] = False
            
            if percentage > 50:
                validation['issues'].append(f"Component {i+1} has unusually high percentage: {percentage:.1f}%")
        
        return validation
    
    def _validate_chemistry(self, formula: Dict) -> Dict:
        """Validate chemical aspects of the formula"""
        validation = {'passed': True, 'issues': []}
        
        components = formula.get('components', [])
        if not components:
            return validation
        
        # Check SMILES validity
        for i, component in enumerate(components):
            smiles = component.get('smiles', '')
            is_valid, error_msg = self.molecular_utils.validate_smiles(smiles)
            
            if not is_valid:
                validation['passed'] = False
                validation['issues'].append(f"Component {i+1} has invalid SMILES: {error_msg}")
        
        # Check for chemical incompatibilities
        if VALIDATION['check_chemical_compatibility']:
            smiles_list = [c.get('smiles', '') for c in components]
            compatibility_warnings = self.molecular_utils.check_chemical_compatibility(smiles_list)
            
            for category, warnings in compatibility_warnings.items():
                if warnings:
                    validation['issues'].extend([f"Chemical compatibility ({category}): {w}" for w in warnings])
                    if category in ['reactive_combinations', 'ph_issues']:
                        validation['passed'] = False
        
        return validation
    
    def _validate_perfumery_rules(self, formula: Dict) -> Dict:
        """Validate against standard perfumery practices"""
        validation = {'passed': True, 'issues': []}
        
        # Check note distribution
        note_percentages = formula.get('metadata', {}).get('note_distribution', {}).get('percentages', {})
        
        if note_percentages.get('top', 0) < 5:
            validation['issues'].append("Very low top note percentage - may lack opening impact")
        
        if note_percentages.get('base', 0) < 5:
            validation['issues'].append("Very low base note percentage - may lack longevity")
        
        if note_percentages.get('middle', 0) < 30:
            validation['issues'].append("Low middle note percentage - may lack body")
        
        # Check for molecular weight distribution
        components = formula.get('components', [])
        mw_values = [c.get('molecular_weight', 0) for c in components if c.get('molecular_weight')]
        
        if mw_values:
            if all(mw < 150 for mw in mw_values):
                validation['issues'].append("All molecules are light - formula may lack depth")
            
            if all(mw > 250 for mw in mw_values):
                validation['issues'].append("All molecules are heavy - formula may lack freshness")
        
        return validation
    
    def _generate_evaluation_recommendations(self, formula: Dict) -> List[str]:
        """Generate specific recommendations for formula improvement"""
        recommendations = []
        
        # Add recommendations based on scores and validations
        # This would be expanded based on specific evaluation results
        
        return recommendations
    
    def _calculate_overall_rating(self, scores: Dict[str, float]) -> str:
        """Calculate overall quality rating"""
        if not scores:
            return 'Unknown'
        
        # Weighted average of scores
        weights = {
            'chemical_validity': 0.25,
            'note_balance': 0.20,
            'complexity': 0.15,
            'diversity': 0.15,
            'realism': 0.15,
            'descriptor_consistency': 0.10
        }
        
        weighted_score = sum(weights.get(metric, 0) * score for metric, score in scores.items())
        
        if weighted_score >= 85:
            return 'Excellent'
        elif weighted_score >= 70:
            return 'Good'
        elif weighted_score >= 55:
            return 'Fair'
        elif weighted_score >= 40:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def compare_formulas(self, formulas: List[Dict]) -> Dict:
        """Compare multiple formulas and rank them"""
        if not formulas:
            return {'error': 'No formulas to compare'}
        
        evaluations = []
        for i, formula in enumerate(formulas):
            evaluation = self.evaluate_formula(formula)
            evaluation['formula_index'] = i
            evaluations.append(evaluation)
        
        # Rank by overall score
        evaluations.sort(key=lambda x: sum(x['scores'].values()), reverse=True)
        
        return {
            'rankings': evaluations,
            'best_formula': evaluations[0] if evaluations else None,
            'comparison_summary': self._generate_comparison_summary(evaluations)
        }
    
    def _generate_comparison_summary(self, evaluations: List[Dict]) -> Dict:
        """Generate summary of formula comparison"""
        if not evaluations:
            return {}
        
        # Calculate statistics across all formulas
        all_scores = {}
        for evaluation in evaluations:
            for metric, score in evaluation['scores'].items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        summary = {}
        for metric, scores in all_scores.items():
            summary[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return summary