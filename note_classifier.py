# note_classifier.py
# Purpose: Classifies molecules into top/middle/base notes
# What it does: Calculates molecular weight from SMILES, applies thresholds
# Responsibilities: SMILES → molecular weight, weight → note position classification
# Input: List of SMILES strings
# Output: Same molecules tagged as "top", "middle", or "base"

import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional
from molecular_utils import MolecularUtils
from config import MW_THRESHOLDS

logger = logging.getLogger(__name__)

class NoteClassifier:
    """
    Classifies fragrance molecules into note positions (top/middle/base) based on molecular properties
    """
    
    def __init__(self):
        """Initialize NoteClassifier"""
        self.molecular_utils = MolecularUtils()
        
    def classify_molecules(self, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method: Classify molecules by note position
        
        Args:
            molecules_df: DataFrame with molecules (must have 'smiles' column)
            
        Returns:
            DataFrame with added 'note_position' and molecular property columns
        """
        logger.info(f"Classifying {len(molecules_df)} molecules by note position")
        
        result_df = molecules_df.copy()
        
        # Calculate molecular properties if not already present
        if 'molecular_weight' not in result_df.columns:
            result_df = self._add_molecular_properties(result_df)
        
        # Apply note position classification
        result_df['note_position'] = result_df.apply(
            lambda row: self._classify_single_molecule(row), axis=1
        )
        
        # Add volatility estimates
        result_df['volatility_score'] = result_df['smiles'].apply(
            self.molecular_utils.estimate_volatility
        )
        
        # Log classification results
        self._log_classification_results(result_df)
        
        return result_df
    
    def _add_molecular_properties(self, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add molecular properties to DataFrame
        
        Args:
            molecules_df: DataFrame with molecules
            
        Returns:
            DataFrame with added molecular properties
        """
        properties_list = []
        
        for smiles in molecules_df['smiles']:
            properties = self.molecular_utils.calculate_molecular_properties(smiles)
            if properties:
                properties_list.append(properties)
            else:
                # Add empty properties for invalid molecules
                properties_list.append({
                    'molecular_weight': None,
                    'logp': None,
                    'note_position': None,
                    'num_atoms': None,
                    'num_heavy_atoms': None,
                    'num_rotatable_bonds': None,
                    'tpsa': None
                })
        
        # Add properties to DataFrame
        properties_df = pd.DataFrame(properties_list)
        result_df = pd.concat([molecules_df.reset_index(drop=True), properties_df], axis=1)
        
        return result_df
    
    def _classify_single_molecule(self, molecule_row: pd.Series) -> Optional[str]:
        """
        Classify a single molecule's note position
        
        Args:
            molecule_row: Row from DataFrame with molecule data
            
        Returns:
            Note position ('top', 'middle', 'base') or None if invalid
        """
        smiles = molecule_row['smiles']
        mw = molecule_row.get('molecular_weight')
        
        # Check for descriptor-based overrides first
        if 'matched_descriptors' in molecule_row:
            override = self.molecular_utils.suggest_note_position_override(
                smiles, molecule_row['matched_descriptors']
            )
            if override:
                return override
        
        # Use molecular weight classification
        if mw is None:
            return None
        
        if mw < MW_THRESHOLDS['top_max']:
            return 'top'
        elif mw < MW_THRESHOLDS['middle_max']:
            return 'middle'
        else:
            return 'base'
    
    def _log_classification_results(self, classified_df: pd.DataFrame):
        """Log summary of classification results"""
        note_counts = classified_df['note_position'].value_counts()
        
        logger.info("Note position classification results:")
        for note_position in ['top', 'middle', 'base']:
            count = note_counts.get(note_position, 0)
            percentage = count / len(classified_df) * 100 if len(classified_df) > 0 else 0
            logger.info(f"  {note_position.capitalize()}: {count} molecules ({percentage:.1f}%)")
        
        invalid_count = classified_df['note_position'].isna().sum()
        if invalid_count > 0:
            logger.warning(f"  Invalid: {invalid_count} molecules could not be classified")
    
    def get_classification_summary(self, classified_df: pd.DataFrame) -> Dict:
        """
        Get detailed summary of classification results
        
        Args:
            classified_df: DataFrame with classified molecules
            
        Returns:
            Dict with classification statistics
        """
        summary = {}
        
        # Note position distribution
        note_counts = classified_df['note_position'].value_counts()
        total_valid = len(classified_df[classified_df['note_position'].notna()])
        
        summary['note_distribution'] = {}
        for note_position in ['top', 'middle', 'base']:
            count = note_counts.get(note_position, 0)
            percentage = count / total_valid * 100 if total_valid > 0 else 0
            summary['note_distribution'][note_position] = {
                'count': count,
                'percentage': percentage
            }
        
        # Molecular weight statistics by note position
        summary['molecular_weight_stats'] = {}
        for note_position in ['top', 'middle', 'base']:
            position_molecules = classified_df[classified_df['note_position'] == note_position]
            if len(position_molecules) > 0:
                mw_values = position_molecules['molecular_weight'].dropna()
                if len(mw_values) > 0:
                    summary['molecular_weight_stats'][note_position] = {
                        'mean': float(mw_values.mean()),
                        'min': float(mw_values.min()),
                        'max': float(mw_values.max()),
                        'std': float(mw_values.std())
                    }
        
        # Overall statistics
        summary['overall_stats'] = {
            'total_molecules': len(classified_df),
            'valid_classifications': total_valid,
            'invalid_classifications': len(classified_df) - total_valid,
            'classification_success_rate': total_valid / len(classified_df) * 100 if len(classified_df) > 0 else 0
        }
        
        # Quality metrics
        if 'volatility_score' in classified_df.columns:
            volatility_by_note = {}
            for note_position in ['top', 'middle', 'base']:
                position_molecules = classified_df[classified_df['note_position'] == note_position]
                volatility_values = position_molecules['volatility_score'].dropna()
                if len(volatility_values) > 0:
                    volatility_by_note[note_position] = {
                        'mean_volatility': float(volatility_values.mean()),
                        'expected_range': self._get_expected_volatility_range(note_position)
                    }
            summary['volatility_analysis'] = volatility_by_note
        
        return summary
    
    def _get_expected_volatility_range(self, note_position: str) -> Dict[str, float]:
        """Get expected volatility range for note position"""
        ranges = {
            'top': {'min': 0.6, 'max': 1.0},     # High volatility
            'middle': {'min': 0.3, 'max': 0.7},  # Medium volatility
            'base': {'min': 0.0, 'max': 0.4}     # Low volatility
        }
        return ranges.get(note_position, {'min': 0.0, 'max': 1.0})
    
    def validate_classification(self, classified_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate classification results and identify potential issues
        
        Args:
            classified_df: DataFrame with classified molecules
            
        Returns:
            Dict with validation results and warnings
        """
        warnings = {
            'molecular_weight_outliers': [],
            'volatility_mismatches': [],
            'descriptor_conflicts': [],
            'general_issues': []
        }
        
        for _, row in classified_df.iterrows():
            smiles = row['smiles']
            note_position = row['note_position']
            mw = row.get('molecular_weight')
            volatility = row.get('volatility_score')
            
            if note_position is None:
                continue
            
            # Check for molecular weight outliers
            if mw is not None:
                if note_position == 'top' and mw > 200:
                    warnings['molecular_weight_outliers'].append(
                        f"{smiles}: Heavy molecule ({mw:.1f}) classified as top note"
                    )
                elif note_position == 'base' and mw < 150:
                    warnings['molecular_weight_outliers'].append(
                        f"{smiles}: Light molecule ({mw:.1f}) classified as base note"
                    )
            
            # Check for volatility mismatches
            if volatility is not None:
                expected_range = self._get_expected_volatility_range(note_position)
                if volatility < expected_range['min'] or volatility > expected_range['max']:
                    warnings['volatility_mismatches'].append(
                        f"{smiles}: Volatility {volatility:.2f} doesn't match {note_position} note expectations"
                    )
            
            # Check for descriptor conflicts
            if 'matched_descriptors' in row and row['matched_descriptors']:
                descriptors = row['matched_descriptors']
                conflicts = self._check_descriptor_conflicts(descriptors, note_position)
                if conflicts:
                    warnings['descriptor_conflicts'].extend(conflicts)
        
        # Check overall distribution
        note_counts = classified_df['note_position'].value_counts()
        total = len(classified_df[classified_df['note_position'].notna()])
        
        if total > 0:
            top_percentage = note_counts.get('top', 0) / total * 100
            base_percentage = note_counts.get('base', 0) / total * 100
            
            if top_percentage < 10:
                warnings['general_issues'].append("Very few top notes - fragrance may lack initial impact")
            elif top_percentage > 70:
                warnings['general_issues'].append("Too many top notes - fragrance may lack depth")
            
            if base_percentage < 5:
                warnings['general_issues'].append("Very few base notes - fragrance may lack longevity")
            elif base_percentage > 60:
                warnings['general_issues'].append("Too many base notes - fragrance may be overly heavy")
        
        return warnings
    
    def _check_descriptor_conflicts(self, descriptors: List[str], note_position: str) -> List[str]:
        """Check for conflicts between descriptors and note position"""
        conflicts = []
        
        # Strong top note descriptors
        top_descriptors = {'citrus', 'fresh', 'aldehydic', 'mint', 'bergamot', 'lemon'}
        # Strong base note descriptors  
        base_descriptors = {'musk', 'amber', 'vanilla', 'sandalwood', 'woody', 'leather'}
        
        descriptor_set = set(descriptors)
        
        if note_position == 'base' and descriptor_set & top_descriptors:
            conflicts.append(f"Base note has top note descriptors: {descriptor_set & top_descriptors}")
        
        if note_position == 'top' and descriptor_set & base_descriptors:
            conflicts.append(f"Top note has base note descriptors: {descriptor_set & base_descriptors}")
        
        return conflicts
    
    def suggest_reclassification(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Suggest reclassification for molecules that may be misclassified
        
        Args:
            classified_df: DataFrame with classified molecules
            
        Returns:
            DataFrame with suggested reclassifications
        """
        suggestions = []
        
        for _, row in classified_df.iterrows():
            smiles = row['smiles']
            current_position = row['note_position']
            descriptors = row.get('matched_descriptors', [])
            
            # Check for descriptor-based reclassification
            suggested_position = self.molecular_utils.suggest_note_position_override(smiles, descriptors)
            
            if suggested_position and suggested_position != current_position:
                suggestions.append({
                    'smiles': smiles,
                    'current_position': current_position,
                    'suggested_position': suggested_position,
                    'reason': f"Descriptors {descriptors} strongly indicate {suggested_position} note",
                    'confidence': 'high' if len(set(descriptors) & {'citrus', 'fresh', 'musk', 'amber'}) > 0 else 'medium'
                })
        
        return pd.DataFrame(suggestions)