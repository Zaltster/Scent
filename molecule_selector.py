# molecule_selector.py
# Purpose: Finds molecules that match the selected descriptors
# What it does: Searches your dataset for molecules with matching descriptor flags
# Responsibilities: Query dataset, apply weights, handle overlapping molecules
# Input: ["fresh", "clean", "powdery"] with weights [0.9, 0.8, 0.7]
# Output: List of candidate molecules with relevance scores

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from data_loader import DataLoader
from molecular_utils import MolecularUtils
from config import MOLECULE_SELECTION, DESCRIPTOR_COLUMNS

logger = logging.getLogger(__name__)

class MoleculeSelector:
    """
    Selects molecules from dataset based on scent descriptor requirements
    """
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize MoleculeSelector
        
        Args:
            data_loader: Initialized DataLoader instance
        """
        self.data_loader = data_loader
        self.df = data_loader.df
        self.molecular_utils = MolecularUtils()
        
    def select_molecules(self, descriptor_scores: Dict[str, float]) -> pd.DataFrame:
        """
        Main method: Select molecules based on descriptor requirements
        
        Args:
            descriptor_scores: Dict mapping descriptor names to similarity scores
            
        Returns:
            DataFrame with selected molecules and their relevance scores
        """
        logger.info(f"Selecting molecules for {len(descriptor_scores)} descriptors")
        
        # Get candidate molecules for each descriptor
        candidates = self._get_candidate_molecules(descriptor_scores)
        
        # Calculate relevance scores
        scored_molecules = self._calculate_relevance_scores(candidates, descriptor_scores)
        
        # Apply diversity and quality filters
        filtered_molecules = self._apply_selection_filters(scored_molecules)
        
        logger.info(f"Selected {len(filtered_molecules)} candidate molecules")
        
        return filtered_molecules
    
    def _get_candidate_molecules(self, descriptor_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Get all molecules that match each descriptor
        
        Args:
            descriptor_scores: Descriptor to score mapping
            
        Returns:
            Dict mapping descriptor to list of matching SMILES
        """
        candidates = {}
        
        for descriptor, score in descriptor_scores.items():
            if descriptor in DESCRIPTOR_COLUMNS:
                molecules = self.data_loader.get_molecules_by_descriptor(descriptor)
                candidates[descriptor] = molecules
                logger.debug(f"Found {len(molecules)} molecules for '{descriptor}'")
            else:
                logger.warning(f"Unknown descriptor: {descriptor}")
                candidates[descriptor] = []
        
        return candidates
    
    def _calculate_relevance_scores(self, candidates: Dict[str, List[str]], 
                                  descriptor_scores: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate relevance score for each molecule based on descriptor matches
        
        Args:
            candidates: Dict mapping descriptor to molecule lists
            descriptor_scores: Descriptor similarity scores
            
        Returns:
            DataFrame with molecules and their scores
        """
        # Collect all unique molecules
        all_molecules = set()
        for molecules in candidates.values():
            all_molecules.update(molecules)
        
        molecule_data = []
        
        for smiles in all_molecules:
            # Calculate base relevance score
            relevance_score = 0
            matched_descriptors = []
            descriptor_count = 0
            
            for descriptor, molecules in candidates.items():
                if smiles in molecules:
                    # Weight by descriptor similarity score
                    relevance_score += descriptor_scores[descriptor]
                    matched_descriptors.append(descriptor)
                    descriptor_count += 1
            
            # Apply overlap bonus for molecules matching multiple descriptors
            if descriptor_count > 1:
                overlap_bonus = MOLECULE_SELECTION['overlap_bonus'] * (descriptor_count - 1)
                relevance_score += overlap_bonus
            
            # Get molecule properties
            properties = self.molecular_utils.calculate_molecular_properties(smiles)
            
            molecule_data.append({
                'smiles': smiles,
                'relevance_score': relevance_score,
                'descriptor_count': descriptor_count,
                'matched_descriptors': matched_descriptors,
                'molecular_weight': properties['molecular_weight'] if properties else None,
                'note_position': properties['note_position'] if properties else None,
                'logp': properties['logp'] if properties else None,
                'valid_structure': properties is not None
            })
        
        df = pd.DataFrame(molecule_data)
        
        # Remove invalid structures
        df = df[df['valid_structure'] == True]
        
        # Sort by relevance score
        df = df.sort_values('relevance_score', ascending=False)
        
        return df
    
    def _apply_selection_filters(self, scored_molecules: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filters to improve molecule selection quality
        
        Args:
            scored_molecules: DataFrame with scored molecules
            
        Returns:
            Filtered DataFrame
        """
        if len(scored_molecules) == 0:
            return scored_molecules
        
        # Filter 1: Remove molecules with very low scores
        min_score_threshold = scored_molecules['relevance_score'].quantile(0.3)
        filtered = scored_molecules[scored_molecules['relevance_score'] >= min_score_threshold]
        
        # Filter 2: Apply diversity to avoid too many similar molecules
        if MOLECULE_SELECTION['diversity_weight'] > 0:
            filtered = self._apply_diversity_filter(filtered)
        
        # Filter 3: Ensure reasonable molecular weight distribution
        filtered = self._filter_by_molecular_properties(filtered)
        
        return filtered.reset_index(drop=True)
    
    def _apply_diversity_filter(self, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply structural diversity filter to avoid too many similar molecules
        
        Args:
            molecules_df: DataFrame with molecules
            
        Returns:
            Filtered DataFrame with improved diversity
        """
        if len(molecules_df) <= 10:
            return molecules_df  # Keep all if we don't have many
        
        # Simple diversity filter: remove molecules that are too similar
        # This is a placeholder - in production you'd use molecular fingerprints
        
        # Group by molecular weight ranges to ensure diversity
        molecules_df['mw_range'] = pd.cut(
            molecules_df['molecular_weight'], 
            bins=5, 
            labels=['very_light', 'light', 'medium', 'heavy', 'very_heavy']
        )
        
        # Sample from each weight range
        diverse_molecules = []
        for mw_range in molecules_df['mw_range'].unique():
            if pd.isna(mw_range):
                continue
            
            range_molecules = molecules_df[molecules_df['mw_range'] == mw_range]
            # Take top molecules from each range, but limit quantity
            max_per_range = max(1, len(range_molecules) // 3)
            diverse_molecules.append(range_molecules.head(max_per_range))
        
        if diverse_molecules:
            result = pd.concat(diverse_molecules).drop_duplicates('smiles')
            result = result.sort_values('relevance_score', ascending=False)
        else:
            result = molecules_df
        
        return result.drop('mw_range', axis=1)
    
    def _filter_by_molecular_properties(self, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter molecules based on molecular properties suitable for fragrance
        
        Args:
            molecules_df: DataFrame with molecules
            
        Returns:
            Filtered DataFrame
        """
        # Remove molecules that are too large or too small for fragrance use
        filtered = molecules_df[
            (molecules_df['molecular_weight'] >= 50) & 
            (molecules_df['molecular_weight'] <= 500)
        ]
        
        # Remove molecules with extreme LogP values (too hydrophilic or lipophilic)
        if 'logp' in filtered.columns:
            filtered = filtered[
                (filtered['logp'] >= -3) & 
                (filtered['logp'] <= 8)
            ]
        
        return filtered
    
    def get_molecules_by_note_position(self, molecules_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group selected molecules by note position (top/middle/base)
        
        Args:
            molecules_df: DataFrame with selected molecules
            
        Returns:
            Dict mapping note position to DataFrame of molecules
        """
        grouped = {}
        
        for note_position in ['top', 'middle', 'base']:
            position_molecules = molecules_df[molecules_df['note_position'] == note_position]
            grouped[note_position] = position_molecules.sort_values('relevance_score', ascending=False)
        
        return grouped
    
    def get_selection_summary(self, molecules_df: pd.DataFrame, 
                            descriptor_scores: Dict[str, float]) -> Dict:
        """
        Generate summary of molecule selection process
        
        Args:
            molecules_df: Selected molecules DataFrame
            descriptor_scores: Original descriptor scores
            
        Returns:
            Dict with selection summary
        """
        if len(molecules_df) == 0:
            return {
                'total_selected': 0,
                'note_distribution': {'top': 0, 'middle': 0, 'base': 0},
                'descriptor_coverage': {},
                'issues': ['No molecules selected - descriptors may be too specific']
            }
        
        # Note position distribution
        note_counts = molecules_df['note_position'].value_counts().to_dict()
        note_distribution = {
            'top': note_counts.get('top', 0),
            'middle': note_counts.get('middle', 0),
            'base': note_counts.get('base', 0)
        }
        
        # Descriptor coverage
        descriptor_coverage = {}
        for descriptor in descriptor_scores.keys():
            coverage_count = sum(
                1 for matched_list in molecules_df['matched_descriptors']
                if descriptor in matched_list
            )
            descriptor_coverage[descriptor] = {
                'molecules_found': coverage_count,
                'coverage_percentage': coverage_count / len(molecules_df) * 100 if len(molecules_df) > 0 else 0
            }
        
        # Quality metrics
        avg_relevance = molecules_df['relevance_score'].mean()
        avg_descriptors_per_molecule = molecules_df['descriptor_count'].mean()
        
        # Identify potential issues
        issues = []
        if note_distribution['top'] == 0:
            issues.append("No top notes found - fragrance may lack initial impact")
        if note_distribution['base'] == 0:
            issues.append("No base notes found - fragrance may lack longevity")
        if avg_descriptors_per_molecule < 1.2:
            issues.append("Low descriptor overlap - molecules may not fully represent concept")
        if avg_relevance < 0.5:
            issues.append("Low relevance scores - consider broadening descriptor selection")
        
        return {
            'total_selected': len(molecules_df),
            'note_distribution': note_distribution,
            'descriptor_coverage': descriptor_coverage,
            'quality_metrics': {
                'avg_relevance_score': avg_relevance,
                'avg_descriptors_per_molecule': avg_descriptors_per_molecule,
                'score_range': {
                    'min': molecules_df['relevance_score'].min(),
                    'max': molecules_df['relevance_score'].max()
                }
            },
            'molecular_weight_stats': {
                'mean': molecules_df['molecular_weight'].mean(),
                'min': molecules_df['molecular_weight'].min(),
                'max': molecules_df['molecular_weight'].max()
            },
            'issues': issues
        }
    
    def suggest_improvements(self, molecules_df: pd.DataFrame, 
                           descriptor_scores: Dict[str, float]) -> List[str]:
        """
        Suggest improvements to the molecule selection
        
        Args:
            molecules_df: Selected molecules DataFrame
            descriptor_scores: Original descriptor scores
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        summary = self.get_selection_summary(molecules_df, descriptor_scores)
        
        # Address note balance issues
        note_dist = summary['note_distribution']
        total_molecules = summary['total_selected']
        
        if total_molecules > 0:
            top_percentage = note_dist['top'] / total_molecules * 100
            base_percentage = note_dist['base'] / total_molecules * 100
            
            if top_percentage < 20:
                suggestions.append("Consider adding more fresh/citrus descriptors for better top notes")
            if base_percentage < 15:
                suggestions.append("Consider adding woody/musky descriptors for better base notes")
            if top_percentage > 60:
                suggestions.append("Too many light molecules - add some heavier descriptors for balance")
        
        # Address descriptor coverage issues
        uncovered_descriptors = [
            desc for desc, info in summary['descriptor_coverage'].items()
            if info['molecules_found'] == 0
        ]
        
        if uncovered_descriptors:
            suggestions.append(f"No molecules found for: {', '.join(uncovered_descriptors[:3])}")
        
        # Address quality issues
        if summary['quality_metrics']['avg_relevance_score'] < 0.4:
            suggestions.append("Low relevance scores - try more specific or common descriptors")
        
        if total_molecules < 5:
            suggestions.append("Very few molecules found - consider lowering similarity threshold")
        elif total_molecules > 50:
            suggestions.append("Too many molecules found - consider raising similarity threshold")
        
        return suggestions
    
    def find_alternative_molecules(self, descriptor_scores: Dict[str, float], 
                                 excluded_smiles: List[str] = None) -> pd.DataFrame:
        """
        Find alternative molecules for the same descriptors
        
        Args:
            descriptor_scores: Descriptor requirements
            excluded_smiles: SMILES to exclude from results
            
        Returns:
            DataFrame with alternative molecules
        """
        excluded_smiles = excluded_smiles or []
        
        # Get all candidates
        all_candidates = self.select_molecules(descriptor_scores)
        
        # Filter out excluded molecules
        alternatives = all_candidates[~all_candidates['smiles'].isin(excluded_smiles)]
        
        return alternatives
    
    def get_molecule_details(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Get detailed information about specific molecules
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            DataFrame with detailed molecule information
        """
        details = []
        
        for smiles in smiles_list:
            # Get basic info from dataset
            molecule_row = self.df[self.df[self.data_loader.dataset_path.stem == 'nonStereoSMILES'] == smiles]
            
            if len(molecule_row) > 0:
                row = molecule_row.iloc[0]
                
                # Get all descriptors for this molecule
                active_descriptors = []
                for desc in DESCRIPTOR_COLUMNS:
                    if row[desc] == 1:
                        active_descriptors.append(desc)
                
                # Get molecular properties
                properties = self.molecular_utils.calculate_molecular_properties(smiles)
                
                # Get functional groups
                functional_groups = self.molecular_utils.get_functional_groups(smiles)
                
                details.append({
                    'smiles': smiles,
                    'descriptors': active_descriptors,
                    'num_descriptors': len(active_descriptors),
                    'molecular_weight': properties['molecular_weight'] if properties else None,
                    'note_position': properties['note_position'] if properties else None,
                    'logp': properties['logp'] if properties else None,
                    'functional_groups': functional_groups,
                    'volatility_estimate': self.molecular_utils.estimate_volatility(smiles)
                })
            else:
                details.append({
                    'smiles': smiles,
                    'descriptors': [],
                    'num_descriptors': 0,
                    'molecular_weight': None,
                    'note_position': None,
                    'logp': None,
                    'functional_groups': {},
                    'volatility_estimate': None
                })
        
        return pd.DataFrame(details)
    
    def optimize_selection(self, descriptor_scores: Dict[str, float], 
                         target_note_distribution: Dict[str, int] = None) -> pd.DataFrame:
        """
        Optimize molecule selection to meet specific note distribution targets
        
        Args:
            descriptor_scores: Descriptor requirements
            target_note_distribution: Desired number of molecules per note (e.g., {'top': 3, 'middle': 4, 'base': 2})
            
        Returns:
            Optimized DataFrame with molecules
        """
        if target_note_distribution is None:
            target_note_distribution = {'top': 3, 'middle': 4, 'base': 2}
        
        # Get all candidates
        all_candidates = self.select_molecules(descriptor_scores)
        
        if len(all_candidates) == 0:
            return all_candidates
        
        # Group by note position
        grouped = self.get_molecules_by_note_position(all_candidates)
        
        # Select molecules for each note position
        optimized_molecules = []
        
        for note_position, target_count in target_note_distribution.items():
            if note_position in grouped and len(grouped[note_position]) > 0:
                # Take top molecules for this note position
                selected = grouped[note_position].head(target_count)
                optimized_molecules.append(selected)
        
        if optimized_molecules:
            result = pd.concat(optimized_molecules)
            result = result.sort_values('relevance_score', ascending=False)
            return result.reset_index(drop=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no molecules found