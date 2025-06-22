# formula_composer.py
# Purpose: Assembles final fragrance formula with proportions
# What it does: Selects best molecules from each note layer, assigns percentages
# Responsibilities: Pick top candidates per layer, calculate proportions, create final formula
# Input: Classified molecules with scores
# Output: Final formula like "30% limonene + 25% linalool + 20% white musk + ..."

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from molecular_utils import MolecularUtils
from config import PROPORTION_PROFILES, MOLECULE_SELECTION, FORMULA_SETTINGS

logger = logging.getLogger(__name__)

class FormulaComposer:
    """
    Composes final fragrance formulas from classified molecules
    """
    
    def __init__(self):
        """Initialize FormulaComposer"""
        self.molecular_utils = MolecularUtils()
        
    def compose_formula(self, classified_molecules: pd.DataFrame, 
                       fragrance_type: str = 'balanced',
                       custom_proportions: Optional[Dict[str, float]] = None) -> Dict:
        """
        Main method: Compose complete fragrance formula
        
        Args:
            classified_molecules: DataFrame with classified molecules
            fragrance_type: Type of fragrance ('balanced', 'fresh', 'oriental', 'floral')
            custom_proportions: Custom proportion ratios (overrides fragrance_type)
            
        Returns:
            Dict with complete formula information
        """
        logger.info(f"Composing {fragrance_type} fragrance formula from {len(classified_molecules)} molecules")
        
        if len(classified_molecules) == 0:
            return self._create_empty_formula(fragrance_type)
        
        # Get proportion profile
        proportions = custom_proportions or PROPORTION_PROFILES.get(fragrance_type, PROPORTION_PROFILES['balanced'])
        
        # Group molecules by note position
        molecule_groups = self._group_by_note_position(classified_molecules)
        
        # Select final molecules for each note position
        selected_molecules = self._select_final_molecules(molecule_groups)
        
        # Calculate individual percentages
        formula_components = self._calculate_individual_percentages(selected_molecules, proportions)
        
        # Normalize to 100%
        normalized_components = self._normalize_percentages(formula_components)
        
        # Generate metadata
        metadata = self._generate_metadata(normalized_components, classified_molecules)
        
        # Compile final formula
        formula = {
            'components': normalized_components,
            'metadata': metadata,
            'proportions_used': proportions,
            'fragrance_type': fragrance_type,
            'total_percentage': sum(comp['percentage'] for comp in normalized_components),
            'num_components': len(normalized_components)
        }
        
        logger.info(f"Formula composed: {len(normalized_components)} components, {formula['total_percentage']:.1f}% total")
        return formula
    
    def _create_empty_formula(self, fragrance_type: str) -> Dict:
        """Create empty formula structure when no molecules available"""
        return {
            'components': [],
            'metadata': {
                'note_distribution': {'counts': {'top': 0, 'middle': 0, 'base': 0}, 'percentages': {'top': 0, 'middle': 0, 'base': 0}},
                'molecular_weight_range': {},
                'descriptor_coverage': {'total_unique_descriptors': 0, 'descriptor_frequencies': {}, 'most_common_descriptors': []},
                'quality_metrics': {'avg_relevance_score': 0, 'avg_descriptors_per_molecule': 0, 'formula_diversity': 0},
                'warnings': ['No molecules available for formula composition']
            },
            'proportions_used': PROPORTION_PROFILES.get(fragrance_type, PROPORTION_PROFILES['balanced']),
            'fragrance_type': fragrance_type,
            'total_percentage': 0,
            'num_components': 0
        }
    
    def _group_by_note_position(self, molecules_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group molecules by note position and sort by relevance"""
        groups = {}
        
        for note_position in ['top', 'middle', 'base']:
            note_molecules = molecules_df[molecules_df['note_position'] == note_position].copy()
            
            if len(note_molecules) > 0:
                # Sort by relevance score (highest first)
                if 'relevance_score' in note_molecules.columns:
                    note_molecules = note_molecules.sort_values('relevance_score', ascending=False)
                
                groups[note_position] = note_molecules
                logger.debug(f"{note_position}: {len(note_molecules)} molecules available")
            else:
                groups[note_position] = pd.DataFrame()
                logger.warning(f"No {note_position} note molecules available")
        
        return groups
    
    def _select_final_molecules(self, molecule_groups: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Select final molecules for each note position"""
        selected = {}
        
        for note_position, molecules in molecule_groups.items():
            if len(molecules) == 0:
                selected[note_position] = pd.DataFrame()
                continue
            
            # Determine selection count
            max_molecules = MOLECULE_SELECTION['max_molecules_per_note']
            min_molecules = MOLECULE_SELECTION['min_molecules_per_note']
            
            available_count = len(molecules)
            target_count = max(min_molecules, min(max_molecules, available_count))
            
            # If we have fewer than minimum, take what we have
            if available_count < min_molecules:
                target_count = available_count
                logger.warning(f"Only {available_count} molecules for {note_position} (need {min_molecules})")
            
            # Select top molecules
            selected_molecules = molecules.head(target_count).copy()
            selected_molecules['selection_rank'] = range(1, len(selected_molecules) + 1)
            
            selected[note_position] = selected_molecules
            logger.debug(f"Selected {len(selected_molecules)} molecules for {note_position}")
        
        return selected
    
    def _calculate_individual_percentages(self, selected_molecules: Dict[str, pd.DataFrame], 
                                        proportions: Dict[str, float]) -> List[Dict]:
        """Calculate percentage for each individual molecule"""
        components = []
        
        for note_position, target_proportion in proportions.items():
            molecules = selected_molecules.get(note_position, pd.DataFrame())
            
            if len(molecules) == 0:
                logger.warning(f"No molecules for {note_position} notes")
                continue
            
            # Calculate how to distribute the proportion among molecules
            individual_percentages = self._distribute_proportion(molecules, target_proportion)
            
            # Create component entries
            for _, row in molecules.iterrows():
                smiles = row['smiles']
                percentage = individual_percentages.get(smiles, 0)
                
                component = {
                    'smiles': smiles,
                    'note_position': note_position,
                    'percentage': percentage,
                    'relevance_score': row.get('relevance_score', 0),
                    'molecular_weight': row.get('molecular_weight'),
                    'matched_descriptors': row.get('matched_descriptors', []),
                    'selection_rank': row.get('selection_rank', 0)
                }
                
                components.append(component)
        
        return components
    
    def _distribute_proportion(self, molecules: pd.DataFrame, total_proportion: float) -> Dict[str, float]:
        """Distribute proportion percentage among molecules in a note position"""
        if len(molecules) == 0:
            return {}
        
        # Convert proportion (0-1) to percentage (0-100)
        total_percentage = total_proportion * 100
        
        # Use relevance scores for weighted distribution if available
        if 'relevance_score' in molecules.columns and molecules['relevance_score'].sum() > 0:
            total_relevance = molecules['relevance_score'].sum()
            percentages = {}
            
            for _, row in molecules.iterrows():
                smiles = row['smiles']
                weight = row['relevance_score'] / total_relevance
                percentages[smiles] = total_percentage * weight
        else:
            # Equal distribution if no relevance scores
            percentage_per_molecule = total_percentage / len(molecules)
            percentages = {row['smiles']: percentage_per_molecule for _, row in molecules.iterrows()}
        
        return percentages
    
    def _normalize_percentages(self, components: List[Dict]) -> List[Dict]:
        """Normalize percentages to sum to 100%"""
        if not components:
            return components
        
        # Calculate current total
        current_total = sum(comp['percentage'] for comp in components)
        
        if current_total == 0:
            logger.error("Total percentage is zero - cannot normalize")
            return components
        
        # Scale to 100%
        scale_factor = 100.0 / current_total
        
        for component in components:
            component['percentage'] *= scale_factor
        
        # Round to specified precision
        precision = FORMULA_SETTINGS.get('precision', 1)
        for component in components:
            component['percentage'] = round(component['percentage'], precision)
        
        # Remove components with zero percentage after rounding
        components = [comp for comp in components if comp['percentage'] > 0]
        
        # Sort by percentage (highest first)
        components.sort(key=lambda x: x['percentage'], reverse=True)
        
        logger.info(f"Normalized formula: {len(components)} components totaling {sum(c['percentage'] for c in components):.1f}%")
        
        return components
    
    def _generate_metadata(self, components: List[Dict], all_molecules: pd.DataFrame) -> Dict:
        """Generate comprehensive metadata about the formula"""
        if not components:
            return {
                'note_distribution': {'counts': {'top': 0, 'middle': 0, 'base': 0}, 'percentages': {'top': 0, 'middle': 0, 'base': 0}},
                'molecular_weight_range': {},
                'descriptor_coverage': {'total_unique_descriptors': 0, 'descriptor_frequencies': {}, 'most_common_descriptors': []},
                'quality_metrics': {'avg_relevance_score': 0, 'avg_descriptors_per_molecule': 0, 'formula_diversity': 0},
                'warnings': ['No components in formula']
            }
        
        # Note distribution analysis
        note_counts = {'top': 0, 'middle': 0, 'base': 0}
        note_percentages = {'top': 0, 'middle': 0, 'base': 0}
        
        for component in components:
            note = component['note_position']
            if note in note_counts:
                note_counts[note] += 1
                note_percentages[note] += component['percentage']
        
        # Molecular weight analysis
        mw_values = [c['molecular_weight'] for c in components if c['molecular_weight'] is not None]
        mw_range = {}
        if mw_values:
            mw_range = {
                'min': min(mw_values),
                'max': max(mw_values),
                'mean': sum(mw_values) / len(mw_values),
                'range': max(mw_values) - min(mw_values)
            }
        
        # Descriptor coverage analysis
        all_descriptors = []
        descriptor_counts = {}
        
        for component in components:
            descriptors = component.get('matched_descriptors', [])
            all_descriptors.extend(descriptors)
            for desc in descriptors:
                descriptor_counts[desc] = descriptor_counts.get(desc, 0) + 1
        
        unique_descriptors = set(all_descriptors)
        most_common = sorted(descriptor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Quality metrics
        avg_relevance = sum(c['relevance_score'] for c in components) / len(components) if components else 0
        avg_descriptors = len(all_descriptors) / len(components) if components else 0
        diversity = len(set(c['smiles'] for c in components))
        
        # Generate warnings
        warnings = self._generate_warnings(components, note_percentages)
        
        return {
            'note_distribution': {
                'counts': note_counts,
                'percentages': note_percentages
            },
            'molecular_weight_range': mw_range,
            'descriptor_coverage': {
                'total_unique_descriptors': len(unique_descriptors),
                'descriptor_frequencies': descriptor_counts,
                'most_common_descriptors': most_common
            },
            'quality_metrics': {
                'avg_relevance_score': avg_relevance,
                'avg_descriptors_per_molecule': avg_descriptors,
                'formula_diversity': diversity
            },
            'warnings': warnings
        }
    
    def _generate_warnings(self, components: List[Dict], note_percentages: Dict[str, float]) -> List[str]:
        """Generate warnings about potential formula issues"""
        warnings = []
        
        # Note balance warnings
        if note_percentages.get('top', 0) < 10:
            warnings.append("Very low top note percentage - fragrance may lack initial impact")
        elif note_percentages.get('top', 0) > 60:
            warnings.append("Very high top note percentage - fragrance may lack depth")
        
        if note_percentages.get('middle', 0) < 30:
            warnings.append("Low middle note percentage - fragrance may lack body")
        
        if note_percentages.get('base', 0) < 5:
            warnings.append("Very low base note percentage - fragrance may lack longevity")
        elif note_percentages.get('base', 0) > 50:
            warnings.append("Very high base note percentage - fragrance may be too heavy")
        
        # Complexity warnings
        if len(components) < 3:
            warnings.append("Very few components - consider adding more molecules for complexity")
        elif len(components) > 12:
            warnings.append("Many components - formula may be overly complex")
        
        # Individual percentage warnings
        for component in components:
            percentage = component['percentage']
            if percentage > 40:
                warnings.append(f"Very high individual percentage ({percentage:.1f}%) may dominate fragrance")
            elif percentage < 1:
                warnings.append(f"Very low individual percentage ({percentage:.1f}%) may be ineffective")
        
        # Chemical compatibility warnings
        smiles_list = [c['smiles'] for c in components]
        compatibility_issues = self.molecular_utils.check_chemical_compatibility(smiles_list)
        
        for category, issues in compatibility_issues.items():
            if issues:
                warnings.extend(issues[:2])  # Limit to 2 warnings per category
        
        return warnings
    
    def format_formula_display(self, formula: Dict) -> str:
        """Format formula for human-readable display"""
        if not formula['components']:
            return f"=== {formula['fragrance_type'].title()} Fragrance Formula ===\nNo components available"
        
        lines = []
        lines.append(f"=== {formula['fragrance_type'].title()} Fragrance Formula ===")
        lines.append(f"Total: {formula['total_percentage']:.1f}% | Components: {formula['num_components']}")
        lines.append("")
        
        # Group components by note position for display
        for note_position in ['top', 'middle', 'base']:
            note_components = [c for c in formula['components'] if c['note_position'] == note_position]
            
            if note_components:
                note_total = sum(c['percentage'] for c in note_components)
                lines.append(f"{note_position.upper()} NOTES ({note_total:.1f}%):")
                
                for component in sorted(note_components, key=lambda x: x['percentage'], reverse=True):
                    mw = component.get('molecular_weight', 0)
                    descriptors = ', '.join(component.get('matched_descriptors', [])[:3])
                    if len(component.get('matched_descriptors', [])) > 3:
                        descriptors += "..."
                    
                    lines.append(f"  • {component['percentage']:5.1f}% - {component['smiles']} (MW: {mw:.1f}) [{descriptors}]")
                
                lines.append("")
        
        # Add summary
        metadata = formula['metadata']
        lines.append("SUMMARY:")
        lines.append(f"  • Unique descriptors: {metadata['descriptor_coverage']['total_unique_descriptors']}")
        lines.append(f"  • Average relevance: {metadata['quality_metrics']['avg_relevance_score']:.2f}")
        
        mw_range = metadata['molecular_weight_range']
        if mw_range:
            lines.append(f"  • MW range: {mw_range.get('min', 0):.1f} - {mw_range.get('max', 0):.1f}")
        
        # Add warnings if any
        warnings = metadata.get('warnings', [])
        if warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in warnings[:3]:  # Show max 3 warnings
                lines.append(f"  ⚠ {warning}")
        
        return "\n".join(lines)
    
    def create_alternative_formulas(self, classified_molecules: pd.DataFrame, 
                                  num_alternatives: int = 3) -> List[Dict]:
        """Create multiple alternative formulas with different characteristics"""
        alternatives = []
        
        if len(classified_molecules) == 0:
            return alternatives
        
        # Different approaches for alternatives
        approaches = [
            ('fresh', 'Fresh-focused formula'),
            ('oriental', 'Oriental-focused formula'),
            ('balanced', 'Balanced formula with different selection')
        ]
        
        for i in range(min(num_alternatives, len(approaches))):
            fragrance_type, description = approaches[i]
            
            try:
                # Modify selection for variety
                modified_molecules = self._create_alternative_selection(classified_molecules, i)
                
                # Compose alternative formula
                formula = self.compose_formula(modified_molecules, fragrance_type)
                formula['alternative_id'] = i + 1
                formula['description'] = description
                
                alternatives.append(formula)
                
            except Exception as e:
                logger.warning(f"Failed to create alternative formula {i+1}: {e}")
                continue
        
        return alternatives
    
    def _create_alternative_selection(self, molecules_df: pd.DataFrame, variation_index: int) -> pd.DataFrame:
        """Create alternative molecule selection for variety"""
        if len(molecules_df) <= 3:
            return molecules_df  # Too few molecules to create alternatives
        
        # Different selection strategies
        if variation_index == 0:
            # Favor molecules with more descriptors
            if 'descriptor_count' in molecules_df.columns:
                return molecules_df.nlargest(len(molecules_df) // 2, 'descriptor_count')
            else:
                return molecules_df.sample(n=max(3, len(molecules_df) // 2))
        
        elif variation_index == 1:
            # Favor extreme molecular weights
            if 'molecular_weight' in molecules_df.columns:
                light = molecules_df[molecules_df['molecular_weight'] < 150]
                heavy = molecules_df[molecules_df['molecular_weight'] > 250]
                return pd.concat([light, heavy]).drop_duplicates()
            else:
                return molecules_df.sample(n=max(3, len(molecules_df) // 2))
        
        else:
            # Random selection with bias toward higher scores
            if 'relevance_score' in molecules_df.columns and molecules_df['relevance_score'].sum() > 0:
                sample_size = max(3, len(molecules_df) // 2)
                weights = molecules_df['relevance_score'] / molecules_df['relevance_score'].sum()
                sampled_indices = np.random.choice(
                    molecules_df.index, 
                    size=min(sample_size, len(molecules_df)), 
                    replace=False, 
                    p=weights
                )
                return molecules_df.loc[sampled_indices]
            else:
                return molecules_df.sample(n=max(3, len(molecules_df) // 2))