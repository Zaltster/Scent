# molecular_utils.py
# Purpose: Chemistry-specific utilities using RDKit
# What it does: SMILES parsing, molecular weight calculation, chemical validation
# Responsibilities: Convert SMILES to molecules, calculate properties, validate structures
# Input: SMILES strings
# Output: Molecular weights, chemical properties, validity checks

import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from config import MW_THRESHOLDS

logger = logging.getLogger(__name__)

class MolecularUtils:
    """
    Utilities for molecular property calculation and validation
    """
    
    @staticmethod
    def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit molecule object
        
        Args:
            smiles: SMILES string
            
        Returns:
            RDKit molecule object or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
            return mol
        except Exception as e:
            logger.error(f"Error parsing SMILES {smiles}: {e}")
            return None
    
    @staticmethod
    def calculate_molecular_weight(smiles: str) -> Optional[float]:
        """
        Calculate molecular weight from SMILES
        
        Args:
            smiles: SMILES string
            
        Returns:
            Molecular weight in g/mol or None if invalid
        """
        mol = MolecularUtils.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        try:
            return Descriptors.MolWt(mol)
        except Exception as e:
            logger.error(f"Error calculating MW for {smiles}: {e}")
            return None
    
    @staticmethod
    def classify_note_position(smiles: str) -> Optional[str]:
        """
        Classify molecule into fragrance note position based on molecular weight
        
        Args:
            smiles: SMILES string
            
        Returns:
            'top', 'middle', 'base', or None if invalid
        """
        mw = MolecularUtils.calculate_molecular_weight(smiles)
        if mw is None:
            return None
        
        if mw < MW_THRESHOLDS['top_max']:
            return 'top'
        elif mw < MW_THRESHOLDS['middle_max']:
            return 'middle'
        else:
            return 'base'
    
    @staticmethod
    def calculate_molecular_properties(smiles: str) -> Optional[Dict]:
        """
        Calculate comprehensive molecular properties
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dict with molecular properties or None if invalid
        """
        mol = MolecularUtils.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        try:
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                'num_rotatable_bonds': CalcNumRotatableBonds(mol),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'tpsa': Descriptors.TPSA(mol),  # Topological polar surface area
                'num_hbd': Descriptors.NumHDonors(mol),  # Hydrogen bond donors
                'num_hba': Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
            }
            
            # Add note position
            properties['note_position'] = MolecularUtils.classify_note_position(smiles)
            
            # Add Lipinski rule compliance
            properties['lipinski_compliant'] = (
                properties['molecular_weight'] <= 500 and
                properties['logp'] <= 5 and
                properties['num_hbd'] <= 5 and
                properties['num_hba'] <= 10
            )
            
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating properties for {smiles}: {e}")
            return None
    
    @staticmethod
    def validate_smiles(smiles: str) -> Tuple[bool, str]:
        """
        Validate SMILES string
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES syntax"
            
            # Additional checks
            if mol.GetNumAtoms() == 0:
                return False, "Empty molecule"
            
            if mol.GetNumHeavyAtoms() == 0:
                return False, "No heavy atoms"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    @staticmethod
    def batch_calculate_properties(smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate properties for multiple SMILES in batch
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            DataFrame with SMILES and their properties
        """
        results = []
        
        for smiles in smiles_list:
            properties = MolecularUtils.calculate_molecular_properties(smiles)
            if properties is not None:
                properties['smiles'] = smiles
                results.append(properties)
            else:
                # Add row with NaN values for invalid SMILES
                results.append({
                    'smiles': smiles,
                    'molecular_weight': None,
                    'note_position': None,
                    'logp': None,
                    'num_atoms': None,
                    'num_heavy_atoms': None,
                    'num_rotatable_bonds': None,
                    'num_rings': None,
                    'num_aromatic_rings': None,
                    'tpsa': None,
                    'num_hbd': None,
                    'num_hba': None,
                    'lipinski_compliant': None
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def get_functional_groups(smiles: str) -> Dict[str, int]:
        """
        Identify functional groups in molecule
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dict with functional group counts
        """
        mol = MolecularUtils.smiles_to_mol(smiles)
        if mol is None:
            return {}
        
        try:
            # Define functional group SMARTS patterns
            functional_groups = {
                'alcohol': '[OH]',
                'aldehyde': '[CX3H1](=O)[#6]',
                'ketone': '[#6][CX3](=O)[#6]',
                'ester': '[#6][CX3](=O)[OX2H0][#6]',
                'ether': '[OD2]([#6])[#6]',
                'carboxylic_acid': '[CX3](=O)[OX2H1]',
                'amine': '[NX3;H2,H1;!$(NC=O)]',
                'aromatic_ring': 'c1ccccc1',
                'phenol': '[OH][c]',
                'benzyl': '[CH2][c]',
                'methyl': '[CH3]',
                'ethyl': '[CH2][CH3]',
                'propyl': '[CH2][CH2][CH3]',
                'isopropyl': '[CH](C)C',
                'butyl': '[CH2][CH2][CH2][CH3]',
                'tert_butyl': '[C](C)(C)C',
                'vinyl': '[CH]=[CH2]',
                'allyl': '[CH2][CH]=[CH2]',
                'nitrile': '[C]#[N]',
                'nitro': '[N+](=O)[O-]',
                'sulfur': '[#16]',
                'halogen': '[F,Cl,Br,I]',
            }
            
            group_counts = {}
            for group_name, smarts in functional_groups.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    group_counts[group_name] = len(matches)
                else:
                    group_counts[group_name] = 0
            
            return group_counts
            
        except Exception as e:
            logger.error(f"Error identifying functional groups for {smiles}: {e}")
            return {}
    
    @staticmethod
    def estimate_volatility(smiles: str) -> Optional[float]:
        """
        Estimate relative volatility based on molecular properties
        Higher values = more volatile (better for top notes)
        
        Args:
            smiles: SMILES string
            
        Returns:
            Volatility score (0-1) or None if invalid
        """
        properties = MolecularUtils.calculate_molecular_properties(smiles)
        if properties is None:
            return None
        
        try:
            # Simple volatility estimation based on MW and LogP
            # Lower MW = more volatile
            # Lower LogP = more volatile (less lipophilic)
            
            mw = properties['molecular_weight']
            logp = properties['logp']
            
            # Normalize MW (typical range 50-500 for fragrance molecules)
            mw_score = max(0, min(1, (500 - mw) / 450))
            
            # Normalize LogP (typical range -2 to 8)
            logp_score = max(0, min(1, (8 - logp) / 10))
            
            # Combine scores (MW is more important for volatility)
            volatility = 0.7 * mw_score + 0.3 * logp_score
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error estimating volatility for {smiles}: {e}")
            return None
    
    @staticmethod
    def check_chemical_compatibility(smiles_list: List[str]) -> Dict[str, List[str]]:
        """
        Check for potential chemical incompatibilities in a formula
        
        Args:
            smiles_list: List of SMILES in the formula
            
        Returns:
            Dict with compatibility warnings
        """
        warnings = {
            'reactive_combinations': [],
            'ph_issues': [],
            'oxidation_risks': [],
            'general_warnings': []
        }
        
        # Get functional groups for all molecules
        all_groups = []
        for smiles in smiles_list:
            groups = MolecularUtils.get_functional_groups(smiles)
            all_groups.append((smiles, groups))
        
        # Check for problematic combinations
        for i, (smiles1, groups1) in enumerate(all_groups):
            for j, (smiles2, groups2) in enumerate(all_groups[i+1:], i+1):
                
                # Aldehydes + amines can form imines
                if groups1.get('aldehyde', 0) > 0 and groups2.get('amine', 0) > 0:
                    warnings['reactive_combinations'].append(
                        f"Aldehyde ({smiles1}) + Amine ({smiles2}) may react"
                    )
                
                # Acids + bases
                if groups1.get('carboxylic_acid', 0) > 0 and groups2.get('amine', 0) > 0:
                    warnings['ph_issues'].append(
                        f"Acid ({smiles1}) + Base ({smiles2}) may neutralize"
                    )
                
                # Phenols are sensitive to oxidation
                if groups1.get('phenol', 0) > 0 or groups2.get('phenol', 0) > 0:
                    warnings['oxidation_risks'].append(
                        "Phenolic compounds present - protect from oxidation"
                    )
        
        # Remove duplicates
        for category in warnings:
            warnings[category] = list(set(warnings[category]))
        
        return warnings
    
    @staticmethod
    def get_molecular_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[list]:
        """
        Generate molecular fingerprint for similarity calculations
        
        Args:
            smiles: SMILES string
            radius: Fingerprint radius
            n_bits: Number of bits in fingerprint
            
        Returns:
            Fingerprint as list of integers or None if invalid
        """
        mol = MolecularUtils.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        try:
            from rdkit.Chem import rdMolDescriptors
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=n_bits
            )
            return list(fingerprint)
        except Exception as e:
            logger.error(f"Error generating fingerprint for {smiles}: {e}")
            return None
    
    @staticmethod
    def calculate_similarity(smiles1: str, smiles2: str) -> Optional[float]:
        """
        Calculate Tanimoto similarity between two molecules
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Similarity score (0-1) or None if invalid
        """
        fp1 = MolecularUtils.get_molecular_fingerprint(smiles1)
        fp2 = MolecularUtils.get_molecular_fingerprint(smiles2)
        
        if fp1 is None or fp2 is None:
            return None
        
        try:
            from rdkit import DataStructs
            
            # Convert to RDKit fingerprint objects
            mol1 = MolecularUtils.smiles_to_mol(smiles1)
            mol2 = MolecularUtils.smiles_to_mol(smiles2)
            
            from rdkit.Chem import rdMolDescriptors
            fp1_obj = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2_obj = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)
            
            return DataStructs.TanimotoSimilarity(fp1_obj, fp2_obj)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return None
    
    @staticmethod
    def suggest_note_position_override(smiles: str, descriptors: List[str]) -> Optional[str]:
        """
        Suggest note position override based on descriptors and chemistry
        Some molecules defy MW-based classification
        
        Args:
            smiles: SMILES string
            descriptors: List of scent descriptors for this molecule
            
        Returns:
            Suggested note position or None for no override
        """
        # Strong top note indicators regardless of MW
        top_indicators = {'citrus', 'fresh', 'aldehydic', 'mint', 'eucalyptus', 'bergamot', 'lemon'}
        
        # Strong base note indicators regardless of MW
        base_indicators = {'musk', 'amber', 'vanilla', 'sandalwood', 'woody', 'leather'}
        
        descriptor_set = set(descriptors)
        
        # Check for strong indicators
        if descriptor_set & top_indicators:
            properties = MolecularUtils.calculate_molecular_properties(smiles)
            if properties and properties['molecular_weight'] < 250:  # Not too heavy
                return 'top'
        
        if descriptor_set & base_indicators:
            properties = MolecularUtils.calculate_molecular_properties(smiles)
            if properties and properties['molecular_weight'] > 150:  # Not too light
                return 'base'
        
        return None  # No override suggested