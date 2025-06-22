# data_loader.py
# Purpose: Handles all dataset loading and preprocessing
# What it does: Loads your CSV, validates format, prepares data structures
# Responsibilities: Read CSV, handle missing data, create lookup tables
# Input: File path to your merged CSV
# Output: Clean pandas DataFrame ready for the pipeline

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config import (
    MERGED_DATASET_PATH, SMILES_COL, DESCRIPTORS_COL, 
    DESCRIPTOR_COLUMNS, DEBUG
)

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and preprocessing of the fragrance molecule dataset
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize DataLoader
        
        Args:
            dataset_path: Path to dataset CSV. Uses config default if None.
        """
        self.dataset_path = dataset_path or MERGED_DATASET_PATH
        self.df = None
        self.descriptor_matrix = None
        self.smiles_to_descriptors = None
        self.descriptor_to_smiles = None
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the main dataset from CSV
        
        Returns:
            pd.DataFrame: Loaded and validated dataset
        """
        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            
            if not Path(self.dataset_path).exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            # Load CSV
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded {len(self.df)} molecules from dataset")
            
            # Validate format
            self._validate_dataset_format()
            
            # Clean and preprocess
            self._preprocess_dataset()
            
            # Create lookup structures
            self._create_lookup_tables()
            
            logger.info("Dataset loaded and preprocessed successfully")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _validate_dataset_format(self):
        """Validate that dataset has expected columns and format"""
        required_columns = [SMILES_COL, DESCRIPTORS_COL] + DESCRIPTOR_COLUMNS
        
        missing_columns = set(required_columns) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info("Dataset format validation passed")
    
    def _preprocess_dataset(self):
        """Clean and preprocess the dataset"""
        initial_count = len(self.df)
        
        # Remove rows with missing SMILES
        self.df = self.df.dropna(subset=[SMILES_COL])
        
        # Remove empty SMILES
        self.df = self.df[self.df[SMILES_COL].str.strip() != '']
        
        # Remove duplicates based on SMILES
        self.df = self.df.drop_duplicates(subset=[SMILES_COL])
        
        # Ensure binary columns are numeric
        for col in DESCRIPTOR_COLUMNS:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
        
        # Remove molecules with no descriptors
        descriptor_sum = self.df[DESCRIPTOR_COLUMNS].sum(axis=1)
        self.df = self.df[descriptor_sum > 0]
        
        final_count = len(self.df)
        logger.info(f"Dataset preprocessing: {initial_count} → {final_count} molecules")
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
    
    def _create_lookup_tables(self):
        """Create efficient lookup structures for the pipeline"""
        # Extract binary descriptor matrix
        self.descriptor_matrix = self.df[DESCRIPTOR_COLUMNS].values
        
        # Create SMILES to descriptors mapping
        self.smiles_to_descriptors = {}
        for idx, row in self.df.iterrows():
            smiles = row[SMILES_COL]
            active_descriptors = [
                desc for desc in DESCRIPTOR_COLUMNS 
                if row[desc] == 1
            ]
            self.smiles_to_descriptors[smiles] = active_descriptors
        
        # Create descriptor to SMILES mapping
        self.descriptor_to_smiles = {desc: [] for desc in DESCRIPTOR_COLUMNS}
        for idx, row in self.df.iterrows():
            smiles = row[SMILES_COL]
            for desc in DESCRIPTOR_COLUMNS:
                if row[desc] == 1:
                    self.descriptor_to_smiles[desc].append(smiles)
        
        logger.info("Lookup tables created successfully")
    
    def get_molecules_by_descriptor(self, descriptor: str) -> List[str]:
        """
        Get all SMILES that have a specific descriptor
        
        Args:
            descriptor: Scent descriptor name
            
        Returns:
            List of SMILES strings
        """
        if descriptor not in self.descriptor_to_smiles:
            logger.warning(f"Unknown descriptor: {descriptor}")
            return []
        
        return self.descriptor_to_smiles[descriptor]
    
    def get_molecules_by_descriptors(self, descriptors: List[str]) -> Dict[str, List[str]]:
        """
        Get molecules for multiple descriptors
        
        Args:
            descriptors: List of descriptor names
            
        Returns:
            Dict mapping descriptor to list of SMILES
        """
        result = {}
        for desc in descriptors:
            result[desc] = self.get_molecules_by_descriptor(desc)
        return result
    
    def get_descriptors_for_molecule(self, smiles: str) -> List[str]:
        """
        Get all descriptors for a specific molecule
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of descriptor names
        """
        return self.smiles_to_descriptors.get(smiles, [])
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the loaded dataset
        
        Returns:
            Dict with dataset statistics
        """
        stats = {
            'total_molecules': len(self.df),
            'total_descriptors': len(DESCRIPTOR_COLUMNS),
            'molecules_per_descriptor': {},
            'descriptors_per_molecule': {},
            'most_common_descriptors': {},
            'rarest_descriptors': {}
        }
        
        # Count molecules per descriptor
        for desc in DESCRIPTOR_COLUMNS:
            count = self.df[desc].sum()
            stats['molecules_per_descriptor'][desc] = count
        
        # Count descriptors per molecule
        descriptor_counts = self.df[DESCRIPTOR_COLUMNS].sum(axis=1)
        stats['descriptors_per_molecule'] = {
            'mean': float(descriptor_counts.mean()),
            'median': float(descriptor_counts.median()),
            'min': int(descriptor_counts.min()),
            'max': int(descriptor_counts.max())
        }
        
        # Most and least common descriptors
        descriptor_counts = {
            desc: self.df[desc].sum() 
            for desc in DESCRIPTOR_COLUMNS
        }
        sorted_descriptors = sorted(descriptor_counts.items(), key=lambda x: x[1], reverse=True)
        
        stats['most_common_descriptors'] = dict(sorted_descriptors[:10])
        stats['rarest_descriptors'] = dict(sorted_descriptors[-10:])
        
        return stats
    
    def search_molecules(self, query_descriptors: List[str], 
                        min_matches: int = 1) -> pd.DataFrame:
        """
        Search for molecules matching query descriptors
        
        Args:
            query_descriptors: List of descriptors to search for
            min_matches: Minimum number of descriptors that must match
            
        Returns:
            DataFrame with matching molecules and match counts
        """
        # Calculate match scores for each molecule
        match_scores = np.zeros(len(self.df))
        
        for desc in query_descriptors:
            if desc in DESCRIPTOR_COLUMNS:
                match_scores += self.df[desc].values
        
        # Filter by minimum matches
        mask = match_scores >= min_matches
        result_df = self.df[mask].copy()
        result_df['match_score'] = match_scores[mask]
        
        # Sort by match score
        result_df = result_df.sort_values('match_score', ascending=False)
        
        return result_df