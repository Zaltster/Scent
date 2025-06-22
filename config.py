# config.py
# Purpose: Central configuration for all settings
# What it does: Stores thresholds, proportions, file paths, model settings
# Responsibilities: MW thresholds (150/280), proportion ratios (30/50/20), embedding model choice
# Contains: All magic numbers, file paths, tweakable parameters

import os
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MERGED_DATASET_PATH = DATA_DIR / "curated_GS_LF_merged_4983.csv"

# Dataset column names
SMILES_COL = 'nonStereoSMILES'
DESCRIPTORS_COL = 'descriptors'

# All 138 descriptor columns in order
DESCRIPTOR_COLUMNS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic', 
    'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot', 
    'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 'cabbage', 
    'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy', 'cherry', 
    'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 'coconut', 'coffee', 
    'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic', 'creamy', 'cucumber', 
    'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 
    'fresh', 'fruit skin', 'fruity', 'garlic', 'gassy', 'geranium', 'grape', 
    'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 
    'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 
    'leafy', 'leathery', 'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 
    'metallic', 'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 
    'nutty', 'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 
    'ozone', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted', 
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 'solvent', 
    'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 
    'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'vetiver', 
    'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

# Molecular weight thresholds for note classification
MW_THRESHOLDS = {
    'top_max': 160,      # Top notes: MW < 160
    'middle_max': 280,   # Middle notes: MW 160-280
    # Base notes: MW > 280
}

# Default proportion ratios for different fragrance types
PROPORTION_PROFILES = {
    'balanced': {'top': 0.30, 'middle': 0.50, 'base': 0.20},
    'fresh': {'top': 0.40, 'middle': 0.45, 'base': 0.15},      # More top impact
    'oriental': {'top': 0.20, 'middle': 0.40, 'base': 0.40},   # More base richness
    'floral': {'top': 0.25, 'middle': 0.55, 'base': 0.20},     # Heart-focused
}

# Text processing settings
EMBEDDING_SETTINGS = {
    'model_name': 'all-MiniLM-L6-v2',  # Sentence transformers model
    'top_descriptors_count': 6,        # How many descriptors to select
    'similarity_threshold': 0.3,       # Minimum similarity to consider
}

# Molecule selection settings
MOLECULE_SELECTION = {
    'max_molecules_per_note': 5,       # Max molecules per note layer
    'min_molecules_per_note': 2,       # Min molecules per note layer
    'overlap_bonus': 0.1,              # Bonus for molecules with multiple descriptors
    'diversity_weight': 0.2,           # Weight for structural diversity
}

# Formula composition settings
FORMULA_SETTINGS = {
    'min_total_percentage': 95.0,      # Minimum total percentage (allow for rounding)
    'max_total_percentage': 105.0,     # Maximum total percentage
    'precision': 1,                    # Decimal places for percentages
}

# Validation settings
VALIDATION = {
    'max_formula_components': 15,      # Maximum total molecules in formula
    'min_formula_components': 5,       # Minimum total molecules in formula
    'check_chemical_compatibility': True,
    'warn_unusual_combinations': True,
}

# Debug and logging settings
DEBUG = True
LOG_LEVEL = 'INFO'
VERBOSE_OUTPUT = True