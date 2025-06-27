"""
Enhanced SMILES to Ingredient Mapper
Maps SMILES strings to real fragrance ingredient information
"""

def map_smiles_to_ingredient(smiles, molecular_weight=None):
    """Map SMILES string to fragrance ingredient information"""
    
    # Known ingredients database with real fragrance materials
    known_ingredients = {
        "COc1cc(C=O)ccc1O": {
            "name": "Vanillin",
            "odor_description": "Sweet, creamy, vanilla, balsamic",
            "source_type": "synthetic",
            "note_position": "base",
            "commercial_name": "Vanillin Synthetic",
            "cas_number": "121-33-5",
            "found_in_database": True
        },
        "CC(C)=CCCC(C)=CCO": {
            "name": "Geraniol",
            "odor_description": "Sweet, rosy, citrusy with floral facets",
            "source_type": "natural",
            "note_position": "top",
            "commercial_name": "Geraniol Natural",
            "cas_number": "106-24-1",
            "found_in_database": True
        },
        "CCCCCCCCCC=O": {
            "name": "Decanal",
            "odor_description": "Fresh, aldehydic, citrus peel, slightly fatty",
            "source_type": "synthetic",
            "note_position": "top",
            "commercial_name": "Aldehyde C-10",
            "cas_number": "112-31-2",
            "found_in_database": True
        },
        "COc1ccc(CCO)cc1": {
            "name": "Anisyl Alcohol",
            "odor_description": "Sweet, floral, slightly spicy",
            "source_type": "synthetic",
            "note_position": "middle",
            "commercial_name": "Anisyl Alcohol",
            "cas_number": "105-13-5",
            "found_in_database": True
        },
        "C=CCc1ccc(O)cc1": {
            "name": "Chavicol",
            "odor_description": "Spicy, clove-like, phenolic",
            "source_type": "natural",
            "note_position": "middle",
            "commercial_name": "Chavicol Natural",
            "cas_number": "501-92-8",
            "found_in_database": True
        },
        "CC(C)c1ccc(O)cc1": {
            "name": "Thymol",
            "odor_description": "Medicinal, thyme-like, phenolic",
            "source_type": "natural",
            "note_position": "top",
            "commercial_name": "Thymol Natural",
            "cas_number": "89-83-8",
            "found_in_database": True
        },
        "Oc1ccccc1": {
            "name": "Phenol",
            "odor_description": "Medicinal, phenolic, sharp",
            "source_type": "synthetic",
            "note_position": "base",
            "commercial_name": "Phenol Technical",
            "cas_number": "108-95-2",
            "found_in_database": True
        },
        "CCO": {
            "name": "Ethanol",
            "odor_description": "Alcoholic, slightly sweet",
            "source_type": "synthetic",
            "note_position": "top",
            "commercial_name": "Perfumer's Alcohol",
            "cas_number": "64-17-5",
            "found_in_database": True
        },
        "CO": {
            "name": "Methanol",
            "odor_description": "Alcoholic, pungent",
            "source_type": "synthetic",
            "note_position": "top",
            "commercial_name": "Methanol Technical",
            "cas_number": "67-56-1",
            "found_in_database": True
        },
        "O": {
            "name": "Water",
            "odor_description": "Odorless",
            "source_type": "natural",
            "note_position": "base",
            "commercial_name": "Distilled Water",
            "cas_number": "7732-18-5",
            "found_in_database": True
        },
        "CC1=CCC(CC1)C(C)(C)O": {
            "name": "Linalool",
            "odor_description": "Fresh, floral, citrusy, lavender-like",
            "source_type": "natural",
            "note_position": "top",
            "commercial_name": "Linalool Natural",
            "cas_number": "78-70-6",
            "found_in_database": True
        },
        "CC(=O)OCC=C(C)C": {
            "name": "Linalyl Acetate",
            "odor_description": "Fresh, floral, bergamot-like",
            "source_type": "natural",
            "note_position": "top",
            "commercial_name": "Linalyl Acetate Natural",
            "cas_number": "115-95-7",
            "found_in_database": True
        },
        "CC1(C)C2CCC1(C)C(O)C2": {
            "name": "Borneol",
            "odor_description": "Camphoraceous, woody, fresh",
            "source_type": "natural",
            "note_position": "top",
            "commercial_name": "Borneol Natural",
            "cas_number": "507-70-0",
            "found_in_database": True
        },
        # Add some common aromatic compounds that might appear
        "Cc1ccc(C(C)(C)C)cc1": {
            "name": "4-tert-Butyl Toluene",
            "odor_description": "Aromatic, sweet, slightly floral",
            "source_type": "synthetic",
            "note_position": "middle",
            "commercial_name": "Para-Tertiary Butyl Toluene",
            "cas_number": "98-51-1",
            "found_in_database": True
        },
        "Cc1ccccc1C": {
            "name": "o-Xylene",
            "odor_description": "Aromatic, sweet, petroleum-like",
            "source_type": "synthetic", 
            "note_position": "top",
            "commercial_name": "Ortho-Xylene",
            "cas_number": "95-47-6",
            "found_in_database": True
        }
    }
    
    # Clean SMILES input
    clean_smiles = smiles.strip() if smiles else ""
    
    # Direct lookup for known ingredients
    if clean_smiles in known_ingredients:
        return known_ingredients[clean_smiles]
    
    # Pattern-based analysis for unknown SMILES
    return analyze_unknown_smiles(clean_smiles, molecular_weight)

def analyze_unknown_smiles(smiles, molecular_weight=None):
    """Analyze unknown SMILES using structural patterns"""
    
    result = {
        "name": "Unknown Compound",
        "odor_description": "Unknown odor profile",
        "source_type": "unknown",
        "note_position": "middle", 
        "commercial_name": None,
        "cas_number": "Not available",
        "found_in_database": False
    }
    
    if not smiles:
        return result
    
    # Pattern detection for fragrance-like compounds
    smiles_lower = smiles.lower()
    
    # Aldehyde detection (C=O but not carboxylic acid)
    if "=O" in smiles and "C(=O)O" not in smiles and "c" not in smiles_lower:
        result["name"] = "Aldehyde Compound"
        result["odor_description"] = "Likely aldehydic, fresh, citrusy character"
        result["note_position"] = "top"
        result["source_type"] = "synthetic"
    
    # Alcohol detection (OH groups)
    elif ("CO" in smiles and "C=O" not in smiles and "COc" not in smiles) or "O" in smiles:
        result["name"] = "Alcohol Compound"
        result["odor_description"] = "Likely alcoholic, fresh character"
        result["note_position"] = "middle"
        result["source_type"] = "synthetic"
    
    # Aromatic detection (benzene rings)
    elif any(pattern in smiles_lower for pattern in ["c1cc", "c1ccc", "cccc", "ccc("]):
        result["name"] = "Aromatic Compound"
        result["odor_description"] = "Likely aromatic, complex character"
        result["note_position"] = "middle"
        result["source_type"] = "synthetic"
        
        # More specific aromatic patterns
        if "O" in smiles:
            result["name"] = "Phenolic Compound"
            result["odor_description"] = "Likely phenolic, medicinal character"
        elif "N" in smiles:
            result["name"] = "Aromatic Amine"
            result["odor_description"] = "Likely amine-like, potentially floral"
    
    # Ester detection
    elif "C(=O)O" in smiles and "c" not in smiles_lower:
        result["name"] = "Ester Compound"
        result["odor_description"] = "Likely fruity, sweet character"
        result["note_position"] = "top"
        result["source_type"] = "synthetic"
    
    # Terpene-like patterns (common in natural fragrances)
    elif "CC(C)" in smiles and "=" in smiles:
        result["name"] = "Terpene Compound"
        result["odor_description"] = "Likely fresh, natural, woody character"
        result["note_position"] = "top"
        result["source_type"] = "natural"
    
    # Ketone detection
    elif "C(=O)C" in smiles or "CC(=O)" in smiles:
        result["name"] = "Ketone Compound"
        result["odor_description"] = "Likely sweet, fruity character"
        result["note_position"] = "middle"
        result["source_type"] = "synthetic"
    
    # Molecular weight-based classification
    if molecular_weight:
        if molecular_weight < 120:
            result["note_position"] = "top"
            result["odor_description"] += " (highly volatile)"
        elif molecular_weight > 250:
            result["note_position"] = "base"
            result["odor_description"] += " (low volatility, fixative)"
        elif molecular_weight > 180:
            result["note_position"] = "middle"
            result["odor_description"] += " (moderate volatility)"
    
    return result

# Dataset statistics for reporting
DATASET_STATS = {
    "total_known_ingredients": 15,
    "natural_ingredients": 7,
    "synthetic_ingredients": 8,
    "coverage_percentage": "Coverage depends on your specific SMILES data"
}

# Additional utility functions
def get_ingredient_by_name(name):
    """Get ingredient info by name (case insensitive)"""
    # This is a reverse lookup function
    all_ingredients = {}
    # Call the main function to get the known ingredients
    temp_result = map_smiles_to_ingredient("dummy")  # Get structure
    
    known_smiles = [
        "COc1cc(C=O)ccc1O", "CC(C)=CCCC(C)=CCO", "CCCCCCCCCC=O",
        "COc1ccc(CCO)cc1", "C=CCc1ccc(O)cc1", "CC(C)c1ccc(O)cc1",
        "Oc1ccccc1", "CCO", "CO", "O", "CC1=CCC(CC1)C(C)(C)O",
        "CC(=O)OCC=C(C)C", "CC1(C)C2CCC1(C)C(O)C2",
        "Cc1ccc(C(C)(C)C)cc1", "Cc1ccccc1C"
    ]
    
    for smiles in known_smiles:
        ingredient = map_smiles_to_ingredient(smiles)
        if ingredient["name"].lower() == name.lower():
            return {"smiles": smiles, **ingredient}
    
    return None

def list_all_ingredients():
    """List all known ingredients"""
    known_smiles = [
        "COc1cc(C=O)ccc1O", "CC(C)=CCCC(C)=CCO", "CCCCCCCCCC=O",
        "COc1ccc(CCO)cc1", "C=CCc1ccc(O)cc1", "CC(C)c1ccc(O)cc1",
        "Oc1ccccc1", "CCO", "CO", "O", "CC1=CCC(CC1)C(C)(C)O",
        "CC(=O)OCC=C(C)C", "CC1(C)C2CCC1(C)C(O)C2",
        "Cc1ccc(C(C)(C)C)cc1", "Cc1ccccc1C"
    ]
    
    ingredients = []
    for smiles in known_smiles:
        ingredient = map_smiles_to_ingredient(smiles)
        ingredients.append({"smiles": smiles, **ingredient})
    
    return ingredients