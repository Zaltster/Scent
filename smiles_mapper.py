"""
SMILES to Fragrance Ingredient Mapper - AUTO-GENERATED
Converts SMILES chemical codes to actual fragrance ingredient names
Generated from dataset analysis of 4983 unique molecules
"""

import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SmilesMapper:
    """Maps SMILES codes to actual fragrance ingredient names"""

    def __init__(self):
        # Known fragrance molecules database (exact matches)
        self.smiles_database = {
            'CC(C)=CCCC(C)=CCO': {'name': 'Nerol', 'odor': 'Rose, Citrus, Green', 'source': 'Natural'},
            'CC(C)=CCCC(C)=CC(C)O': {'name': 'Linalool', 'odor': 'Lavender, Bergamot, Fresh', 'source': 'Natural'},
            'CC1=CCC(CC1)C(C)C': {'name': 'Limonene', 'odor': 'Orange, Lemon, Fresh', 'source': 'Natural'},
            'CC(C)C1CCC(C)CC1': {'name': 'Menthol', 'odor': 'Mint, Cooling, Fresh', 'source': 'Natural'},
            'CC(C)=CCC=C(C)C': {'name': 'Myrcene', 'odor': 'Herbal, Resinous, Earthy', 'source': 'Natural'},
            'CC1=CCC(=CC1)C(C)C': {'name': 'p-Cymene', 'odor': 'Citrus, Woody, Warm', 'source': 'Natural'},
            'CC(C)=CCCC(C)=CC=O': {'name': 'Citronellal', 'odor': 'Lemon, Rose, Fresh', 'source': 'Natural'},
            'OCC1=CC=CC=C1': {'name': 'Benzyl Alcohol', 'odor': 'Rose, Floral, Sweet', 'source': 'Synthetic'},
            'CC(=O)OCC1=CC=CC=C1': {'name': 'Benzyl Acetate', 'odor': 'Jasmine, Sweet, Fruity', 'source': 'Synthetic'},
            'OC1=CC=CC=C1': {'name': 'Phenol', 'odor': 'Rose, Medicinal, Sharp', 'source': 'Synthetic'},
            'CC1=CC=CC=C1O': {'name': 'o-Cresol', 'odor': 'Rose, Phenolic, Warm', 'source': 'Synthetic'},
            'OCc1ccc(O)cc1': {'name': '4-Hydroxybenzyl Alcohol', 'odor': 'Floral, Sweet, Vanilla', 'source': 'Synthetic'},
            'O=Cc1ccc(O)cc1': {'name': '4-Hydroxybenzaldehyde', 'odor': 'Vanilla, Sweet, Almond', 'source': 'Synthetic'},
            'O=C(O)c1ccc(O)cc1': {'name': '4-Hydroxybenzoic Acid', 'odor': 'Phenolic, Medicinal', 'source': 'Synthetic'},
            'COC1=CC(=CC=C1O)C=O': {'name': 'Vanillin', 'odor': 'Vanilla, Sweet, Warm', 'source': 'Synthetic'},
            'COC1=CC=C(C=C1)C=O': {'name': 'p-Anisaldehyde', 'odor': 'Sweet, Floral, Anisic', 'source': 'Synthetic'},
            'COC1=CC=C(C=C1)CC=C': {'name': 'Eugenol', 'odor': 'Clove, Spicy, Sweet', 'source': 'Natural'},
            'COC1=CC=CC=C1': {'name': 'Anisole', 'odor': 'Sweet, Anisic, Ethereal', 'source': 'Synthetic'},
            'CCCCCCCCCC=O': {'name': 'Decanal', 'odor': 'Orange, Waxy, Aldehydic', 'source': 'Synthetic'},
            'CCCCCCCCC=O': {'name': 'Nonanal', 'odor': 'Rose, Waxy, Aldehydic', 'source': 'Synthetic'},
            'CCCCCCCC=O': {'name': 'Octanal', 'odor': 'Citrus, Orange, Fresh', 'source': 'Synthetic'},
            'CCCCCCC=O': {'name': 'Heptanal', 'odor': 'Green, Fatty, Fresh', 'source': 'Synthetic'},
            'CCCCCC=O': {'name': 'Hexanal', 'odor': 'Green, Apple, Fresh', 'source': 'Synthetic'},
            'CC1(C)C2CCC1(C)C(=O)C2': {'name': 'Camphor', 'odor': 'Camphor, Cooling, Medicinal', 'source': 'Natural'},
            'CC12CCC(CC1)C(C)(C)C2O': {'name': 'Borneol', 'odor': 'Camphor, Woody, Fresh', 'source': 'Natural'},
            'CC1=CCC2CC1C2(C)C': {'name': 'Camphene', 'odor': 'Pine, Fresh, Woody', 'source': 'Natural'},
            'CCC=CCC=O': {'name': 'trans-2-Hexenal', 'odor': 'Green, Apple, Fresh', 'source': 'Natural'},
            'CC=CCC=O': {'name': 'trans-2-Pentenal', 'odor': 'Green, Fruity, Sharp', 'source': 'Natural'},
            'CC(=O)OCC(C)C': {'name': 'Isobutyl Acetate', 'odor': 'Banana, Fruity, Sweet', 'source': 'Synthetic'},
            'CC(=O)OCCC': {'name': 'Propyl Acetate', 'odor': 'Pear, Fruity, Ethereal', 'source': 'Synthetic'},
            'CC(=O)OCC': {'name': 'Ethyl Acetate', 'odor': 'Fruity, Solvent, Sweet', 'source': 'Synthetic'},
            'CC(=O)OCCCC': {'name': 'Butyl Acetate', 'odor': 'Banana, Apple, Sweet', 'source': 'Synthetic'},
            'CC1CCOC(=O)C1': {'name': 'γ-Hexalactone', 'odor': 'Coconut, Creamy, Sweet', 'source': 'Synthetic'},
            'CCCCCCC1CCOC(=O)C1': {'name': 'γ-Decalactone', 'odor': 'Peach, Creamy, Fruity', 'source': 'Synthetic'},
            'CCCCCCCCCCCCCCCC(=O)O': {'name': 'Palmitic Acid', 'odor': 'Waxy, Soapy, Fatty', 'source': 'Natural'},
            'CCCCCCCCCCCCCC(=O)O': {'name': 'Myristic Acid', 'odor': 'Waxy, Coconut, Fatty', 'source': 'Natural'},
            'CCCCCCCCCC(=O)O': {'name': 'Decanoic Acid', 'odor': 'Fatty, Rancid, Waxy', 'source': 'Natural'},
            'O=C(O)CCc1ccccc1': {'name': 'Phenylacetic Acid', 'odor': 'Honey, Rose, Sweet', 'source': 'Natural'},
            'CCC(=O)C(=O)O': {'name': '2-Oxobutanoic Acid', 'odor': 'Fruity, Sharp, Fermented', 'source': 'Synthetic'},
            'CCCSSCCC': {'name': 'Diethyl Disulfide', 'odor': 'Pungent, Garlic, Savory', 'source': 'Synthetic'},
            'CCSC': {'name': 'Ethyl Methyl Sulfide', 'odor': 'Sulfurous, Pungent, Cabbage', 'source': 'Synthetic'},
            'CSC': {'name': 'Dimethyl Sulfide', 'odor': 'Sulfurous, Cabbage, Marine', 'source': 'Natural'},
            'CCSCC': {'name': 'Diethyl Sulfide', 'odor': 'Garlic, Penetrating, Sulfurous', 'source': 'Synthetic'},
            'CC(O)CN': {'name': '1-Amino-2-propanol', 'odor': 'Amine, Basic, Fishy', 'source': 'Synthetic'},
            'CCCCCCCO': {'name': 'Heptanol', 'odor': 'Green, Fatty, Waxy', 'source': 'Synthetic'},
            'CCCCCCO': {'name': 'Hexanol', 'odor': 'Green, Herbaceous, Fresh', 'source': 'Synthetic'},
            'CC1CCC(C(C)C)C(OCC(C)(O)CO)C1': {'name': 'Complex Terpene Ether', 'odor': 'Woody, Fresh, Complex', 'source': 'Synthetic'},
            'CCCCCCC=CCCCCCCCC(=O)OCC=C(C)CC': {'name': 'Long-chain Ester', 'odor': 'Waxy, Fatty, Base', 'source': 'Synthetic'},
        }

        # Pattern-based recognition for structural classes
        self.structure_patterns = [
            (r'.*c1ccccc1.*|.*C1=CC=CC=C1.*', 'Aromatic Compound', 'Floral, Sweet, Phenolic', 'Natural or Synthetic'),
            (r'CC\\(C\\)=C.*|.*=C\\(C\\)C.*', 'Terpene', 'Fresh, Natural, Citrusy', 'Natural or Synthetic'),
            (r'.*C=O$', 'Aldehyde', 'Fresh, Sharp, Aldehydic', 'Natural or Synthetic'),
            (r'.*CO$|.*C\\(.*\\)O$', 'Alcohol', 'Sweet, Floral, Soft', 'Natural or Synthetic'),
            (r'.*C\\(=O\\)O[^H].*', 'Ester', 'Fruity, Sweet, Pleasant', 'Natural or Synthetic'),
            (r'.*C\\(=O\\)O$', 'Carboxylic Acid', 'Sharp, Pungent, Sour', 'Natural or Synthetic'),
            (r'.*C\\(=O\\)C.*', 'Ketone', 'Fruity, Solvent, Sharp', 'Natural or Synthetic'),
            (r'.*C1.*OC\\(=O\\).*C1.*', 'Lactone', 'Creamy, Sweet, Coconut', 'Natural or Synthetic'),
            (r'.*S.*', 'Sulfur Compound', 'Pungent, Savory, Penetrating', 'Natural or Synthetic'),
            (r'.*N.*', 'Nitrogen Compound', 'Basic, Amine, Fishy', 'Natural or Synthetic'),
            (r'.*c.*O|.*C.*=C.*O', 'Phenolic Compound', 'Medicinal, Sharp, Smoky', 'Natural or Synthetic'),
        ]

        # MW-based classification for unknowns
        self.mw_classifications = [
            (50, 100, "Very Light Volatile", "Very Fresh, Sharp, Penetrating"),
            (100, 150, "Light Volatile", "Fresh, Citrusy, Green"),
            (150, 200, "Medium Volatile", "Floral, Spicy, Balanced"),
            (200, 250, "Heavy Volatile", "Woody, Resinous, Warm"),
            (250, 350, "Very Heavy", "Base Notes, Fixative, Long-lasting"),
            (350, 500, "Ultra Heavy", "Deep Base, Very Long-lasting"),
        ]

    def get_ingredient_name(self, smiles: str, molecular_weight: Optional[float] = None) -> Tuple[str, str]:
        """
        Convert SMILES to ingredient name and description
        
        Args:
            smiles: SMILES chemical notation
            molecular_weight: Optional molecular weight for classification
            
        Returns:
            Tuple of (ingredient_name, odor_description)
        """
        # Direct lookup first
        if smiles in self.smiles_database:
            ingredient_info = self.smiles_database[smiles]
            return ingredient_info['name'], ingredient_info['odor']
        
        # Pattern-based recognition
        for pattern, compound_type, odor_desc, source_type in self.structure_patterns:
            if re.search(pattern, smiles, re.IGNORECASE):
                return compound_type, odor_desc
        
        # Fallback to MW-based classification
        if molecular_weight:
            mw_class, mw_desc = self._get_mw_classification(molecular_weight)
            return f"Unknown {mw_class}", mw_desc
        
        # Ultimate fallback
        return "Unknown Fragrance Material", "Unknown properties"
    
    def _get_mw_classification(self, mw: float) -> Tuple[str, str]:
        """Get classification based on molecular weight"""
        for min_mw, max_mw, classification, description in self.mw_classifications:
            if min_mw <= mw <= max_mw:
                return classification, description
        return "Unknown Weight Class", "Unknown properties"
    
    def is_natural_or_synthetic(self, smiles: str) -> str:
        """Determine if ingredient is typically natural or synthetic"""
        if smiles in self.smiles_database:
            return self.smiles_database[smiles]['source']
        
        # Heuristic classification
        if any(pattern in smiles for pattern in ['CC(C)=C', '=C(C)C', 'C1CCC']):
            return "Natural or Synthetic"
        elif len(smiles) > 30:
            return "Synthetic"
        elif 'S' in smiles or 'N' in smiles and 'c' not in smiles.lower():
            return "Synthetic"
        else:
            return "Natural or Synthetic"
    
    def get_commercial_name(self, smiles: str) -> Optional[str]:
        """Get commercial/trade name if available"""
        if smiles in self.smiles_database:
            return self.smiles_database[smiles].get('commercial')
        return None

# Global instance
smiles_mapper = SmilesMapper()

def map_smiles_to_ingredient(smiles: str, molecular_weight: Optional[float] = None) -> Dict[str, str]:
    """
    Convenience function to map SMILES to ingredient information
    
    Returns:
        Dict with keys: name, odor_description, source_type, commercial_name
    """
    name, odor = smiles_mapper.get_ingredient_name(smiles, molecular_weight)
    
    return {
        'name': name,
        'odor_description': odor,
        'source_type': smiles_mapper.is_natural_or_synthetic(smiles),
        'commercial_name': smiles_mapper.get_commercial_name(smiles),
        'smiles': smiles
    }

# Dataset Statistics (for reference)
DATASET_STATS = {
    "total_unique_molecules": 4983,
    "pattern_breakdown": {
        "Phenolic Compound": {"count": 1034, "percentage": 20.8},
        "Ketone": {"count": 752, "percentage": 15.1},
        "Ester": {"count": 627, "percentage": 12.6},
        "Aromatic Compound": {"count": 615, "percentage": 12.3},
        "Sulfur Compound": {"count": 488, "percentage": 9.8},
        "Terpene": {"count": 468, "percentage": 9.4},
        "Alcohol": {"count": 370, "percentage": 7.4},
        "Nitrogen Compound": {"count": 297, "percentage": 6.0},
        "Aldehyde": {"count": 174, "percentage": 3.5},
        "Unknown Compound": {"count": 113, "percentage": 2.3},
        "Simple Molecule": {"count": 15, "percentage": 0.3},
        "Complex Molecule": {"count": 3, "percentage": 0.1},
        "Complex Terpene Ether": {"count": 1, "percentage": 0.0},
        "1-Amino-2-propanol": {"count": 1, "percentage": 0.0},
        "2-Oxobutanoic Acid": {"count": 1, "percentage": 0.0},
    }
}
