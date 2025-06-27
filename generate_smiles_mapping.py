#!/usr/bin/env python3
"""
Generate SMILES Mapping Script
Takes dataset analysis CSV and automatically generates ingredient mappings
"""

import pandas as pd
import re
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmilesMappingGenerator:
    """Automatically generates ingredient mappings from SMILES analysis"""
    
    def __init__(self):
        # Known fragrance molecules - comprehensive database
        self.known_mappings = {
            # Essential Oil Components
            'CC(C)=CCCC(C)=CCO': ('Geraniol', 'Rose, Citrus, Sweet', 'Natural'),
            'CC(C)=CCCC(C)=CC(C)O': ('Linalool', 'Lavender, Bergamot, Fresh', 'Natural'),
            'CC1=CCC(CC1)C(C)C': ('Limonene', 'Orange, Lemon, Fresh', 'Natural'),
            'CC(C)C1CCC(C)CC1': ('Menthol', 'Mint, Cooling, Fresh', 'Natural'),
            'CC(C)=CCC=C(C)C': ('Myrcene', 'Herbal, Resinous, Earthy', 'Natural'),
            'CC1=CCC(=CC1)C(C)C': ('p-Cymene', 'Citrus, Woody, Warm', 'Natural'),
            'CC(C)=CCCC(C)=CCO': ('Nerol', 'Rose, Citrus, Green', 'Natural'),
            'CC(C)=CCCC(C)=CC=O': ('Citronellal', 'Lemon, Rose, Fresh', 'Natural'),
            
            # Floral Components
            'OCC1=CC=CC=C1': ('Benzyl Alcohol', 'Rose, Floral, Sweet', 'Synthetic'),
            'CC(=O)OCC1=CC=CC=C1': ('Benzyl Acetate', 'Jasmine, Sweet, Fruity', 'Synthetic'),
            'OC1=CC=CC=C1': ('Phenol', 'Rose, Medicinal, Sharp', 'Synthetic'),
            'CC1=CC=CC=C1O': ('o-Cresol', 'Rose, Phenolic, Warm', 'Synthetic'),
            'OCc1ccc(O)cc1': ('4-Hydroxybenzyl Alcohol', 'Floral, Sweet, Vanilla', 'Synthetic'),
            'O=Cc1ccc(O)cc1': ('4-Hydroxybenzaldehyde', 'Vanilla, Sweet, Almond', 'Synthetic'),
            'O=C(O)c1ccc(O)cc1': ('4-Hydroxybenzoic Acid', 'Phenolic, Medicinal', 'Synthetic'),
            
            # Vanilla and Sweet
            'COC1=CC(=CC=C1O)C=O': ('Vanillin', 'Vanilla, Sweet, Warm', 'Synthetic'),
            'COC1=CC=C(C=C1)C=O': ('p-Anisaldehyde', 'Sweet, Floral, Anisic', 'Synthetic'),
            'COC1=CC=C(C=C1)CC=C': ('Eugenol', 'Clove, Spicy, Sweet', 'Natural'),
            'COC1=CC=CC=C1': ('Anisole', 'Sweet, Anisic, Ethereal', 'Synthetic'),
            
            # Citrus and Aldehydes
            'CCCCCCCCCC=O': ('Decanal', 'Orange, Waxy, Aldehydic', 'Synthetic'),
            'CCCCCCCCC=O': ('Nonanal', 'Rose, Waxy, Aldehydic', 'Synthetic'),
            'CCCCCCCC=O': ('Octanal', 'Citrus, Orange, Fresh', 'Synthetic'),
            'CCCCCCC=O': ('Heptanal', 'Green, Fatty, Fresh', 'Synthetic'),
            'CCCCCC=O': ('Hexanal', 'Green, Apple, Fresh', 'Synthetic'),
            
            # Woody and Camphoraceous
            'CC1(C)C2CCC1(C)C(=O)C2': ('Camphor', 'Camphor, Cooling, Medicinal', 'Natural'),
            'CC12CCC(CC1)C(C)(C)C2O': ('Borneol', 'Camphor, Woody, Fresh', 'Natural'),
            'CC1=CCC2CC1C2(C)C': ('Camphene', 'Pine, Fresh, Woody', 'Natural'),
            
            # Green and Fresh
            'CCC=CCC=O': ('trans-2-Hexenal', 'Green, Apple, Fresh', 'Natural'),
            'CC=CCC=O': ('trans-2-Pentenal', 'Green, Fruity, Sharp', 'Natural'),
            
            # Fruity Esters
            'CC(=O)OCC(C)C': ('Isobutyl Acetate', 'Banana, Fruity, Sweet', 'Synthetic'),
            'CC(=O)OCCC': ('Propyl Acetate', 'Pear, Fruity, Ethereal', 'Synthetic'),
            'CC(=O)OCC': ('Ethyl Acetate', 'Fruity, Solvent, Sweet', 'Synthetic'),
            'CC(=O)OCCCC': ('Butyl Acetate', 'Banana, Apple, Sweet', 'Synthetic'),
            
            # Lactones
            'CC1CCOC(=O)C1': ('γ-Hexalactone', 'Coconut, Creamy, Sweet', 'Synthetic'),
            'CCCCCCC1CCOC(=O)C1': ('γ-Decalactone', 'Peach, Creamy, Fruity', 'Synthetic'),
            
            # Acids
            'CCCCCCCCCCCCCCCC(=O)O': ('Palmitic Acid', 'Waxy, Soapy, Fatty', 'Natural'),
            'CCCCCCCCCCCCCC(=O)O': ('Myristic Acid', 'Waxy, Coconut, Fatty', 'Natural'),
            'CCCCCCCCCC(=O)O': ('Decanoic Acid', 'Fatty, Rancid, Waxy', 'Natural'),
            'O=C(O)CCc1ccccc1': ('Phenylacetic Acid', 'Honey, Rose, Sweet', 'Natural'),
            'CCC(=O)C(=O)O': ('2-Oxobutanoic Acid', 'Fruity, Sharp, Fermented', 'Synthetic'),
            
            # Sulfur Compounds
            'CCCSSCCC': ('Diethyl Disulfide', 'Pungent, Garlic, Savory', 'Synthetic'),
            'CCSC': ('Ethyl Methyl Sulfide', 'Sulfurous, Pungent, Cabbage', 'Synthetic'),
            'CSC': ('Dimethyl Sulfide', 'Sulfurous, Cabbage, Marine', 'Natural'),
            'CCSCC': ('Diethyl Sulfide', 'Garlic, Penetrating, Sulfurous', 'Synthetic'),
            
            # Alcohols and Phenols
            'CC(O)CN': ('1-Amino-2-propanol', 'Amine, Basic, Fishy', 'Synthetic'),
            'CCCCCCCO': ('Heptanol', 'Green, Fatty, Waxy', 'Synthetic'),
            'CCCCCCO': ('Hexanol', 'Green, Herbaceous, Fresh', 'Synthetic'),
            
            # Complex molecules from your examples
            'CC1CCC(C(C)C)C(OCC(C)(O)CO)C1': ('Complex Terpene Ether', 'Woody, Fresh, Complex', 'Synthetic'),
            'CCCCCCC=CCCCCCCCC(=O)OCC=C(C)CC': ('Long-chain Ester', 'Waxy, Fatty, Base', 'Synthetic'),
        }
        
        # Pattern-based classification rules
        self.pattern_rules = [
            # Aromatic patterns
            (r'.*c1ccccc1.*|.*C1=CC=CC=C1.*', 'Aromatic Compound', 'Floral, Sweet, Phenolic'),
            
            # Terpene patterns
            (r'CC\(C\)=C.*|.*=C\(C\)C.*', 'Terpene', 'Fresh, Natural, Citrusy'),
            
            # Aldehyde patterns
            (r'.*C=O$', 'Aldehyde', 'Fresh, Sharp, Aldehydic'),
            
            # Alcohol patterns
            (r'.*CO$|.*C\(.*\)O$', 'Alcohol', 'Sweet, Floral, Soft'),
            
            # Ester patterns
            (r'.*C\(=O\)O[^H].*', 'Ester', 'Fruity, Sweet, Pleasant'),
            
            # Acid patterns
            (r'.*C\(=O\)O$', 'Carboxylic Acid', 'Sharp, Pungent, Sour'),
            
            # Ketone patterns
            (r'.*C\(=O\)C.*', 'Ketone', 'Fruity, Solvent, Sharp'),
            
            # Lactone patterns
            (r'.*C1.*OC\(=O\).*C1.*', 'Lactone', 'Creamy, Sweet, Coconut'),
            
            # Sulfur patterns
            (r'.*S.*', 'Sulfur Compound', 'Pungent, Savory, Penetrating'),
            
            # Nitrogen patterns
            (r'.*N.*', 'Nitrogen Compound', 'Basic, Amine, Fishy'),
            
            # Phenol patterns
            (r'.*c.*O|.*C.*=C.*O', 'Phenolic Compound', 'Medicinal, Sharp, Smoky'),
        ]
        
        # MW-based classifications
        self.mw_classes = [
            (50, 100, 'Very Light', 'Very Fresh, Volatile, Sharp'),
            (100, 150, 'Light', 'Fresh, Citrusy, Green'),
            (150, 200, 'Medium', 'Floral, Spicy, Balanced'),
            (200, 250, 'Heavy', 'Woody, Resinous, Warm'),
            (250, 350, 'Very Heavy', 'Base, Fixative, Long-lasting'),
        ]
    
    def classify_by_patterns(self, smiles: str) -> Tuple[str, str, str]:
        """Classify SMILES using pattern matching"""
        
        # Check known mappings first
        if smiles in self.known_mappings:
            name, odor, source = self.known_mappings[smiles]
            return name, odor, source
        
        # Pattern-based classification
        for pattern, compound_type, odor_desc in self.pattern_rules:
            if re.search(pattern, smiles, re.IGNORECASE):
                return compound_type, odor_desc, self._guess_source_type(smiles)
        
        # Fallback based on length and complexity
        if len(smiles) < 10:
            return 'Simple Molecule', 'Basic, Building Block', 'Synthetic'
        elif len(smiles) > 40:
            return 'Complex Molecule', 'Rich, Complex, Multifaceted', 'Synthetic'
        else:
            return 'Unknown Compound', 'Unknown Properties', 'Unknown'
    
    def _guess_source_type(self, smiles: str) -> str:
        """Guess if compound is natural or synthetic based on structure"""
        # Simple heuristics
        if any(pattern in smiles for pattern in ['CC(C)=C', '=C(C)C', 'C1CCC']):
            return 'Natural or Synthetic'
        elif len(smiles) > 30:
            return 'Synthetic'
        elif 'S' in smiles or 'N' in smiles:
            return 'Synthetic'
        else:
            return 'Natural or Synthetic'
    
    def classify_by_mw(self, mw: float) -> Tuple[str, str]:
        """Classify by molecular weight"""
        for min_mw, max_mw, weight_class, odor_desc in self.mw_classes:
            if min_mw <= mw <= max_mw:
                return weight_class, odor_desc
        return 'Unknown Weight', 'Unknown Properties'
    
    def generate_mapping_code(self, analysis_csv: str = "dataset_analysis.csv", output_file: str = "updated_smiles_mapper.py"):
        """Generate updated SMILES mapper code from analysis CSV"""
        
        try:
            # Load analysis data
            df = pd.read_csv(analysis_csv)
            print(f"✅ Loaded {len(df)} SMILES from analysis")
            
            # Process each SMILES
            mappings = {}
            pattern_stats = {}
            
            for idx, row in df.iterrows():
                smiles = row['smiles']
                frequency = row['frequency']
                
                # Classify this SMILES
                name, odor, source = self.classify_by_patterns(smiles)
                
                # Track pattern usage
                pattern_key = name
                if pattern_key not in pattern_stats:
                    pattern_stats[pattern_key] = 0
                pattern_stats[pattern_key] += frequency
                
                # Store mapping
                mappings[smiles] = {
                    'name': name,
                    'odor': odor,
                    'source': source,
                    'frequency': frequency
                }
            
            # Generate the new mapper code
            self._write_updated_mapper(mappings, pattern_stats, output_file)
            
            # Print statistics
            print(f"\n📊 PATTERN CLASSIFICATION RESULTS:")
            print(f"   Total SMILES processed: {len(mappings):,}")
            
            total_freq = sum(row['frequency'] for _, row in df.iterrows())
            print(f"   Total molecule instances: {total_freq:,}")
            
            print(f"\n🏷️  CLASSIFICATION BREAKDOWN:")
            for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_freq) * 100
                print(f"   {pattern:<25}: {count:,} instances ({percentage:.1f}%)")
            
            print(f"\n💾 Updated mapper saved to: {output_file}")
            print(f"💡 Replace your current smiles_mapper.py with this file!")
            
            return mappings
            
        except FileNotFoundError:
            print(f"❌ Analysis file not found: {analysis_csv}")
            print("💡 Run 'python analyze_dataset.py' first to generate the analysis")
            return None
        except Exception as e:
            print(f"❌ Error generating mapping: {e}")
            return None
    
    def _write_updated_mapper(self, mappings: Dict, pattern_stats: Dict, output_file: str):
        """Write the updated smiles_mapper.py file"""
        
        # Separate known vs pattern-based mappings
        known_mappings = {k: v for k, v in mappings.items() if v['name'] in [name for name, _, _ in self.known_mappings.values()]}
        
        # Create the code
        with open(output_file, 'w') as f:
            f.write('"""\n')
            f.write('SMILES to Fragrance Ingredient Mapper - AUTO-GENERATED\n')
            f.write('Converts SMILES chemical codes to actual fragrance ingredient names\n')
            f.write(f'Generated from dataset analysis of {len(mappings)} unique molecules\n')
            f.write('"""\n\n')
            
            f.write('import logging\n')
            f.write('import re\n')
            f.write('from typing import Dict, Optional, Tuple\n\n')
            f.write('logger = logging.getLogger(__name__)\n\n')
            
            f.write('class SmilesMapper:\n')
            f.write('    """Maps SMILES codes to actual fragrance ingredient names"""\n\n')
            f.write('    def __init__(self):\n')
            
            # Write known mappings
            f.write('        # Known fragrance molecules database (exact matches)\n')
            f.write('        self.smiles_database = {\n')
            for smiles, (name, odor, source) in list(self.known_mappings.items())[:50]:
                name_safe = name.replace("'", "\\'")
                odor_safe = odor.replace("'", "\\'") 
                source_safe = source.replace("'", "\\'")
                f.write(f"            '{smiles}': {{'name': '{name_safe}', 'odor': '{odor_safe}', 'source': '{source_safe}'}},\n")
            f.write('        }\n\n')
            
            # Write pattern rules
            f.write('        # Pattern-based recognition for structural classes\n')
            f.write('        self.structure_patterns = [\n')
            for pattern, compound_type, odor_desc in self.pattern_rules:
                pattern_safe = pattern.replace("'", "\\'").replace("\\", "\\\\")
                type_safe = compound_type.replace("'", "\\'")
                odor_safe = odor_desc.replace("'", "\\'")
                f.write(f"            (r'{pattern_safe}', '{type_safe}', '{odor_safe}', 'Natural or Synthetic'),\n")
            f.write('        ]\n\n')
            
            # Write MW classifications
            f.write('        # MW-based classification for unknowns\n')
            f.write('        self.mw_classifications = [\n')
            f.write('            (50, 100, "Very Light Volatile", "Very Fresh, Sharp, Penetrating"),\n')
            f.write('            (100, 150, "Light Volatile", "Fresh, Citrusy, Green"),\n')
            f.write('            (150, 200, "Medium Volatile", "Floral, Spicy, Balanced"),\n')
            f.write('            (200, 250, "Heavy Volatile", "Woody, Resinous, Warm"),\n')
            f.write('            (250, 350, "Very Heavy", "Base Notes, Fixative, Long-lasting"),\n')
            f.write('            (350, 500, "Ultra Heavy", "Deep Base, Very Long-lasting"),\n')
            f.write('        ]\n\n')
            
            # Write methods
            f.write('''    def get_ingredient_name(self, smiles: str, molecular_weight: Optional[float] = None) -> Tuple[str, str]:
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

''')
            
            # Write dataset statistics
            f.write('# Dataset Statistics (for reference)\n')
            f.write('DATASET_STATS = {\n')
            f.write(f'    "total_unique_molecules": {len(mappings)},\n')
            f.write('    "pattern_breakdown": {\n')
            
            total = sum(pattern_stats.values())
            for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True)[:15]:
                percentage = (count / total) * 100
                pattern_safe = pattern.replace('"', '\\"')
                f.write(f'        "{pattern_safe}": {{"count": {count}, "percentage": {percentage:.1f}}},\n')
            
            f.write('    }\n')
            f.write('}\n')
    
    def _format_known_mappings(self) -> str:
        """Format known mappings for code generation"""
        lines = []
        for smiles, (name, odor, source) in self.known_mappings.items():
            # Escape quotes in strings
            name_safe = name.replace("'", "\\'")
            odor_safe = odor.replace("'", "\\'")
            source_safe = source.replace("'", "\\'")
            
            lines.append(f"            '{smiles}': {{'name': '{name_safe}', 'odor': '{odor_safe}', 'source': '{source_safe}'}},")
        
        return '\n'.join(lines[:50])  # Limit to top 50 to keep file manageable
    
    def _format_pattern_rules(self) -> str:
        """Format pattern rules for code generation"""
        lines = []
        for pattern, compound_type, odor_desc in self.pattern_rules:
            # Escape quotes and backslashes
            pattern_safe = pattern.replace("'", "\\'").replace("\\", "\\\\")
            type_safe = compound_type.replace("'", "\\'")
            odor_safe = odor_desc.replace("'", "\\'")
            
            lines.append(f"            (r'{pattern_safe}', '{type_safe}', '{odor_safe}', 'Natural or Synthetic'),")
        
        return '\n'.join(lines)
    
    def _format_pattern_stats(self, pattern_stats: Dict) -> str:
        """Format pattern statistics for code generation"""
        lines = []
        total = sum(pattern_stats.values())
        
        for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            pattern_safe = pattern.replace("'", "\\'")
            lines.append(f"        '{pattern_safe}': {{'count': {count}, 'percentage': {percentage:.1f}}},")
        
        return '\n'.join(lines[:20])  # Top 20 patterns

def main():
    """Main function to generate SMILES mapping"""
    print("🧪 SMILES MAPPING GENERATOR")
    print("=" * 40)
    
    generator = SmilesMappingGenerator()
    
    # Check if analysis file exists
    analysis_file = "dataset_analysis.csv"
    
    try:
        mappings = generator.generate_mapping_code(analysis_file)
        
        if mappings:
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Generated mappings for {len(mappings)} unique SMILES")
            print(f"📁 Updated mapper saved as 'updated_smiles_mapper.py'")
            print(f"\n💡 NEXT STEPS:")
            print(f"   1. Review the generated file")
            print(f"   2. Replace your current smiles_mapper.py with updated_smiles_mapper.py")
            print(f"   3. Restart your Streamlit app to see ingredient names!")
            
            # Test a few examples
            print(f"\n🧪 TESTING EXAMPLES:")
            test_smiles = list(mappings.keys())[:5]
            for smiles in test_smiles:
                mapping = mappings[smiles]
                print(f"   {smiles[:30]:<30} → {mapping['name']} ({mapping['odor']})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Make sure you've run 'python analyze_dataset.py' first!")

if __name__ == "__main__":
    main()