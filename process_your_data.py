#!/usr/bin/env python3
"""
Clean SMILES Processing Script
Analyzes your SMILES data and creates updated mapping files
"""

import pandas as pd
import json
from pathlib import Path
import io

def process_smiles_data():
    """Process SMILES data and create enhanced mapper"""
    
    print("🔍 Processing SMILES data...")
    
    # Read your paste.txt file
    try:
        if Path("paste.txt").exists():
            df = pd.read_csv("paste.txt")
            print(f"✅ Loaded {len(df)} molecules from paste.txt")
        else:
            # Use sample data if file not found
            sample_data = """smiles,frequency,length
COc1cc(C=O)ccc1O,1,16
CC(C)=CCCC(C)=CCO,1,20
CCCCCCCCCC=O,1,12
COc1ccc(CCO)cc1,1,14
C=CCc1ccc(O)cc1,1,15
CC(C)c1ccc(O)cc1,1,16
Oc1ccccc1,1,9
CCO,1,3
CO,1,2
O,1,1"""
            df = pd.read_csv(io.StringIO(sample_data))
            print(f"⚠️  Using sample data: {len(df)} molecules")
    
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return False
    
    # Create enhanced SMILES mapper
    create_enhanced_mapper(df)
    
    # Create simple integration code
    create_integration_code()
    
    # Create summary report
    create_summary_report(df)
    
    print("✅ Processing complete!")
    return True

def create_enhanced_mapper(df):
    """Create enhanced SMILES mapper with known ingredients"""
    
    # Known ingredient mappings
    known_ingredients = {
        "COc1cc(C=O)ccc1O": {
            "name": "Vanillin",
            "odor_description": "Sweet, creamy, vanilla, balsamic",
            "source_type": "synthetic",
            "note_position": "base",
            "commercial_name": "Vanillin Synthetic",
            "cas_number": "121-33-5"
        },
        "CC(C)=CCCC(C)=CCO": {
            "name": "Geraniol",
            "odor_description": "Sweet, rosy, citrusy with floral facets",
            "source_type": "natural",
            "note_position": "top",
            "commercial_name": "Geraniol Natural",
            "cas_number": "106-24-1"
        },
        "CCCCCCCCCC=O": {
            "name": "Decanal",
            "odor_description": "Fresh, aldehydic, citrus peel, slightly fatty",
            "source_type": "synthetic",
            "note_position": "top",
            "commercial_name": "Aldehyde C-10",
            "cas_number": "112-31-2"
        },
        "COc1ccc(CCO)cc1": {
            "name": "Anisyl Alcohol",
            "odor_description": "Sweet, floral, slightly spicy",
            "source_type": "synthetic",
            "note_position": "middle",
            "commercial_name": "Anisyl Alcohol",
            "cas_number": "105-13-5"
        },
        "C=CCc1ccc(O)cc1": {
            "name": "Chavicol",
            "odor_description": "Spicy, clove-like, phenolic",
            "source_type": "natural",
            "note_position": "middle",
            "commercial_name": "Chavicol Natural",
            "cas_number": "501-92-8"
        },
        "CC(C)c1ccc(O)cc1": {
            "name": "Thymol",
            "odor_description": "Medicinal, thyme-like, phenolic",
            "source_type": "natural",
            "note_position": "top",
            "commercial_name": "Thymol Natural",
            "cas_number": "89-83-8"
        },
        "Oc1ccccc1": {
            "name": "Phenol",
            "odor_description": "Medicinal, phenolic, sharp",
            "source_type": "synthetic",
            "note_position": "base",
            "commercial_name": "Phenol Technical",
            "cas_number": "108-95-2"
        },
        "CCO": {
            "name": "Ethanol",
            "odor_description": "Alcoholic, slightly sweet",
            "source_type": "synthetic",
            "note_position": "top",
            "commercial_name": "Perfumer's Alcohol",
            "cas_number": "64-17-5"
        },
        "CO": {
            "name": "Methanol",
            "odor_description": "Alcoholic, pungent",
            "source_type": "synthetic",
            "note_position": "top",
            "commercial_name": "Methanol Technical",
            "cas_number": "67-56-1"
        },
        "O": {
            "name": "Water",
            "odor_description": "Odorless",
            "source_type": "natural",
            "note_position": "base",
            "commercial_name": "Distilled Water",
            "cas_number": "7732-18-5"
        }
    }
    
    # Create the enhanced mapper file
    mapper_code = '''"""
Enhanced SMILES to Ingredient Mapper
Auto-generated from SMILES data analysis
"""

def map_smiles_to_ingredient(smiles, molecular_weight=None):
    """Map SMILES string to fragrance ingredient information"""
    
    # Known ingredients database
    known_ingredients = {'''
    
    # Add known ingredients to the code
    for smiles, info in known_ingredients.items():
        mapper_code += f'''
        "{smiles}": {{
            "name": "{info['name']}",
            "odor_description": "{info['odor_description']}",
            "source_type": "{info['source_type']}",
            "note_position": "{info['note_position']}",
            "commercial_name": "{info['commercial_name']}",
            "cas_number": "{info['cas_number']}",
            "found_in_database": True
        }},'''
    
    mapper_code += '''
    }
    
    # Clean SMILES input
    clean_smiles = smiles.strip()
    
    # Direct lookup
    if clean_smiles in known_ingredients:
        return known_ingredients[clean_smiles]
    
    # Pattern-based analysis for unknown SMILES
    return analyze_unknown_smiles(clean_smiles, molecular_weight)

def analyze_unknown_smiles(smiles, molecular_weight=None):
    """Analyze unknown SMILES using patterns"""
    
    result = {
        "name": "Unknown Compound",
        "odor_description": "Unknown odor profile",
        "source_type": "unknown",
        "note_position": "middle",
        "commercial_name": None,
        "cas_number": "Not available",
        "found_in_database": False
    }
    
    # Simple pattern detection
    if "=O" in smiles and "C(=O)O" not in smiles:
        result["name"] = "Aldehyde Compound"
        result["odor_description"] = "Likely aldehydic character"
        result["note_position"] = "top"
    elif "CO" in smiles and "C=O" not in smiles:
        result["name"] = "Alcohol Compound"
        result["odor_description"] = "Likely alcoholic character"
        result["note_position"] = "middle"
    elif "c1cc" in smiles:
        result["name"] = "Aromatic Compound"
        result["odor_description"] = "Likely aromatic character"
        result["note_position"] = "middle"
    
    # Molecular weight classification
    if molecular_weight:
        if molecular_weight < 150:
            result["note_position"] = "top"
            result["odor_description"] += " (volatile)"
        elif molecular_weight > 250:
            result["note_position"] = "base"
            result["odor_description"] += " (heavy)"
    
    return result

# Statistics
DATASET_STATS = {
    "total_molecules": ''' + str(len(df)) + ''',
    "known_molecules": ''' + str(sum(1 for smiles in df['smiles'] if smiles in known_ingredients)) + ''',
    "coverage_percentage": ''' + str(round((sum(1 for smiles in df['smiles'] if smiles in known_ingredients) / len(df)) * 100, 1)) + '''
}
'''
    
    # Save the mapper file
    with open("smiles_mapper_enhanced.py", "w") as f:
        f.write(mapper_code)
    
    print("✅ Created smiles_mapper_enhanced.py")

def create_integration_code():
    """Create simple integration code for Streamlit"""
    
    integration_code = '''"""
Simple integration for your Streamlit app
Replace your ingredient display function with this
"""

import streamlit as st
from smiles_mapper_enhanced import map_smiles_to_ingredient

def display_enhanced_ingredient(component):
    """Display ingredient with professional information"""
    
    smiles = component.get('smiles', '')
    percentage = component.get('percentage', 0)
    
    # Get ingredient info
    ingredient_info = map_smiles_to_ingredient(smiles)
    
    # Simple, clean display using Streamlit components
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Main ingredient info
            st.markdown(f"**{ingredient_info['name']}** ({percentage:.2f}%)")
            st.caption(f"🌿 {ingredient_info['odor_description']}")
            
            # Tags
            tags = []
            tags.append(f"{ingredient_info['note_position'].title()} Note")
            tags.append(ingredient_info['source_type'].title())
            if ingredient_info.get('commercial_name'):
                tags.append(ingredient_info['commercial_name'])
            
            st.markdown(" • ".join(tags))
            
            # CAS number if available
            if ingredient_info.get('cas_number') != 'Not available':
                st.caption(f"CAS: {ingredient_info['cas_number']}")
        
        with col2:
            # SMILES in expandable section
            with st.expander("SMILES"):
                st.code(smiles)

# Usage: Replace your existing display function with display_enhanced_ingredient()
'''
    
    with open("streamlit_integration.py", "w") as f:
        f.write(integration_code)
    
    print("✅ Created streamlit_integration.py")

def create_summary_report(df):
    """Create a summary report"""
    
    # Analyze the dataset
    total_molecules = len(df)
    avg_length = df['length'].mean() if 'length' in df.columns else 0
    max_freq = df['frequency'].max() if 'frequency' in df.columns else 0
    
    # Count known vs unknown
    known_smiles = [
        "COc1cc(C=O)ccc1O", "CC(C)=CCCC(C)=CCO", "CCCCCCCCCC=O",
        "COc1ccc(CCO)cc1", "C=CCc1ccc(O)cc1", "CC(C)c1ccc(O)cc1",
        "Oc1ccccc1", "CCO", "CO", "O"
    ]
    
    known_count = sum(1 for smiles in df['smiles'] if smiles in known_smiles)
    unknown_count = total_molecules - known_count
    
    report = f"""# SMILES Processing Report

## Dataset Overview
- **Total Molecules**: {total_molecules:,}
- **Average SMILES Length**: {avg_length:.1f} characters
- **Maximum Frequency**: {max_freq}

## Mapping Results
- **Known Ingredients**: {known_count} ({(known_count/total_molecules)*100:.1f}%)
- **Unknown Molecules**: {unknown_count} ({(unknown_count/total_molecules)*100:.1f}%)

## Files Created
1. `smiles_mapper_enhanced.py` - Enhanced mapping function
2. `streamlit_integration.py` - Integration code for your app
3. `processing_report.md` - This report

## Next Steps
1. Replace your existing `smiles_mapper.py` with `smiles_mapper_enhanced.py`
2. Update your app.py with the code from `streamlit_integration.py`
3. Test your enhanced fragrance generator

## Benefits
- Real ingredient names instead of "Unknown Compound"
- Professional odor descriptions
- Source type information (natural/synthetic)
- Note position classification (top/middle/base)
- Commercial names and CAS numbers

Your fragrance generator will now show names like:
- "Vanillin" instead of "Unknown Compound"
- "Geraniol" instead of "Alcohol Compound"
- "Decanal" instead of "Aldehyde Compound"
"""
    
    with open("processing_report.md", "w") as f:
        f.write(report)
    
    print("✅ Created processing_report.md")

def main():
    """Main function"""
    print("🌸 SMILES Processing Script")
    print("=" * 50)
    
    success = process_smiles_data()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 SUCCESS!")
        print("=" * 50)
        print("Files created:")
        print("✅ smiles_mapper_enhanced.py")
        print("✅ streamlit_integration.py") 
        print("✅ processing_report.md")
        print("\nYour fragrance generator is now enhanced!")
        print("Check processing_report.md for next steps.")
    else:
        print("\n❌ Processing failed. Check error messages above.")

if __name__ == "__main__":
    main()