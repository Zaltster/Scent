#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes the fragrance dataset to understand SMILES codes and build better mappings
"""

import pandas as pd
import numpy as np
from collections import Counter
import logging
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dataset(dataset_path="data/curated_GS_LF_merged_4983.csv"):
    """Analyze the fragrance dataset to understand SMILES distribution"""
    
    print("🔍 FRAGRANCE DATASET ANALYSIS")
    print("=" * 50)
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        print(f"✅ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Show basic info
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total molecules: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        
        # Identify SMILES column
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower() or 'smi' in col.lower():
                smiles_col = col
                break
        
        if not smiles_col:
            # Try first column if no obvious SMILES column
            smiles_col = df.columns[0]
            print(f"⚠️  No obvious SMILES column found, using first column: '{smiles_col}'")
        else:
            print(f"✅ Found SMILES column: '{smiles_col}'")
        
        # Analyze SMILES codes
        smiles_data = df[smiles_col].dropna()
        print(f"\n🧪 SMILES ANALYSIS:")
        print(f"   Valid SMILES: {len(smiles_data):,}")
        print(f"   Unique SMILES: {smiles_data.nunique():,}")
        print(f"   Duplicates: {len(smiles_data) - smiles_data.nunique():,}")
        
        # Show length distribution
        lengths = smiles_data.str.len()
        print(f"\n📏 SMILES LENGTH DISTRIBUTION:")
        print(f"   Min length: {lengths.min()}")
        print(f"   Max length: {lengths.max()}")
        print(f"   Average length: {lengths.mean():.1f}")
        print(f"   Median length: {lengths.median():.1f}")
        
        # Show most common SMILES
        print(f"\n🔥 TOP 20 MOST COMMON SMILES:")
        smiles_counts = smiles_data.value_counts().head(20)
        for i, (smiles, count) in enumerate(smiles_counts.items(), 1):
            display_smiles = smiles[:50] + "..." if len(smiles) > 50 else smiles
            print(f"   {i:2d}. {display_smiles:<53} (appears {count} times)")
        
        # Show random sample of unique SMILES
        print(f"\n🎲 RANDOM SAMPLE OF UNIQUE SMILES:")
        unique_smiles = smiles_data.unique()
        np.random.shuffle(unique_smiles)
        sample_size = min(15, len(unique_smiles))
        
        for i, smiles in enumerate(unique_smiles[:sample_size], 1):
            display_smiles = smiles[:60] + "..." if len(smiles) > 60 else smiles
            print(f"   {i:2d}. {display_smiles}")
        
        # Analyze molecular patterns
        print(f"\n🔬 MOLECULAR PATTERN ANALYSIS:")
        patterns = {
            'Aromatics (benzene rings)': smiles_data.str.contains('c1ccccc1|C1=CC=CC=C1', case=False, na=False).sum(),
            'Alcohols (-OH)': smiles_data.str.contains('O(?![=C])', case=False, na=False).sum(),
            'Aldehydes (=O)': smiles_data.str.contains('C=O', case=False, na=False).sum(),
            'Esters (COO)': smiles_data.str.contains('C\(=O\)O', case=False, na=False).sum(),
            'Terpenes (complex chains)': smiles_data.str.contains('CC\(C\)=C|C=C.*C=C', case=False, na=False).sum(),
            'Sulfur compounds': smiles_data.str.contains('S', case=False, na=False).sum(),
            'Nitrogen compounds': smiles_data.str.contains('N', case=False, na=False).sum(),
            'Simple chains (C only)': smiles_data.str.contains('^C+$', case=False, na=False).sum(),
        }
        
        for pattern, count in patterns.items():
            percentage = (count / len(smiles_data)) * 100
            print(f"   {pattern:<25}: {count:,} ({percentage:.1f}%)")
        
        # Analyze descriptor columns
        print(f"\n🏷️  DESCRIPTOR ANALYSIS:")
        descriptor_cols = [col for col in df.columns if col != smiles_col]
        
        if descriptor_cols:
            print(f"   Found {len(descriptor_cols)} descriptor columns:")
            
            # Show first few descriptor columns
            for col in descriptor_cols[:10]:
                if df[col].dtype in ['int64', 'float64']:
                    non_zero = (df[col] != 0).sum()
                    print(f"   - {col:<20}: {non_zero:,} non-zero values")
                else:
                    unique_vals = df[col].nunique()
                    print(f"   - {col:<20}: {unique_vals:,} unique values")
            
            if len(descriptor_cols) > 10:
                print(f"   ... and {len(descriptor_cols) - 10} more columns")
        
        # Look for specific common fragrance descriptors
        common_descriptors = ['rose', 'floral', 'citrus', 'woody', 'fresh', 'sweet', 'vanilla', 'lavender', 'lemon', 'sandalwood']
        found_descriptors = []
        
        for desc in common_descriptors:
            for col in df.columns:
                if desc.lower() in col.lower():
                    found_descriptors.append(col)
                    break
        
        if found_descriptors:
            print(f"\n🌸 FOUND COMMON FRAGRANCE DESCRIPTORS:")
            for desc in found_descriptors[:10]:
                if df[desc].dtype in ['int64', 'float64']:
                    active_count = (df[desc] > 0).sum()
                    print(f"   - {desc:<20}: {active_count:,} molecules have this descriptor")
        
        # Create mapping suggestions
        print(f"\n💡 MAPPING SUGGESTIONS:")
        print("Based on the analysis, here are SMILES codes that should be added to the mapping:")
        
        # Get top unique SMILES to suggest for mapping
        top_unique = smiles_data.value_counts().head(30)
        print("\n📝 TOP CANDIDATES FOR MANUAL MAPPING:")
        for i, (smiles, count) in enumerate(top_unique.items(), 1):
            if len(smiles) < 50:  # Reasonable length for manual mapping
                print(f"   '{smiles}': 'Unknown Ingredient (appears {count} times)',")
        
        # Export for further analysis
        output_file = "dataset_analysis.csv"
        analysis_df = pd.DataFrame({
            'smiles': smiles_data.value_counts().index,
            'frequency': smiles_data.value_counts().values,
            'length': [len(s) for s in smiles_data.value_counts().index]
        })
        analysis_df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed analysis saved to: {output_file}")
        
        return df, smiles_col, analysis_df
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        print("💡 Available files in data/ directory:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                print(f"   - {file}")
        else:
            print("   No data/ directory found")
        return None, None, None
    
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None, None, None

def suggest_ingredient_mapping(smiles_code):
    """Suggest possible ingredient based on structural analysis"""
    suggestions = []
    
    # Pattern-based suggestions
    if 'c1ccccc1' in smiles_code.lower() or 'C1=CC=CC=C1' in smiles_code:
        suggestions.append("Aromatic compound (floral/sweet)")
    
    if 'CC(C)=C' in smiles_code:
        suggestions.append("Terpene (natural, fresh)")
    
    if 'C=O' in smiles_code and not 'C(=O)O' in smiles_code:
        suggestions.append("Aldehyde (fresh, sharp)")
    
    if 'C(=O)O' in smiles_code:
        suggestions.append("Carboxylic acid (sharp, pungent)")
    
    if 'OC' in smiles_code and not '=O' in smiles_code:
        suggestions.append("Alcohol (sweet, floral)")
    
    if 'S' in smiles_code:
        suggestions.append("Sulfur compound (may have pungent/savory notes)")
    
    if len(smiles_code) > 30:
        suggestions.append("Complex molecule (likely synthetic)")
    elif len(smiles_code) < 15:
        suggestions.append("Simple molecule (likely natural or building block)")
    
    return suggestions

if __name__ == "__main__":
    # Run analysis
    df, smiles_col, analysis = analyze_dataset()
    
    if df is not None:
        print(f"\n🎉 Analysis complete!")
        print(f"💡 Use the output above to improve the SMILES mapping in smiles_mapper.py")
        
        # Optional: Interactive mode to look up specific SMILES
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            print(f"\n🔍 INTERACTIVE MODE")
            print("Enter SMILES codes to get structural suggestions (or 'quit' to exit):")
            
            while True:
                smiles_input = input("\nSMILES code: ").strip()
                if smiles_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if smiles_input:
                    suggestions = suggest_ingredient_mapping(smiles_input)
                    print(f"Suggestions for '{smiles_input}':")
                    for suggestion in suggestions:
                        print(f"  - {suggestion}")
                    
                    # Check if it's in the dataset
                    if smiles_col and smiles_input in df[smiles_col].values:
                        count = (df[smiles_col] == smiles_input).sum()
                        print(f"  💡 This SMILES appears {count} times in your dataset")
    else:
        print("❌ Could not analyze dataset. Please check the file path and try again.")