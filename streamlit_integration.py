"""
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
