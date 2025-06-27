import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys
from pathlib import Path

# Import your fragrance generator
from fragrance_generator import FragranceGenerator
try:
    from smiles_mapper import map_smiles_to_ingredient
    SMILES_MAPPING_AVAILABLE = True
except ImportError:
    SMILES_MAPPING_AVAILABLE = False
    print("⚠️ SMILES mapping not available - showing SMILES codes only")

# Page configuration
st.set_page_config(
    page_title="AI Fragrance Generator",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .formula-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .component-item {
        background: white;
        border-radius: 8px;
        padding: 12px;
        margin: 5px 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    .descriptor-tag {
        background: linear-gradient(135deg, #ddd6fe, #c7d2fe);
        color: #4c1d95;
        padding: 4px 8px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
        display: inline-block;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
    st.session_state.initialized = False
    st.session_state.generation_history = []

@st.cache_resource
def initialize_generator():
    """Initialize the fragrance generator (cached)"""
    try:
        generator = FragranceGenerator()
        generator.initialize()
        return generator, None
    except Exception as e:
        return None, str(e)

def create_note_distribution_chart(note_percentages):
    """Create a donut chart for note distribution"""
    labels = ['Top Notes', 'Middle Notes', 'Base Notes']
    values = [
        note_percentages.get('top', 0),
        note_percentages.get('middle', 0),
        note_percentages.get('base', 0)
    ]
    colors = ['#60a5fa', '#34d399', '#f472b6']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.6,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12,
        textfont_color='white'
    )])
    
    fig.update_layout(
        title={
            'text': "Note Distribution",
            'x': 0.5,
            'font': {'size': 16, 'color': '#1f2937'}
        },
        showlegend=False,
        height=300,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    
    return fig

def create_component_chart(components):
    """Create a horizontal bar chart for components"""
    if not components:
        return None
    
    # Take top 10 components
    top_components = components[:10]
    
    df = pd.DataFrame(top_components)
    df['display_name'] = df.apply(lambda x: f"{x['smiles'][:15]}..." if len(x['smiles']) > 15 else x['smiles'], axis=1)
    
    # Color by note position
    color_map = {'top': '#60a5fa', 'middle': '#34d399', 'base': '#f472b6'}
    df['color'] = df['note_position'].map(color_map)
    
    fig = px.bar(
        df, 
        x='percentage', 
        y='display_name',
        color='note_position',
        color_discrete_map=color_map,
        orientation='h',
        title="Top 10 Components"
    )
    
    fig.update_layout(
        xaxis_title="Percentage (%)",
        yaxis_title="",
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=True,
        legend_title="Note Position"
    )
    
    return fig

def display_formula_results(result, unique_id="main"):
    """Display comprehensive formula results"""
    formula = result.get('formula', {})
    components = formula.get('components', [])
    metadata = formula.get('metadata', {})
    quality = result.get('quality_assessment', {})
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3b82f6; margin: 0;">Components</h3>
            <h2 style="margin: 5px 0; color: #1f2937;">{len(components)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_pct = formula.get('total_percentage', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #10b981; margin: 0;">Total %</h3>
            <h2 style="margin: 5px 0; color: #1f2937;">{total_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        quality_score = quality.get('overall_score', 0)
        quality_rating = quality.get('overall_rating', 'Unknown')
        color = '#10b981' if quality_score >= 70 else '#f59e0b' if quality_score >= 50 else '#ef4444'
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}; margin: 0;">Quality</h3>
            <h2 style="margin: 5px 0;">{quality_score:.1f}/100</h2>
            <p style="margin: 0; color: {color};">{quality_rating}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        descriptors_count = metadata.get('descriptor_coverage', {}).get('total_unique_descriptors', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #8b5cf6; margin: 0;">Descriptors</h3>
            <h2 style="margin: 5px 0; color: #1f2937;">{descriptors_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Note distribution chart
        note_percentages = metadata.get('note_distribution', {}).get('percentages', {})
        if note_percentages:
            fig = create_note_distribution_chart(note_percentages)
            st.plotly_chart(fig, use_container_width=True, key=f"note_dist_{unique_id}")
    
    with col2:
        # Component chart
        if components:
            fig = create_component_chart(components)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"components_{unique_id}")
    
    # Detailed formula breakdown
    st.markdown("### 📋 Detailed Formula")
    st.markdown("""
    <div style="background: #fef3c7; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #f59e0b;">
    🧪 <strong>How to use this for real perfume:</strong><br>
    • <strong>Ingredient Names</strong> = Actual fragrance materials you can buy<br>
    • <strong>Odor Descriptions</strong> = How each ingredient smells (🌿 = natural character)<br>
    • <strong>Source Type</strong> = Natural (essential oils) or Synthetic availability<br>
    • <strong>Percentages</strong> = Exact blend ratios for your formula<br>
    • <strong>Commercial Names</strong> = 💼 Where available from suppliers<br>
    • <strong>SMILES</strong> = Click to see chemical structure (for advanced users)
    </div>
    """, unsafe_allow_html=True)
    
    # Group components by note position
    notes_data = {'top': [], 'middle': [], 'base': []}
    for component in components:
        note = component.get('note_position', 'unknown')
        if note in notes_data:
            notes_data[note].append(component)
    
    # Display each note section
    note_colors = {'top': '#dbeafe', 'middle': '#d1fae5', 'base': '#fce7f3'}
    note_icons = {'top': '🌪️', 'middle': '🌸', 'base': '🌰'}
    
    for note_position, note_components in notes_data.items():
        if note_components:
            note_total = sum(c['percentage'] for c in note_components)
            
            st.markdown(f"""
            <div style="background: {note_colors[note_position]}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                <h4 style="margin: 0; color: #1f2937;">{note_icons[note_position]} {note_position.upper()} NOTES ({note_total:.1f}%)</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for component in sorted(note_components, key=lambda x: x['percentage'], reverse=True):
                smiles = component['smiles']
                percentage = component['percentage']
                mw = component.get('molecular_weight', 0)
                descriptors = component.get('matched_descriptors', [])
                
                # Map SMILES to actual ingredient if mapping is available
                if SMILES_MAPPING_AVAILABLE:
                    try:
                        ingredient_info = map_smiles_to_ingredient(smiles, mw)
                        ingredient_name = ingredient_info['name']
                        odor_description = ingredient_info['odor_description']
                        source_type = ingredient_info['source_type']
                        commercial_name = ingredient_info.get('commercial_name')
                        
                        # Create descriptor tags
                        descriptor_tags = ''.join([
                            f'<span class="descriptor-tag">{desc}</span>' 
                            for desc in descriptors[:3]  # Limit to 3 to make room for ingredient info
                        ])
                        
                        # Add commercial name if available
                        commercial_info = f"<br><small style='color: #059669;'>💼 {commercial_name}</small>" if commercial_name else ""
                        
                        st.markdown(f"""
                        <div class="component-item">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div style="flex: 1;">
                                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                        <strong style="color: #1f2937; font-size: 1.1rem;">{percentage:.1f}%</strong>
                                        <span style="color: #6b7280; margin-left: 10px;">MW: {mw:.1f}</span>
                                        <span style="color: #8b5cf6; margin-left: 10px; font-size: 0.8rem;">({source_type})</span>
                                    </div>
                                    <div style="margin-bottom: 5px;">
                                        <strong style="color: #059669; font-size: 1rem;">{ingredient_name}</strong>
                                        <br><em style="color: #6b7280; font-size: 0.9rem;">🌿 {odor_description}</em>
                                        {commercial_info}
                                    </div>
                                    <div style="margin-top: 8px;">
                                        {descriptor_tags}
                                    </div>
                                </div>
                                <div style="font-family: monospace; color: #9ca3af; font-size: 0.7rem; margin-left: 15px; text-align: right;">
                                    <details>
                                        <summary style="cursor: pointer; color: #6b7280;">SMILES</summary>
                                        <code style="font-size: 0.7rem; word-break: break-all;">{smiles}</code>
                                    </details>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        # Fallback to original display if mapping fails
                        descriptor_tags = ''.join([
                            f'<span class="descriptor-tag">{desc}</span>' 
                            for desc in descriptors[:4]
                        ])
                        
                        st.markdown(f"""
                        <div class="component-item">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="color: #1f2937; font-size: 1.1rem;">{percentage:.1f}%</strong>
                                    <span style="color: #6b7280; margin-left: 10px;">MW: {mw:.1f}</span>
                                </div>
                                <div style="font-family: monospace; color: #374151; font-size: 0.9rem;">
                                    {smiles}
                                </div>
                            </div>
                            <div style="margin-top: 8px;">
                                {descriptor_tags}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Original display when mapping is not available
                    descriptor_tags = ''.join([
                        f'<span class="descriptor-tag">{desc}</span>' 
                        for desc in descriptors[:4]
                    ])
                    
                    st.markdown(f"""
                    <div class="component-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #1f2937; font-size: 1.1rem;">{percentage:.1f}%</strong>
                                <span style="color: #6b7280; margin-left: 10px;">MW: {mw:.1f}</span>
                            </div>
                            <div style="font-family: monospace; color: #374151; font-size: 0.9rem;">
                                {smiles}
                            </div>
                        </div>
                        <div style="margin-top: 8px;">
                            {descriptor_tags}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Selected descriptors
    text_analysis = result.get('text_analysis', {})
    selected_descriptors = text_analysis.get('selected_descriptors', {})
    
    if selected_descriptors:
        st.markdown("### 🎯 Selected Scent Descriptors")
        st.markdown("""
        <div style="background: #f0f9ff; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #3b82f6;">
        💡 <strong>What are descriptors?</strong> These are scent categories the AI identified from your text. 
        The numbers (0.544) show similarity scores - higher = better match to your description.
        </div>
        """, unsafe_allow_html=True)
        
        # Sort by similarity score
        sorted_descriptors = sorted(selected_descriptors.items(), key=lambda x: x[1], reverse=True)
        
        descriptor_html = ""
        for desc, score in sorted_descriptors:
            descriptor_html += f'<span class="descriptor-tag">{desc} ({score:.3f})</span> '
        
        st.markdown(f'<div style="padding: 15px;">{descriptor_html}</div>', unsafe_allow_html=True)
    
    # Warnings and recommendations
    warnings = metadata.get('warnings', [])
    recommendations = result.get('recommendations', [])
    
    if warnings or recommendations:
        st.markdown("### ⚠️ Recommendations")
        
        if warnings:
            for warning in warnings[:3]:
                st.warning(f"⚠️ {warning}")
        
        if recommendations:
            for rec in recommendations[:3]:
                st.info(f"💡 {rec}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌸 AI Fragrance Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform words into molecular fragrance formulas using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Fragrance type selection
        fragrance_type = st.selectbox(
            "Fragrance Style",
            options=['balanced', 'fresh', 'floral', 'oriental'],
            help="Choose the overall character of your fragrance"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            include_alternatives = st.checkbox("Generate alternatives", value=False)
            show_detailed_analysis = st.checkbox("Show detailed analysis", value=True)
            
        st.markdown("---")
        
        # System status
        st.markdown("### 📊 System Status")
        
        # Initialize generator if not done
        if not st.session_state.initialized:
            with st.spinner("Initializing AI models..."):
                generator, error = initialize_generator()
                
            if generator:
                st.session_state.generator = generator
                st.session_state.initialized = True
                st.success("✅ System ready!")
                
                # Show dataset stats
                status = generator.get_pipeline_status()
                dataset_stats = status.get('dataset_stats', {})
                st.metric("Molecules in database", f"{dataset_stats.get('total_molecules', 0):,}")
                st.metric("Available descriptors", dataset_stats.get('total_descriptors', 0))
            else:
                st.error(f"❌ Initialization failed: {error}")
                st.stop()
        else:
            st.success("✅ System ready!")
    
    # Main interface
    if st.session_state.initialized:
        
        # Input section
        st.markdown("### 💭 Describe Your Desired Fragrance")
        
        # Example prompts
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("🏔️ Elsa from Frozen", use_container_width=True):
                st.session_state.example_input = "I want to smell like Elsa from Frozen"
        
        with example_col2:
            if st.button("🌊 Ocean Breeze", use_container_width=True):
                st.session_state.example_input = "fresh ocean breeze on summer morning"
        
        with example_col3:
            if st.button("🌹 Romantic Evening", use_container_width=True):
                st.session_state.example_input = "romantic evening with roses"
        
        # Text input
        default_text = getattr(st.session_state, 'example_input', '')
        user_input = st.text_area(
            "Enter your description:",
            value=default_text,
            height=100,
            placeholder="e.g., 'mysterious and seductive', 'cozy library with vanilla', 'summer garden party'..."
        )
        
        # Generate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_button = st.button(
                "🧪 Generate Fragrance Formula",
                type="primary",
                use_container_width=True
            )
        
        # Generation logic
        if generate_button and user_input.strip():
            # Validate input
            is_valid, error_msg = st.session_state.generator.validate_input(user_input)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                # Show generation progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Generate fragrance
                    status_text.text("🧠 Processing text...")
                    progress_bar.progress(25)
                    
                    start_time = time.time()
                    
                    result = st.session_state.generator.generate_fragrance(
                        user_input,
                        fragrance_type=fragrance_type,
                        include_alternatives=include_alternatives
                    )
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ Generated in {generation_time:.2f} seconds!")
                    
                    # Clear progress indicators
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store in history
                    st.session_state.generation_history.append({
                        'input': user_input,
                        'type': fragrance_type,
                        'result': result,
                        'timestamp': time.time()
                    })
                    
                    # Display results
                    st.markdown("## 🎉 Your Fragrance Formula")
                    display_formula_results(result, unique_id="current")
                    
                    # Alternatives
                    if include_alternatives:
                        alternatives = result.get('alternatives', [])
                        if alternatives:
                            st.markdown("### 🔄 Alternative Formulas")
                            
                            for i, alt in enumerate(alternatives, 1):
                                with st.expander(f"Alternative {i}: {alt.get('description', 'Variant')}"):
                                    # Create a mock result structure for alternatives
                                    alt_result = {
                                        'formula': alt, 
                                        'quality_assessment': {'overall_score': 0, 'overall_rating': 'N/A'}, 
                                        'text_analysis': {}
                                    }
                                    display_formula_results(alt_result, unique_id=f"alt_{i}")
                    
                    # Export options
                    st.markdown("### 💾 Export Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON export
                        import json
                        formula_json = json.dumps(result, indent=2, default=str)
                        st.download_button(
                            "📥 Download JSON",
                            data=formula_json,
                            file_name=f"fragrance_formula_{int(time.time())}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # Text export
                        formula_text = result.get('formula_display', 'No formula available')
                        st.download_button(
                            "📄 Download Text",
                            data=formula_text,
                            file_name=f"fragrance_formula_{int(time.time())}.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Generation failed: {str(e)}")
                    
                    if show_detailed_analysis:
                        with st.expander("🔍 Error Details"):
                            st.code(str(e))
        
        elif generate_button:
            st.warning("⚠️ Please enter a fragrance description first!")
        
        # Generation history
        if st.session_state.generation_history:
            st.markdown("---")
            st.markdown("### 📚 Generation History")
            
            for i, entry in enumerate(reversed(st.session_state.generation_history[-5:]), 1):
                with st.expander(f"{i}. \"{entry['input'][:50]}...\" ({entry['type']})"):
                    display_formula_results(entry['result'], unique_id=f"history_{i}")

if __name__ == "__main__":
    main()