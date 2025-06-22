# demo.py
# Purpose: Interactive demonstration script
# What it does: Shows off the system with example inputs and detailed output
# Responsibilities: Educational examples, showcase capabilities
# Contains: Multiple example runs with explanations

import time
import logging
from typing import List, Dict
from pathlib import Path

from fragrance_generator import FragranceGenerator
from evaluator import FormulaEvaluator

# Demo examples with expected outcomes
DEMO_EXAMPLES = [
    {
        'input': 'I want to smell like Elsa from Frozen',
        'expected_descriptors': ['fresh', 'clean', 'powdery', 'ethereal', 'cooling'],
        'description': 'Classic character-based input - should yield icy, pristine scents',
        'fragrance_type': 'fresh'
    },
    {
        'input': 'romantic evening with roses',
        'expected_descriptors': ['rose', 'floral', 'romantic', 'sweet', 'warm'],
        'description': 'Mood + specific scent - should create elegant floral formula',
        'fragrance_type': 'floral'
    },
    {
        'input': 'ocean breeze on a summer morning',
        'expected_descriptors': ['fresh', 'ozone', 'marine', 'citrus', 'clean'],
        'description': 'Environmental description - should emphasize aquatic freshness',
        'fragrance_type': 'fresh'
    },
    {
        'input': 'cozy library with vanilla and wood',
        'expected_descriptors': ['vanilla', 'woody', 'warm', 'cozy', 'amber'],
        'description': 'Specific ingredients mentioned - should honor those requests',
        'fragrance_type': 'oriental'
    },
    {
        'input': 'mysterious and seductive',
        'expected_descriptors': ['musk', 'amber', 'spicy', 'dark', 'sensual'],
        'description': 'Abstract mood - should interpret into concrete scent qualities',
        'fragrance_type': 'oriental'
    }
]

class FragranceDemo:
    """
    Interactive demonstration of the fragrance generation pipeline
    """
    
    def __init__(self):
        """Initialize demo"""
        self.generator = None
        self.evaluator = FormulaEvaluator()
        
        # Setup minimal logging for demo
        logging.basicConfig(level=logging.WARNING)
        
    def run_full_demo(self):
        """Run complete demonstration with all examples"""
        self.print_demo_header()
        
        print("🚀 Initializing AI Fragrance Generator...")
        try:
            self.generator = FragranceGenerator()
            self.generator.initialize()
            print("✅ System ready!\n")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("Please check that all dependencies are installed and dataset is available.")
            return
        
        # Show pipeline status
        self.show_pipeline_status()
        
        # Run example demonstrations
        print("\n" + "="*70)
        print("🎭 DEMONSTRATION EXAMPLES")
        print("="*70)
        
        for i, example in enumerate(DEMO_EXAMPLES, 1):
            print(f"\n--- Example {i}/{len(DEMO_EXAMPLES)} ---")
            self.run_single_example(example)
            
            if i < len(DEMO_EXAMPLES):
                input("\nPress Enter to continue to next example...")
        
        # Interactive section
        print("\n" + "="*70)
        print("🎮 YOUR TURN - Try Your Own Inputs!")
        print("="*70)
        self.interactive_section()
        
        print("\n🎉 Demo complete! Thank you for trying the AI Fragrance Generator!")
    
    def print_demo_header(self):
        """Print demo introduction"""
        header = """
╔══════════════════════════════════════════════════════════════════════╗
║                    AI FRAGRANCE GENERATOR DEMO                      ║
║                                                                      ║
║  This demo showcases how AI can convert text descriptions into       ║
║  actual fragrance formulas using molecular-level scent data.        ║
║                                                                      ║
║  The system:                                                         ║
║  1. Uses embeddings to map text → scent descriptors                 ║
║  2. Finds molecules matching those descriptors                       ║
║  3. Classifies molecules by volatility (top/middle/base notes)       ║
║  4. Composes balanced formulas with realistic proportions            ║
╚══════════════════════════════════════════════════════════════════════╝
        """
        print(header)
    
    def show_pipeline_status(self):
        """Show system status and capabilities"""
        status = self.generator.get_pipeline_status()
        
        print("📊 SYSTEM STATUS:")
        print(f"  • Pipeline initialized: {status['initialized']}")
        print(f"  • Dataset loaded: {status['dataset_loaded']}")
        
        if 'dataset_stats' in status:
            stats = status['dataset_stats']
            print(f"  • Total molecules: {stats.get('total_molecules', 0):,}")
            print(f"  • Available descriptors: {stats.get('total_descriptors', 0)}")
            
            # Show most common descriptors
            most_common = stats.get('most_common_descriptors', {})
            if most_common:
                top_5 = list(most_common.items())[:5]
                descriptor_list = ', '.join([desc for desc, count in top_5])
                print(f"  • Top descriptors: {descriptor_list}")
        
        if 'embedding_model' in status:
            model_info = status['embedding_model']
            if model_info.get('model_loaded'):
                print(f"  • Embedding model: {model_info.get('model_name', 'Unknown')}")
                print(f"  • Embedding dimension: {model_info.get('embedding_dimension', 0)}")
    
    def run_single_example(self, example: Dict):
        """Run a single demo example with detailed explanation"""
        input_text = example['input']
        description = example['description']
        fragrance_type = example['fragrance_type']
        expected_descriptors = example['expected_descriptors']
        
        print(f"📝 Input: \"{input_text}\"")
        print(f"💡 Expected: {description}")
        print(f"🎨 Style: {fragrance_type.title()}")
        print()
        
        try:
            # Generate fragrance with timing
            start_time = time.time()
            
            result = self.generator.generate_fragrance(
                input_text,
                fragrance_type=fragrance_type,
                include_alternatives=False
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Show results
            self.display_example_results(result, expected_descriptors, generation_time)
            
        except Exception as e:
            print(f"❌ Error processing example: {e}")
    
    def display_example_results(self, result: Dict, expected_descriptors: List[str], 
                               generation_time: float):
        """Display results from an example with analysis"""
        
        # Show selected descriptors
        text_analysis = result.get('text_analysis', {})
        selected_descriptors = text_analysis.get('selected_descriptors', {})
        
        print(f"⏱️  Generation time: {generation_time:.2f} seconds")
        print()
        
        print("🔤 SELECTED DESCRIPTORS:")
        if selected_descriptors:
            for desc, score in list(selected_descriptors.items())[:5]:
                # Check if this was expected
                expected_marker = " ✓" if desc in expected_descriptors else ""
                print(f"  • {desc:<15} (similarity: {score:.3f}){expected_marker}")
        else:
            print("  No descriptors selected")
        
        # Check expectation match
        matched_expected = set(selected_descriptors.keys()) & set(expected_descriptors)
        if matched_expected:
            print(f"  ✅ Matched expectations: {', '.join(matched_expected)}")
        
        # Show formula summary
        formula = result.get('formula', {})
        components = formula.get('components', [])
        
        print(f"\n🧪 FORMULA SUMMARY:")
        print(f"  • Components: {len(components)}")
        print(f"  • Total percentage: {formula.get('total_percentage', 0):.1f}%")
        
        # Note distribution
        note_dist = formula.get('metadata', {}).get('note_distribution', {}).get('percentages', {})
        if note_dist:
            print(f"  • Top notes: {note_dist.get('top', 0):.1f}%")
            print(f"  • Middle notes: {note_dist.get('middle', 0):.1f}%")
            print(f"  • Base notes: {note_dist.get('base', 0):.1f}%")
        
        # Quality assessment
        quality = result.get('quality_assessment', {})
        overall_rating = quality.get('overall_rating', 'Unknown')
        overall_score = quality.get('overall_score', 0)
        
        print(f"\n⭐ QUALITY: {overall_rating} ({overall_score:.1f}/100)")
        
        # Show top 3 components
        if components:
            print(f"\n🏆 TOP COMPONENTS:")
            for i, component in enumerate(components[:3], 1):
                smiles = component['smiles']
                percentage = component['percentage']
                note = component['note_position']
                descriptors = ', '.join(component.get('matched_descriptors', [])[:2])
                print(f"  {i}. {percentage:5.1f}% - {smiles} ({note}) [{descriptors}]")
        
        # Show any warnings
        warnings = formula.get('metadata', {}).get('warnings', [])
        if warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in warnings[:2]:
                print(f"  • {warning}")
    
    def interactive_section(self):
        """Interactive section for user input"""
        print("Now you can try your own fragrance descriptions!")
        print("Type 'quit' to exit the demo.\n")
        
        while True:
            try:
                user_input = input("🌸 Describe your desired fragrance: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    print("Please enter a description.")
                    continue
                
                # Validate input
                is_valid, error_msg = self.generator.validate_input(user_input)
                if not is_valid:
                    print(f"❌ {error_msg}")
                    continue
                
                # Ask for fragrance type
                print("\nFragrance types: 1=Fresh, 2=Floral, 3=Oriental, 4=Balanced")
                type_choice = input("Choose type (1-4, default=4): ").strip()
                
                type_map = {'1': 'fresh', '2': 'floral', '3': 'oriental', '4': 'balanced'}
                fragrance_type = type_map.get(type_choice, 'balanced')
                
                print(f"\n⏳ Generating {fragrance_type} fragrance...")
                
                # Generate fragrance
                result = self.generator.generate_fragrance(
                    user_input,
                    fragrance_type=fragrance_type
                )
                
                # Show simplified results
                self.show_interactive_results(result)
                
                print("\n" + "-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting demo...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try a different description.\n")
    
    def show_interactive_results(self, result: Dict):
        """Show simplified results for interactive mode"""
        # Formula display
        formula_display = result.get('formula_display', 'No formula available')
        print(f"\n{formula_display}")
        
        # Quality
        quality = result.get('quality_assessment', {})
        rating = quality.get('overall_rating', 'Unknown')
        score = quality.get('overall_score', 0)
        print(f"\n⭐ Quality: {rating} ({score:.1f}/100)")
        
        # Quick recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Tip: {recommendations[0]}")
    
    def run_quick_demo(self):
        """Run a quick demo with just one example"""
        self.print_demo_header()
        
        print("🚀 Quick Demo - Initializing...")
        try:
            self.generator = FragranceGenerator()
            self.generator.initialize()
            print("✅ Ready!\n")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return
        
        # Run one example
        example = DEMO_EXAMPLES[0]  # Elsa example
        print("📝 Example: Creating a fragrance inspired by Elsa from Frozen")
        print("Expected: Fresh, icy, pristine scents\n")
        
        self.run_single_example(example)
        
        print("\n🎉 Quick demo complete!")
        print("Run 'python demo.py --full' for the complete demonstration.")

def main():
    """Main demo entry point"""
    import sys
    
    demo = FragranceDemo()
    
    if '--full' in sys.argv:
        demo.run_full_demo()
    elif '--quick' in sys.argv:
        demo.run_quick_demo()
    else:
        print("AI Fragrance Generator Demo")
        print("\nOptions:")
        print("  python demo.py --full   # Complete demonstration")
        print("  python demo.py --quick  # Quick single example")
        print()
        
        choice = input("Choose demo mode (full/quick): ").lower()
        
        if choice.startswith('f'):
            demo.run_full_demo()
        elif choice.startswith('q'):
            demo.run_quick_demo()
        else:
            print("Running quick demo by default...")
            demo.run_quick_demo()

if __name__ == '__main__':
    main()