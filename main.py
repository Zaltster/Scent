# main.py
# Purpose: Command-line interface for the system
# What it does: Provides user-friendly way to run the pipeline
# Responsibilities: Parse arguments, display results, handle user input
# Usage: python main.py "I want to smell like Elsa"

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional

from fragrance_generator import FragranceGenerator
from evaluator import FormulaEvaluator
from config import DEBUG, LOG_LEVEL

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose or DEBUG else getattr(logging, LOG_LEVEL)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from external libraries
    if not verbose:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════╗
║               AI FRAGRANCE GENERATOR                 ║
║          Text-to-Scent Formula Pipeline             ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)

def print_formula_result(result: dict, show_details: bool = False):
    """Print formatted formula results"""
    print("\n" + "="*60)
    print("🌟 FRAGRANCE GENERATION RESULTS")
    print("="*60)
    
    # Input information
    input_info = result.get('input', {})
    print(f"📝 Input: '{input_info.get('text', 'Unknown')}'")
    print(f"🎨 Style: {input_info.get('fragrance_type', 'Unknown').title()}")
    
    # Quality assessment
    quality = result.get('quality_assessment', {})
    overall_score = quality.get('overall_score', 0)
    overall_rating = quality.get('overall_rating', 'Unknown')
    
    print(f"⭐ Quality: {overall_rating} ({overall_score:.1f}/100)")
    
    # Formula display
    formula_display = result.get('formula_display', 'No formula available')
    print(f"\n{formula_display}")
    
    if show_details:
        print_detailed_analysis(result)
    
    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")

def print_detailed_analysis(result: dict):
    """Print detailed analysis of the generation process"""
    print("\n" + "="*60)
    print("📊 DETAILED ANALYSIS")
    print("="*60)
    
    # Text processing analysis
    text_analysis = result.get('text_analysis', {})
    selected_descriptors = text_analysis.get('selected_descriptors', {})
    
    if selected_descriptors:
        print("\n🔤 SELECTED SCENT DESCRIPTORS:")
        for desc, score in list(selected_descriptors.items())[:5]:
            print(f"  • {desc:<15} (similarity: {score:.3f})")
    
    # Molecule selection summary
    mol_selection = result.get('molecule_selection', {})
    summary = mol_selection.get('summary', {})
    
    print(f"\n🧪 MOLECULE SELECTION:")
    print(f"  • Total molecules found: {summary.get('total_selected', 0)}")
    
    note_dist = summary.get('note_distribution', {})
    for note, count in note_dist.items():
        print(f"  • {note.capitalize()} notes: {count}")
    
    # Quality scores
    quality = result.get('quality_assessment', {})
    scores = quality.get('scores', {})
    
    if scores:
        print(f"\n📈 QUALITY SCORES:")
        for metric, score in scores.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"  • {metric_name:<20}: {score:.1f}/100")

def print_alternatives(alternatives: list):
    """Print alternative formulas"""
    if not alternatives:
        return
    
    print("\n" + "="*60)
    print("🔄 ALTERNATIVE FORMULAS")
    print("="*60)
    
    for i, alt_formula in enumerate(alternatives, 1):
        print(f"\n--- Alternative {i}: {alt_formula.get('description', 'Unknown')} ---")
        display = alt_formula.get('formula_display')
        if display:
            # Show abbreviated version
            lines = display.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                print(line)
            if len(lines) > 10:
                print("  ... (truncated)")

def save_results(result: dict, output_file: str):
    """Save results to JSON file"""
    try:
        # Clean result for JSON serialization
        clean_result = clean_for_json(result)
        
        with open(output_file, 'w') as f:
            json.dump(clean_result, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error saving results: {e}")

def clean_for_json(obj):
    """Clean object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif str(type(obj)).startswith('<class \'pandas'):  # pandas objects
        return str(obj)
    else:
        return obj

def interactive_mode():
    """Run in interactive mode"""
    print_banner()
    print("🚀 Interactive Mode - Enter fragrance descriptions or 'quit' to exit\n")
    
    # Initialize generator
    print("⏳ Initializing AI models...")
    try:
        generator = FragranceGenerator()
        generator.initialize()
        print("✅ Ready to generate fragrances!\n")
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return
    
    while True:
        try:
            # Get user input
            user_input = input("🌸 Describe your desired fragrance: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("Please enter a description.")
                continue
            
            # Validate input
            is_valid, error_msg = generator.validate_input(user_input)
            if not is_valid:
                print(f"❌ {error_msg}")
                continue
            
            # Generate fragrance
            print(f"\n⏳ Generating fragrance for: '{user_input}'...")
            
            result = generator.generate_fragrance(
                user_input, 
                include_alternatives=True
            )
            
            # Display results
            print_formula_result(result)
            
            # Show alternatives
            alternatives = result.get('alternatives', [])
            if alternatives:
                show_alt = input(f"\n🔄 Show {len(alternatives)} alternatives? (y/n): ").lower().startswith('y')
                if show_alt:
                    print_alternatives(alternatives)
            
            # Option to save
            save_option = input("\n💾 Save results to file? (y/n): ").lower().startswith('y')
            if save_option:
                filename = f"fragrance_result_{user_input[:20].replace(' ', '_')}.json"
                save_results(result, filename)
            
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different input.\n")

def batch_mode(input_file: str, output_dir: str):
    """Run in batch mode from file"""
    print(f"📂 Batch mode: Processing {input_file}")
    
    try:
        # Read input file
        with open(input_file, 'r') as f:
            inputs = [line.strip() for line in f if line.strip()]
        
        if not inputs:
            print("❌ No valid inputs found in file")
            return
        
        print(f"📝 Found {len(inputs)} fragrance descriptions to process")
        
        # Initialize generator
        print("⏳ Initializing AI models...")
        generator = FragranceGenerator()
        generator.initialize()
        
        # Process batch
        results = generator.generate_batch(inputs)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            filename = f"fragrance_{i+1:03d}.json"
            save_results(result, output_path / filename)
        
        print(f"✅ Batch processing complete. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="AI Fragrance Generator - Convert text descriptions into fragrance formulas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "I want to smell like Elsa"
  python main.py "fresh ocean breeze" --type fresh --details
  python main.py --interactive
  python main.py --batch inputs.txt --output results/
  python main.py "romantic evening" --alternatives --save results.json
        """
    )
    
    # Input arguments
    parser.add_argument(
        'text', 
        nargs='?', 
        help='Text description of desired fragrance'
    )
    
    # Mode arguments
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--batch', '-b',
        metavar='INPUT_FILE',
        help='Run in batch mode with input file (one description per line)'
    )
    
    # Generation options
    parser.add_argument(
        '--type', '-t',
        choices=['balanced', 'fresh', 'oriental', 'floral'],
        default='balanced',
        help='Fragrance type profile (default: balanced)'
    )
    
    parser.add_argument(
        '--alternatives', '-a',
        action='store_true',
        help='Generate alternative formulas'
    )
    
    # Output options
    parser.add_argument(
        '--details', '-d',
        action='store_true',
        help='Show detailed analysis'
    )
    
    parser.add_argument(
        '--save', '-s',
        metavar='OUTPUT_FILE',
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        metavar='OUTPUT_DIR',
        default='./results',
        help='Output directory for batch mode (default: ./results)'
    )
    
    # System options
    parser.add_argument(
        '--dataset',
        metavar='CSV_FILE',
        help='Path to custom dataset CSV file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AI Fragrance Generator v1.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Determine mode
        if args.interactive:
            interactive_mode()
            
        elif args.batch:
            if not Path(args.batch).exists():
                print(f"❌ Input file not found: {args.batch}")
                sys.exit(1)
            batch_mode(args.batch, args.output)
            
        elif args.text:
            # Single fragrance generation
            single_generation_mode(args)
            
        else:
            # No input provided - show help and enter interactive mode
            print("No input provided. Here are your options:\n")
            parser.print_help()
            print("\n" + "="*50)
            
            choice = input("\nWould you like to enter interactive mode? (y/n): ").lower()
            if choice.startswith('y'):
                interactive_mode()
            else:
                print("👋 Goodbye!")
                
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def single_generation_mode(args):
    """Handle single fragrance generation"""
    print_banner()
    print(f"🌸 Generating fragrance for: '{args.text}'")
    print(f"🎨 Style: {args.type.title()}")
    
    try:
        # Initialize generator
        print("\n⏳ Initializing AI models...")
        generator = FragranceGenerator(args.dataset)
        generator.initialize()
        
        # Validate input
        is_valid, error_msg = generator.validate_input(args.text)
        if not is_valid:
            print(f"❌ Invalid input: {error_msg}")
            sys.exit(1)
        
        # Generate fragrance
        print("⏳ Processing...")
        result = generator.generate_fragrance(
            args.text,
            fragrance_type=args.type,
            include_alternatives=args.alternatives
        )
        
        # Display results
        print_formula_result(result, args.details)
        
        # Show alternatives if requested
        if args.alternatives:
            alternatives = result.get('alternatives', [])
            if alternatives:
                print_alternatives(alternatives)
            else:
                print("\n🔄 No alternative formulas generated")
        
        # Save results if requested
        if args.save:
            save_results(result, args.save)
        
        # Show status
        pipeline_status = generator.get_pipeline_status()
        dataset_stats = pipeline_status.get('dataset_stats', {})
        total_molecules = dataset_stats.get('total_molecules', 0)
        
        print(f"\n📊 Pipeline Status: {total_molecules} molecules in dataset")
        print("✅ Generation complete!")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        sys.exit(1)

def validate_environment():
    """Validate that required dependencies are available"""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append('sentence-transformers')
    
    try:
        import rdkit
    except ImportError:
        missing_deps.append('rdkit')
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append('scikit-learn')
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  • {dep}")
        print("\nPlease install missing dependencies:")
        print("pip install pandas numpy sentence-transformers rdkit scikit-learn")
        sys.exit(1)

def check_dataset():
    """Check if dataset file exists"""
    from config import MERGED_DATASET_PATH
    
    if not Path(MERGED_DATASET_PATH).exists():
        print(f"❌ Dataset not found: {MERGED_DATASET_PATH}")
        print("\nPlease ensure the dataset file is in the correct location:")
        print(f"  Expected: {MERGED_DATASET_PATH}")
        print("\nThe dataset should contain molecule-to-scent mappings.")
        return False
    
    return True

def print_system_info():
    """Print system information for debugging"""
    import platform
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")

if __name__ == '__main__':
    # Pre-flight checks
    print("🔍 Checking environment...")
    
    # Validate dependencies
    validate_environment()
    
    # Check dataset
    if not check_dataset():
        sys.exit(1)
    
    print("✅ Environment check passed")
    
    # Run main application
    main()