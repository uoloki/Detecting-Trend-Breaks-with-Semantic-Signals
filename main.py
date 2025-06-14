#!/usr/bin/env python3
"""
S&P 500 Stock Price Prediction System

Main entry point for the time series prediction system with sentiment analysis.
Choose from individual model evaluation, ensemble analysis, or data leakage audit.
"""

import argparse
import os
import sys

def run_individual_models(num_seeds=10):
    """Run individual model analysis."""
    print(f"Running individual model analysis with {num_seeds} random seeds...")
    from run_individual_models import run_model_analysis
    run_model_analysis(num_seeds=num_seeds)

def run_ensemble():
    """Run ensemble analysis."""
    print("Running ensemble analysis...")
    from run_ensemble import run_ensemble_analysis
    run_ensemble_analysis()

def run_audit():
    """Run data leakage audit."""
    print("Running data leakage audit...")
    from audit_data_leakage import run_comprehensive_audit
    success = run_comprehensive_audit()
    if success:
        print("\nAudit completed successfully - methodology verified!")
    else:
        print("\nAudit found issues - please investigate before trusting results!")

def main():
    """Main execution function with CLI interface."""
    
    print("=" * 60)
    print("    S&P 500 STOCK PRICE PREDICTION SYSTEM")
    print("=" * 60)
    print("Deep learning models with sentiment analysis from financial tweets")
    print()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description='S&P 500 Stock Price Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --individual              # Run individual model analysis (10 seeds)
  python main.py --individual --seeds 25   # Run with 25 random seeds
  python main.py --ensemble                # Run ensemble analysis  
  python main.py --audit                   # Run data leakage audit
  python main.py --all --seeds 50          # Run all analyses with 50 seeds
        """
    )
    
    parser.add_argument('--individual', action='store_true',
                       help='Run individual model analysis')
    parser.add_argument('--ensemble', action='store_true',
                       help='Run ensemble analysis')
    parser.add_argument('--audit', action='store_true',
                       help='Run data leakage audit')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    parser.add_argument('--seeds', type=int, default=10, 
                       help='Number of random seeds for individual analysis (1-100, default: 10)')
    
    args = parser.parse_args()
    
    # Validate seeds parameter
    if args.seeds < 1 or args.seeds > 100:
        print("Error: Number of seeds must be between 1 and 100")
        sys.exit(1)
    
    # If no arguments provided, show interactive menu
    if not any([args.individual, args.ensemble, args.audit, args.all]):
        print("Available analyses:")
        print("1. Individual Models - Test CNN-LSTM, BiLSTM, Transformer models")
        print("2. Ensemble Analysis - Run intelligent ensemble system")  
        print("3. Data Leakage Audit - Verify methodology integrity")
        print("4. All - Run complete analysis suite")
        print()
        
        # Get number of seeds for individual analysis
        while True:
            try:
                seeds_input = input(f"Number of random seeds for analysis (1-100, default: 10): ").strip()
                if not seeds_input:
                    num_seeds = 10
                else:
                    num_seeds = int(seeds_input)
                    if num_seeds < 1 or num_seeds > 100:
                        print("Please enter a number between 1 and 100.")
                        continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                choice = input("Select option (1-4): ").strip()
                if choice == '1':
                    args.individual = True
                    args.seeds = num_seeds
                    break
                elif choice == '2':
                    args.ensemble = True
                    break
                elif choice == '3':
                    args.audit = True
                    break
                elif choice == '4':
                    args.all = True
                    args.seeds = num_seeds
                    break
                else:
                    print("Invalid choice. Please enter 1-4.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    
    # Display seed configuration if individual analysis is selected
    if args.individual or args.all:
        print(f"Individual model analysis configured with {args.seeds} random seeds")
        print()
    
    # Run selected analyses
    try:
        if args.all:
            print("\nRunning complete analysis suite...")
            run_audit()
            print("\n" + "="*60)
            run_individual_models(num_seeds=args.seeds)
            print("\n" + "="*60)
            run_ensemble()
            
        else:
            if args.audit:
                run_audit()
            
            if args.individual:
                if args.audit:
                    print("\n" + "="*60)
                run_individual_models(num_seeds=args.seeds)
            
            if args.ensemble:
                if args.audit or args.individual:
                    print("\n" + "="*60)
                run_ensemble()
        
        print("\nAnalysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()