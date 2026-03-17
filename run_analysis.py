#!/usr/bin/env python3
"""
Quick Start Script for Instagram Analytics
==========================================

This script provides a simple way to run the Instagram analytics
with different analysis options.

Usage:
    python run_analysis.py --full        # Run complete analysis
    python run_analysis.py --quick       # Run quick summary only
    python run_analysis.py --viz         # Generate visualizations only
    python run_analysis.py --model       # Run ML models only
"""

import argparse
import sys
from instagram_analytics import InstagramAnalytics

def run_quick_analysis(analytics):
    """Run quick summary analysis."""
    print("🚀 Running Quick Analysis...")
    analytics.generate_summary_report()
    print("✅ Quick analysis completed!")

def run_visualization_analysis(analytics):
    """Run visualization-focused analysis."""
    print("📊 Generating Visualizations...")
    analytics.create_reach_analysis_chart()
    analytics.create_engagement_analysis()
    analytics.create_wordcloud_analysis()
    analytics.correlation_analysis()
    print("✅ Visualizations completed!")

def run_model_analysis(analytics):
    """Run machine learning analysis."""
    print("🤖 Running ML Models...")
    analytics.build_prediction_model()
    print("✅ ML analysis completed!")

def run_full_analysis(analytics):
    """Run complete analysis."""
    print("🚀 Running Full Analysis...")
    
    # Summary report
    print("\n📋 Generating Summary Report...")
    analytics.generate_summary_report()
    
    # Visualizations
    print("\n📊 Creating Visualizations...")
    analytics.create_reach_analysis_chart()
    analytics.create_engagement_analysis()
    analytics.create_wordcloud_analysis()
    
    # Correlation analysis
    print("\n🔍 Performing Correlation Analysis...")
    analytics.correlation_analysis()
    
    # Machine learning
    print("\n🤖 Building Prediction Models...")
    analytics.build_prediction_model()
    
    # Performance dashboard
    print("\n📈 Creating Performance Dashboard...")
    analytics.create_performance_dashboard()
    
    print("\n✅ Full analysis completed!")
    print("📋 Check the generated visualizations and insights.")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Instagram Analytics Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --full     # Complete analysis
  python run_analysis.py --quick    # Summary only
  python run_analysis.py --viz      # Visualizations only
  python run_analysis.py --model    # ML models only
        """
    )
    
    parser.add_argument('--full', action='store_true', 
                       help='Run complete analysis (default)')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick summary only')
    parser.add_argument('--viz', action='store_true', 
                       help='Generate visualizations only')
    parser.add_argument('--model', action='store_true', 
                       help='Run ML models only')
    parser.add_argument('--data', default='Instagram data.csv',
                       help='Path to data file (default: Instagram data.csv)')
    
    args = parser.parse_args()
    
    # If no specific option is chosen, run full analysis
    if not any([args.quick, args.viz, args.model]):
        args.full = True
    
    print("📊 Instagram Data Analytics")
    print("=" * 40)
    
    # Initialize analytics
    try:
        analytics = InstagramAnalytics(args.data)
        
        if analytics.df is None:
            print("❌ Failed to load data. Please check the file path.")
            sys.exit(1)
        
        # Run selected analysis
        if args.quick:
            run_quick_analysis(analytics)
        elif args.viz:
            run_visualization_analysis(analytics)
        elif args.model:
            run_model_analysis(analytics)
        elif args.full:
            run_full_analysis(analytics)
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()