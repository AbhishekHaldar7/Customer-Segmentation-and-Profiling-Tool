#!/usr/bin/env python3
"""
Consumer Segmentation and Portfolio Tool - Main Execution Script
================================================================

This script runs the complete analysis pipeline:
1. Data Cleaning
2. Customer Segmentation  
3. Portfolio Analysis
4. Generate Reports

Usage: python run_analysis.py
"""

import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        CONSUMER SEGMENTATION & PORTFOLIO TOOL               ║
    ║                                                              ║
    ║        📊 Complete Business Intelligence Pipeline            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

def check_requirements():
    """Check if all required files and packages are available"""
    print("🔍 Checking requirements...")
    
    # Check data file
    if not os.path.exists('data.csv'):
        print("❌ Error: data.csv not found!")
        print("Please ensure the dataset is in the project directory.")
        return False
    
    # Check required modules
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'warnings', 'datetime'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def run_data_cleaning():
    """Execute data cleaning module"""
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)
    
    try:
        # Import and run data cleaning
        print("🧹 Starting data cleaning process...")
        
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # Load and clean data (simplified version of data_cleaning.py logic)
        print("Loading dataset...")
        df = pd.read_csv('data.csv')
        
        print(f"Original dataset shape: {df.shape}")
        
        # Convert date columns
        date_columns = ['Order Date', 'Ship Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle missing values
        critical_cols = ['Customer ID', 'Sales', 'Profit']
        df_clean = df.dropna(subset=critical_cols)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Clean numeric columns
        numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove outliers
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        original_len = len(df_clean)
        df_clean = remove_outliers(df_clean, 'Sales')
        df_clean = remove_outliers(df_clean, 'Profit')
        
        # Create additional features
        df_clean['Order Year'] = df_clean['Order Date'].dt.year
        df_clean['Order Month'] = df_clean['Order Date'].dt.month
        df_clean['Profit Margin'] = (df_clean['Profit'] / df_clean['Sales']) * 100
        
        # Save cleaned data
        df_clean.to_csv('cleaned_data.csv', index=False)
        
        print(f"✅ Data cleaning completed!")
        print(f"   - Original rows: {len(df):,}")
        print(f"   - Cleaned rows: {len(df_clean):,}")
        print(f"   - Removed outliers: {original_len - len(df_clean):,}")
        print(f"   - Unique customers: {df_clean['Customer ID'].nunique():,}")
        print(f"   - Date range: {df_clean['Order Date'].min().date()} to {df_clean['Order Date'].max().date()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data cleaning: {str(e)}")
        return False

def run_customer_segmentation():
    """Execute customer segmentation module"""
    print("\n" + "="*70)
    print("STEP 2: CUSTOMER SEGMENTATION")
    print("="*70)
    
    try:
        from customer_segmentation import CustomerSegmentation
        
        print("👥 Starting customer segmentation...")
        
        # Initialize segmentation
        segmentation = CustomerSegmentation()
        
        # Create customer features
        print("Creating customer features...")
        segmentation.create_customer_features()
        
        # Perform clustering
        print("Performing clustering analysis...")
        segmentation.perform_clustering()
        
        # Create visualizations
        print("Generating visualizations...")
        segmentation.visualize_segments()
        
        # Save results
        print("Saving results...")
        segmentation.save_results()
        
        print("✅ Customer segmentation completed!")
        print(f"   - Customers segmented: {len(segmentation.customer_features):,}")
        print(f"   - Number of segments: {segmentation.customer_features['Segment_Name'].nunique()}")
        print("   - Files generated: customer_segments.csv, segment_summary.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in customer segmentation: {str(e)}")
        return False

def run_portfolio_analysis():
    """Execute portfolio analysis module"""
    print("\n" + "="*70)
    print("STEP 3: PORTFOLIO ANALYSIS")
    print("="*70)
    
    try:
        from portfolio_analysis import PortfolioAnalysis
        
        print("📈 Starting portfolio analysis...")
        
        # Initialize portfolio analysis
        portfolio = PortfolioAnalysis()
        
        # Perform analyses
        print("Analyzing product performance...")
        portfolio.product_performance_analysis()
        
        print("Analyzing regional performance...")
        portfolio.regional_analysis()
        
        print("Analyzing temporal trends...")
        portfolio.temporal_analysis()
        
        print("Analyzing customer segment portfolio...")
        portfolio.customer_segment_portfolio()
        
        print("Creating portfolio matrix...")
        portfolio.create_portfolio_matrix()
        
        # Create visualizations
        print("Generating visualizations...")
        portfolio.visualize_portfolio_analysis()
        
        # Generate insights
        print("Generating business insights...")
        insights = portfolio.generate_insights_report()
        
        # Save results
        print("Saving results...")
        portfolio.save_results()
        
        print("✅ Portfolio analysis completed!")
        print("   - Files generated: Multiple analysis CSV files and visualizations")
        print("   - Business insights: portfolio_insights.json, portfolio_insights_report.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in portfolio analysis: {str(e)}")
        return False

def generate_summary_report():
    """Generate final summary report"""
    print("\n" + "="*70)
    print("STEP 4: GENERATING SUMMARY REPORT")
    print("="*70)
    
    try:
        import pandas as pd
        import json
        
        # Load results
        df_clean = pd.read_csv('cleaned_data.csv')
        customer_segments = pd.read_csv('customer_segments.csv', index_col=0)
        
        # Load insights
        with open('portfolio_insights.json', 'r') as f:
            insights = json.load(f)
        
        # Create summary report
        report = f"""
CONSUMER SEGMENTATION & PORTFOLIO TOOL - EXECUTIVE SUMMARY
{'='*80}

ANALYSIS COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA OVERVIEW:
- Total Transactions: {len(df_clean):,}
- Unique Customers: {df_clean['Customer ID'].nunique():,}
- Date Range: {df_clean['Order Date'].min()} to {df_clean['Order Date'].max()}
- Total Sales: {insights['executive_summary']['total_sales']}
- Total Profit: {insights['executive_summary']['total_profit']}
- Average Profit Margin: {insights['executive_summary']['avg_profit_margin']}

CUSTOMER SEGMENTATION RESULTS:
- Total Segments Identified: {customer_segments['Segment_Name'].nunique()}
- Segment Distribution:
"""
        
        # Add segment distribution
        segment_counts = customer_segments['Segment_Name'].value_counts()
        for segment, count in segment_counts.items():
            percentage = (count / len(customer_segments)) * 100
            report += f"  • {segment}: {count:,} customers ({percentage:.1f}%)\n"
        
        report += f"""
TOP INSIGHTS:
- Best Performing Category: {insights['product_insights']['best_selling_category']}
- Most Profitable Category: {insights['product_insights']['most_profitable_category']}
- Top Region: {insights['regional_insights']['best_performing_region']}

KEY RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(insights['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
FILES GENERATED:
- cleaned_data.csv: Processed dataset
- customer_segments.csv: Customer segmentation results
- segment_summary.csv: Segment-level statistics
- portfolio_insights.json: Detailed business insights
- portfolio_insights_report.txt: Readable insights report
- Various visualization PNG files

NEXT STEPS:
1. Review the generated visualizations and reports
2. Launch the interactive dashboard: streamlit run main_dashboard.py
3. Implement the recommended business strategies
4. Schedule regular analysis updates

{'='*80}
"""
        
        # Save summary report
        with open('EXECUTIVE_SUMMARY.txt', 'w') as f:
            f.write(report)
        
        print("✅ Executive summary generated!")
        print("   - File: EXECUTIVE_SUMMARY.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating summary: {str(e)}")
        return False

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Print banner
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues and try again.")
        return
    
    # Execute pipeline
    steps = [
        ("Data Cleaning", run_data_cleaning),
        ("Customer Segmentation", run_customer_segmentation),
        ("Portfolio Analysis", run_portfolio_analysis),
        ("Summary Report", generate_summary_report)
    ]
    
    completed_steps = 0
    for step_name, step_function in steps:
        if step_function():
            completed_steps += 1
        else:
            print(f"\n❌ Pipeline stopped at: {step_name}")
            break
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("ANALYSIS PIPELINE COMPLETE")
    print("="*70)
    print(f"✅ Completed steps: {completed_steps}/{len(steps)}")
    print(f"⏱️  Total execution time: {duration:.1f} seconds")
    
    if completed_steps == len(steps):
        print("\n🎉 SUCCESS! All analyses completed successfully.")
        print("\nNext steps:")
        print("1. Review generated files and visualizations")
        print("2. Read EXECUTIVE_SUMMARY.txt for key insights")
        print("3. Launch interactive dashboard: streamlit run main_dashboard.py")
    else:
        print("\n⚠️  Some steps failed. Please check the error messages above.")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()