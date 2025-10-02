import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalysis:
    def __init__(self, data_path='cleaned_data.csv', segments_path='customer_segments.csv'):
        """Initialize with cleaned data and customer segments"""
        self.df = pd.read_csv(data_path)
        try:
            self.customer_segments = pd.read_csv(segments_path, index_col=0)
            print("Customer segments loaded successfully")
        except FileNotFoundError:
            print("Customer segments file not found. Run customer_segmentation.py first.")
            self.customer_segments = None
        
        # Convert date columns
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
        self.df['Order Year'] = self.df['Order Date'].dt.year
        self.df['Order Month'] = self.df['Order Date'].dt.month
        
    def product_performance_analysis(self):
        """Analyze product performance across different dimensions"""
        print("Analyzing product performance...")
        
        # Product-level analysis
        product_analysis = self.df.groupby(['Category', 'Sub-Category']).agg({
            'Sales': ['sum', 'mean', 'count'],
            'Profit': ['sum', 'mean'],
            'Quantity': 'sum',
            'Discount': 'mean',
            'Profit Margin': 'mean'
        }).round(2)
        
        # Flatten column names
        product_analysis.columns = [
            'Total_Sales', 'Avg_Sales', 'Order_Count',
            'Total_Profit', 'Avg_Profit', 'Total_Quantity',
            'Avg_Discount', 'Avg_Profit_Margin'
        ]
        
        # Calculate additional metrics
        product_analysis['Revenue_per_Order'] = product_analysis['Total_Sales'] / product_analysis['Order_Count']
        product_analysis['Profit_per_Order'] = product_analysis['Total_Profit'] / product_analysis['Order_Count']
        
        # Create performance score
        product_analysis['Performance_Score'] = (
            (product_analysis['Total_Sales'] / product_analysis['Total_Sales'].max()) * 0.3 +
            (product_analysis['Total_Profit'] / product_analysis['Total_Profit'].max()) * 0.3 +
            (product_analysis['Avg_Profit_Margin'] / product_analysis['Avg_Profit_Margin'].max()) * 0.2 +
            (product_analysis['Order_Count'] / product_analysis['Order_Count'].max()) * 0.2
        ) * 100
        
        # Sort by performance score
        product_analysis = product_analysis.sort_values('Performance_Score', ascending=False)
        
        self.product_analysis = product_analysis
        return product_analysis
    
    def regional_analysis(self):
        """Analyze performance by region and state"""
        print("Analyzing regional performance...")
        
        # Regional analysis
        regional_analysis = self.df.groupby(['Region', 'State']).agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Customer ID': 'nunique',
            'Order ID': 'nunique',
            'Profit Margin': 'mean'
        }).round(2)
        
        # Flatten column names
        regional_analysis.columns = [
            'Total_Sales', 'Avg_Sales', 'Total_Profit', 'Avg_Profit',
            'Unique_Customers', 'Total_Orders', 'Avg_Profit_Margin'
        ]
        
        # Calculate metrics
        regional_analysis['Sales_per_Customer'] = regional_analysis['Total_Sales'] / regional_analysis['Unique_Customers']
        regional_analysis['Orders_per_Customer'] = regional_analysis['Total_Orders'] / regional_analysis['Unique_Customers']
        
        self.regional_analysis = regional_analysis
        return regional_analysis
    
    def temporal_analysis(self):
        """Analyze trends over time"""
        print("Analyzing temporal trends...")
        
        # Monthly trends
        monthly_trends = self.df.groupby(['Order Year', 'Order Month']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique',
            'Customer ID': 'nunique'
        }).round(2)
        
        monthly_trends.columns = ['Monthly_Sales', 'Monthly_Profit', 'Monthly_Orders', 'Monthly_Customers']
        
        # Calculate growth rates
        monthly_trends['Sales_Growth'] = monthly_trends['Monthly_Sales'].pct_change() * 100
        monthly_trends['Profit_Growth'] = monthly_trends['Monthly_Profit'].pct_change() * 100
        
        # Seasonal analysis
        seasonal_analysis = self.df.groupby('Order Month').agg({
            'Sales': 'mean',
            'Profit': 'mean',
            'Order ID': 'count'
        }).round(2)
        
        seasonal_analysis.columns = ['Avg_Monthly_Sales', 'Avg_Monthly_Profit', 'Avg_Monthly_Orders']
        
        self.monthly_trends = monthly_trends
        self.seasonal_analysis = seasonal_analysis
        return monthly_trends, seasonal_analysis
    
    def customer_segment_portfolio(self):
        """Analyze portfolio performance by customer segments"""
        if self.customer_segments is None:
            print("Customer segments not available. Skipping segment analysis.")
            return None
        
        print("Analyzing portfolio by customer segments...")
        
        # Merge transaction data with customer segments
        df_with_segments = self.df.merge(
            self.customer_segments[['Segment_Name']], 
            left_on='Customer ID', 
            right_index=True, 
            how='left'
        )
        
        # Segment-wise product performance
        segment_product_analysis = df_with_segments.groupby(['Segment_Name', 'Category']).agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Quantity': 'sum',
            'Order ID': 'nunique'
        }).round(2)
        
        # Flatten column names
        segment_product_analysis.columns = [
            'Total_Sales', 'Avg_Sales', 'Total_Profit', 'Avg_Profit',
            'Total_Quantity', 'Order_Count'
        ]
        
        self.segment_product_analysis = segment_product_analysis
        return segment_product_analysis
    
    def create_portfolio_matrix(self):
        """Create BCG-style portfolio matrix"""
        print("Creating portfolio matrix...")
        
        # Calculate market share and growth for each product category
        category_metrics = self.df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique'
        })
        
        # Calculate market share (relative to total)
        category_metrics['Market_Share'] = category_metrics['Sales'] / category_metrics['Sales'].sum() * 100
        
        # Calculate growth rate (comparing first and last year)
        years = sorted(self.df['Order Year'].unique())
        if len(years) > 1:
            first_year = years[0]
            last_year = years[-1]
            
            first_year_sales = self.df[self.df['Order Year'] == first_year].groupby('Category')['Sales'].sum()
            last_year_sales = self.df[self.df['Order Year'] == last_year].groupby('Category')['Sales'].sum()
            
            growth_rates = ((last_year_sales - first_year_sales) / first_year_sales * 100).fillna(0)
            category_metrics['Growth_Rate'] = growth_rates
        else:
            category_metrics['Growth_Rate'] = 0
        
        # Classify into portfolio quadrants
        market_share_median = category_metrics['Market_Share'].median()
        growth_rate_median = category_metrics['Growth_Rate'].median()
        
        def classify_portfolio(row):
            if row['Market_Share'] >= market_share_median and row['Growth_Rate'] >= growth_rate_median:
                return 'Stars'
            elif row['Market_Share'] >= market_share_median and row['Growth_Rate'] < growth_rate_median:
                return 'Cash Cows'
            elif row['Market_Share'] < market_share_median and row['Growth_Rate'] >= growth_rate_median:
                return 'Question Marks'
            else:
                return 'Dogs'
        
        category_metrics['Portfolio_Category'] = category_metrics.apply(classify_portfolio, axis=1)
        
        self.portfolio_matrix = category_metrics
        return category_metrics
    
    def visualize_portfolio_analysis(self):
        """Create comprehensive portfolio visualizations"""
        print("Creating portfolio visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Product Performance by Category
        ax1 = plt.subplot(3, 4, 1)
        category_sales = self.df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        bars = plt.bar(range(len(category_sales)), category_sales.values)
        plt.xticks(range(len(category_sales)), category_sales.index, rotation=45, ha='right')
        plt.ylabel('Total Sales ($)')
        plt.title('Sales by Product Category')
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_sales)*0.01,
                    f'${category_sales.iloc[i]/1000:.0f}K', ha='center', va='bottom')
        
        # 2. Profit by Category
        ax2 = plt.subplot(3, 4, 2)
        category_profit = self.df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
        bars = plt.bar(range(len(category_profit)), category_profit.values, color='green', alpha=0.7)
        plt.xticks(range(len(category_profit)), category_profit.index, rotation=45, ha='right')
        plt.ylabel('Total Profit ($)')
        plt.title('Profit by Product Category')
        
        # 3. Regional Performance
        ax3 = plt.subplot(3, 4, 3)
        regional_sales = self.df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        plt.pie(regional_sales.values, labels=regional_sales.index, autopct='%1.1f%%')
        plt.title('Sales Distribution by Region')
        
        # 4. Monthly Sales Trend
        ax4 = plt.subplot(3, 4, 4)
        monthly_sales = self.df.groupby(['Order Year', 'Order Month'])['Sales'].sum()
        monthly_sales.plot(kind='line', marker='o')
        plt.title('Monthly Sales Trend')
        plt.ylabel('Sales ($)')
        plt.xticks(rotation=45)
        
        # 5. Top 10 Sub-Categories by Sales
        ax5 = plt.subplot(3, 4, 5)
        top_subcategories = self.df.groupby('Sub-Category')['Sales'].sum().nlargest(10)
        plt.barh(range(len(top_subcategories)), top_subcategories.values)
        plt.yticks(range(len(top_subcategories)), top_subcategories.index)
        plt.xlabel('Total Sales ($)')
        plt.title('Top 10 Sub-Categories by Sales')
        
        # 6. Profit Margin Distribution
        ax6 = plt.subplot(3, 4, 6)
        plt.hist(self.df['Profit Margin'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Profit Margin (%)')
        plt.ylabel('Frequency')
        plt.title('Profit Margin Distribution')
        plt.axvline(self.df['Profit Margin'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["Profit Margin"].mean():.1f}%')
        plt.legend()
        
        # 7. Sales vs Profit Scatter
        ax7 = plt.subplot(3, 4, 7)
        category_metrics = self.df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'})
        scatter = plt.scatter(category_metrics['Sales'], category_metrics['Profit'], 
                            s=100, alpha=0.7, c=range(len(category_metrics)), cmap='viridis')
        
        # Add labels for each point
        for i, category in enumerate(category_metrics.index):
            plt.annotate(category, (category_metrics.iloc[i]['Sales'], category_metrics.iloc[i]['Profit']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Total Sales ($)')
        plt.ylabel('Total Profit ($)')
        plt.title('Sales vs Profit by Category')
        
        # 8. Portfolio Matrix (BCG Matrix)
        ax8 = plt.subplot(3, 4, 8)
        if hasattr(self, 'portfolio_matrix'):
            colors = {'Stars': 'gold', 'Cash Cows': 'green', 'Question Marks': 'orange', 'Dogs': 'red'}
            for category, color in colors.items():
                data = self.portfolio_matrix[self.portfolio_matrix['Portfolio_Category'] == category]
                if not data.empty:
                    plt.scatter(data['Market_Share'], data['Growth_Rate'], 
                              c=color, label=category, s=100, alpha=0.7)
            
            plt.axhline(y=self.portfolio_matrix['Growth_Rate'].median(), color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=self.portfolio_matrix['Market_Share'].median(), color='black', linestyle='--', alpha=0.5)
            plt.xlabel('Market Share (%)')
            plt.ylabel('Growth Rate (%)')
            plt.title('Portfolio Matrix (BCG Style)')
            plt.legend()
        
        # 9. Seasonal Analysis
        ax9 = plt.subplot(3, 4, 9)
        monthly_avg = self.df.groupby('Order Month')['Sales'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.plot(range(1, 13), monthly_avg.values, marker='o', linewidth=2)
        plt.xticks(range(1, 13), months, rotation=45)
        plt.ylabel('Average Sales ($)')
        plt.title('Seasonal Sales Pattern')
        plt.grid(True, alpha=0.3)
        
        # 10. Customer Segment Performance (if available)
        ax10 = plt.subplot(3, 4, 10)
        if self.customer_segments is not None:
            df_with_segments = self.df.merge(
                self.customer_segments[['Segment_Name']], 
                left_on='Customer ID', 
                right_index=True, 
                how='left'
            )
            segment_sales = df_with_segments.groupby('Segment_Name')['Sales'].sum().sort_values(ascending=False)
            plt.bar(range(len(segment_sales)), segment_sales.values)
            plt.xticks(range(len(segment_sales)), segment_sales.index, rotation=45, ha='right')
            plt.ylabel('Total Sales ($)')
            plt.title('Sales by Customer Segment')
        else:
            plt.text(0.5, 0.5, 'Customer Segments\nNot Available', ha='center', va='center', transform=ax10.transAxes)
            plt.title('Customer Segment Analysis')
        
        # 11. Discount Impact Analysis
        ax11 = plt.subplot(3, 4, 11)
        # Create discount bins
        self.df['Discount_Bin'] = pd.cut(self.df['Discount'], bins=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
        discount_impact = self.df.groupby('Discount_Bin')['Profit Margin'].mean()
        plt.bar(range(len(discount_impact)), discount_impact.values, color='coral')
        plt.xticks(range(len(discount_impact)), discount_impact.index, rotation=45)
        plt.ylabel('Average Profit Margin (%)')
        plt.title('Discount Impact on Profit Margin')
        
        # 12. Top Customers by Revenue
        ax12 = plt.subplot(3, 4, 12)
        top_customers = self.df.groupby('Customer ID')['Sales'].sum().nlargest(10)
        plt.barh(range(len(top_customers)), top_customers.values)
        plt.yticks(range(len(top_customers)), [f'Customer {i+1}' for i in range(len(top_customers))])
        plt.xlabel('Total Sales ($)')
        plt.title('Top 10 Customers by Revenue')
        
        plt.tight_layout()
        plt.savefig('portfolio_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Portfolio analysis dashboard saved as 'portfolio_analysis_dashboard.png'")
    
    def generate_insights_report(self):
        """Generate actionable insights and recommendations"""
        print("Generating insights and recommendations...")
        
        insights = {
            'executive_summary': {},
            'product_insights': {},
            'regional_insights': {},
            'customer_insights': {},
            'recommendations': []
        }
        
        # Executive Summary
        total_sales = self.df['Sales'].sum()
        total_profit = self.df['Profit'].sum()
        avg_profit_margin = self.df['Profit Margin'].mean()
        total_customers = self.df['Customer ID'].nunique()
        total_orders = self.df['Order ID'].nunique()
        
        insights['executive_summary'] = {
            'total_sales': f"${total_sales:,.2f}",
            'total_profit': f"${total_profit:,.2f}",
            'avg_profit_margin': f"{avg_profit_margin:.2f}%",
            'total_customers': f"{total_customers:,}",
            'total_orders': f"{total_orders:,}",
            'avg_order_value': f"${total_sales/total_orders:.2f}"
        }
        
        # Product Insights
        best_category = self.df.groupby('Category')['Sales'].sum().idxmax()
        worst_category = self.df.groupby('Category')['Sales'].sum().idxmin()
        most_profitable_category = self.df.groupby('Category')['Profit'].sum().idxmax()
        
        insights['product_insights'] = {
            'best_selling_category': best_category,
            'worst_selling_category': worst_category,
            'most_profitable_category': most_profitable_category,
            'top_subcategories': self.df.groupby('Sub-Category')['Sales'].sum().nlargest(5).to_dict()
        }
        
        # Regional Insights
        best_region = self.df.groupby('Region')['Sales'].sum().idxmax()
        best_state = self.df.groupby('State')['Sales'].sum().idxmax()
        
        insights['regional_insights'] = {
            'best_performing_region': best_region,
            'best_performing_state': best_state,
            'regional_distribution': self.df.groupby('Region')['Sales'].sum().to_dict()
        }
        
        # Generate Recommendations
        recommendations = []
        
        # Product recommendations
        low_margin_categories = self.df.groupby('Category')['Profit Margin'].mean()
        low_margin_cat = low_margin_categories[low_margin_categories < 10].index.tolist()
        if low_margin_cat:
            recommendations.append(f"Focus on improving profit margins for {', '.join(low_margin_cat)} categories")
        
        # Regional recommendations
        regional_performance = self.df.groupby('Region')['Sales'].sum()
        underperforming_regions = regional_performance[regional_performance < regional_performance.mean()].index.tolist()
        if underperforming_regions:
            recommendations.append(f"Develop targeted marketing strategies for {', '.join(underperforming_regions)} region(s)")
        
        # Seasonal recommendations
        monthly_sales = self.df.groupby('Order Month')['Sales'].mean()
        peak_months = monthly_sales.nlargest(3).index.tolist()
        recommendations.append(f"Prepare inventory and marketing campaigns for peak months: {', '.join([str(m) for m in peak_months])}")
        
        # Customer segment recommendations (if available)
        if self.customer_segments is not None:
            recommendations.append("Implement targeted retention strategies for 'At Risk' and 'Lost Customers' segments")
            recommendations.append("Develop loyalty programs for 'Champions' and 'Loyal Customers' segments")
        
        insights['recommendations'] = recommendations
        
        # Save insights to file
        import json
        with open('portfolio_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Create readable report
        report = f"""
PORTFOLIO ANALYSIS INSIGHTS REPORT
{'='*50}

EXECUTIVE SUMMARY:
- Total Sales: {insights['executive_summary']['total_sales']}
- Total Profit: {insights['executive_summary']['total_profit']}
- Average Profit Margin: {insights['executive_summary']['avg_profit_margin']}
- Total Customers: {insights['executive_summary']['total_customers']}
- Total Orders: {insights['executive_summary']['total_orders']}
- Average Order Value: {insights['executive_summary']['avg_order_value']}

PRODUCT INSIGHTS:
- Best Selling Category: {insights['product_insights']['best_selling_category']}
- Most Profitable Category: {insights['product_insights']['most_profitable_category']}
- Worst Selling Category: {insights['product_insights']['worst_selling_category']}

REGIONAL INSIGHTS:
- Best Performing Region: {insights['regional_insights']['best_performing_region']}
- Best Performing State: {insights['regional_insights']['best_performing_state']}

KEY RECOMMENDATIONS:
"""
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        with open('portfolio_insights_report.txt', 'w') as f:
            f.write(report)
        
        print("Insights saved to:")
        print("- portfolio_insights.json (detailed data)")
        print("- portfolio_insights_report.txt (readable report)")
        
        return insights
    
    def save_results(self):
        """Save all analysis results"""
        print("Saving portfolio analysis results...")
        
        # Save product analysis
        if hasattr(self, 'product_analysis'):
            self.product_analysis.to_csv('product_performance_analysis.csv')
        
        # Save regional analysis
        if hasattr(self, 'regional_analysis'):
            self.regional_analysis.to_csv('regional_performance_analysis.csv')
        
        # Save temporal analysis
        if hasattr(self, 'monthly_trends'):
            self.monthly_trends.to_csv('monthly_trends_analysis.csv')
        
        if hasattr(self, 'seasonal_analysis'):
            self.seasonal_analysis.to_csv('seasonal_analysis.csv')
        
        # Save portfolio matrix
        if hasattr(self, 'portfolio_matrix'):
            self.portfolio_matrix.to_csv('portfolio_matrix.csv')
        
        # Save segment analysis
        if hasattr(self, 'segment_product_analysis'):
            self.segment_product_analysis.to_csv('segment_product_analysis.csv')
        
        print("Portfolio analysis results saved!")

def main():
    """Main execution function"""
    print("="*60)
    print("PORTFOLIO ANALYSIS")
    print("="*60)
    
    # Initialize portfolio analysis
    portfolio = PortfolioAnalysis()
    
    # Perform analyses
    portfolio.product_performance_analysis()
    portfolio.regional_analysis()
    portfolio.temporal_analysis()
    portfolio.customer_segment_portfolio()
    portfolio.create_portfolio_matrix()
    
    # Create visualizations
    portfolio.visualize_portfolio_analysis()
    
    # Generate insights
    insights = portfolio.generate_insights_report()
    
    # Save results
    portfolio.save_results()
    
    print("\n" + "="*60)
    print("PORTFOLIO ANALYSIS COMPLETE!")
    print("="*60)
    
    return portfolio, insights

if __name__ == "__main__":
    portfolio, insights = main()