import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_cleaning import *
from customer_segmentation import CustomerSegmentation
from portfolio_analysis import PortfolioAnalysis

class Dashboard:
    def __init__(self):
        """Initialize the dashboard"""
        self.setup_page_config()
        self.load_data()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Consumer Segmentation & Portfolio Tool",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1f77b4;
        }
        .insight-box {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #1f77b4;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_data(self):
        """Load and cache data"""
        try:
            self.df = pd.read_csv('cleaned_data.csv')
            self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
            
            # Try to load customer segments
            try:
                self.customer_segments = pd.read_csv('customer_segments.csv', index_col=0)
                self.segments_available = True
            except FileNotFoundError:
                self.customer_segments = None
                self.segments_available = False
            
            self.data_loaded = True
        except FileNotFoundError:
            self.data_loaded = False
    
    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        if not self.data_loaded:
            st.sidebar.error("Data not found! Please run data_cleaning.py first.")
            return None, None, None, None
        
        # Date range filter
        min_date = self.df['Order Date'].min().date()
        max_date = self.df['Order Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Region filter
        regions = ['All'] + list(self.df['Region'].unique())
        selected_region = st.sidebar.selectbox("Select Region", regions)
        
        # Category filter
        categories = ['All'] + list(self.df['Category'].unique())
        selected_category = st.sidebar.selectbox("Select Category", categories)
        
        # Customer segment filter (if available)
        if self.segments_available:
            segments = ['All'] + list(self.customer_segments['Segment_Name'].unique())
            selected_segment = st.sidebar.selectbox("Select Customer Segment", segments)
        else:
            selected_segment = 'All'
        
        return date_range, selected_region, selected_category, selected_segment
    
    def filter_data(self, date_range, region, category, segment):
        """Filter data based on selections"""
        filtered_df = self.df.copy()
        
        # Date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['Order Date'].dt.date >= start_date) & 
                (filtered_df['Order Date'].dt.date <= end_date)
            ]
        
        # Region filter
        if region != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == region]
        
        # Category filter
        if category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == category]
        
        # Segment filter
        if segment != 'All' and self.segments_available:
            # Merge with customer segments
            segment_customers = self.customer_segments[
                self.customer_segments['Segment_Name'] == segment
            ].index
            filtered_df = filtered_df[filtered_df['Customer ID'].isin(segment_customers)]
        
        return filtered_df
    
    def display_kpis(self, df):
        """Display key performance indicators"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_sales = df['Sales'].sum()
        total_profit = df['Profit'].sum()
        avg_profit_margin = df['Profit Margin'].mean()
        total_orders = df['Order ID'].nunique()
        total_customers = df['Customer ID'].nunique()
        
        with col1:
            st.metric(
                label="💰 Total Sales",
                value=f"${total_sales:,.0f}",
                delta=f"{(total_sales/self.df['Sales'].sum()*100):.1f}% of total"
            )
        
        with col2:
            st.metric(
                label="📈 Total Profit", 
                value=f"${total_profit:,.0f}",
                delta=f"{(total_profit/total_sales*100):.1f}% margin"
            )
        
        with col3:
            st.metric(
                label="📊 Avg Profit Margin",
                value=f"{avg_profit_margin:.1f}%",
                delta=f"vs {self.df['Profit Margin'].mean():.1f}% overall"
            )
        
        with col4:
            st.metric(
                label="🛒 Total Orders",
                value=f"{total_orders:,}",
                delta=f"${total_sales/total_orders:.0f} avg order"
            )
        
        with col5:
            st.metric(
                label="👥 Total Customers",
                value=f"{total_customers:,}",
                delta=f"${total_sales/total_customers:.0f} per customer"
            )
    
    def create_sales_trend_chart(self, df):
        """Create interactive sales trend chart"""
        monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
        
        fig = px.line(
            monthly_sales, 
            x='Order Date', 
            y='Sales',
            title='Sales Trend Over Time',
            labels={'Sales': 'Sales ($)', 'Order Date': 'Date'}
        )
        fig.update_layout(height=400)
        return fig
    
    def create_category_performance_chart(self, df):
        """Create category performance chart"""
        category_data = df.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sales by Category', 'Profit by Category'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sales chart
        fig.add_trace(
            go.Bar(x=category_data['Category'], y=category_data['Sales'], name='Sales'),
            row=1, col=1
        )
        
        # Profit chart
        fig.add_trace(
            go.Bar(x=category_data['Category'], y=category_data['Profit'], name='Profit'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    def create_regional_map(self, df):
        """Create regional performance visualization"""
        regional_data = df.groupby('Region').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Customer ID': 'nunique'
        }).reset_index()
        
        fig = px.bar(
            regional_data,
            x='Region',
            y='Sales',
            color='Profit',
            title='Regional Performance',
            labels={'Sales': 'Sales ($)', 'Profit': 'Profit ($)'}
        )
        fig.update_layout(height=400)
        return fig
    
    def create_customer_segment_chart(self, df):
        """Create customer segment analysis chart"""
        if not self.segments_available:
            return None
        
        # Merge with segments
        df_with_segments = df.merge(
            self.customer_segments[['Segment_Name']], 
            left_on='Customer ID', 
            right_index=True, 
            how='left'
        )
        
        segment_data = df_with_segments.groupby('Segment_Name').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Customer ID': 'nunique'
        }).reset_index()
        
        fig = px.sunburst(
            segment_data,
            path=['Segment_Name'],
            values='Sales',
            title='Sales Distribution by Customer Segment'
        )
        fig.update_layout(height=400)
        return fig
    
    def create_product_performance_table(self, df):
        """Create product performance table"""
        product_data = df.groupby(['Category', 'Sub-Category']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Order ID': 'nunique'
        }).round(2)
        
        product_data['Profit_Margin'] = (product_data['Profit'] / product_data['Sales'] * 100).round(2)
        product_data = product_data.sort_values('Sales', ascending=False)
        
        return product_data
    
    def display_insights(self, df):
        """Display key insights"""
        st.markdown("### 🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h4>📊 Performance Highlights</h4>
            """, unsafe_allow_html=True)
            
            # Top performing category
            top_category = df.groupby('Category')['Sales'].sum().idxmax()
            top_category_sales = df.groupby('Category')['Sales'].sum().max()
            
            st.write(f"• **Best Category:** {top_category} (${top_category_sales:,.0f})")
            
            # Most profitable month
            monthly_profit = df.groupby(df['Order Date'].dt.month)['Profit'].sum()
            best_month = monthly_profit.idxmax()
            month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                          7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            
            st.write(f"• **Best Month:** {month_names[best_month]} (${monthly_profit.max():,.0f} profit)")
            
            # Top region
            top_region = df.groupby('Region')['Sales'].sum().idxmax()
            st.write(f"• **Top Region:** {top_region}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>⚠️ Areas for Improvement</h4>
            """, unsafe_allow_html=True)
            
            # Low margin products
            low_margin = df.groupby('Category')['Profit Margin'].mean().idxmin()
            low_margin_value = df.groupby('Category')['Profit Margin'].mean().min()
            
            st.write(f"• **Low Margin Category:** {low_margin} ({low_margin_value:.1f}%)")
            
            # Underperforming region
            worst_region = df.groupby('Region')['Sales'].sum().idxmin()
            st.write(f"• **Underperforming Region:** {worst_region}")
            
            # High discount impact
            high_discount_impact = df[df['Discount'] > 0.5]['Profit Margin'].mean()
            st.write(f"• **High Discount Impact:** {high_discount_impact:.1f}% avg margin with >50% discount")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def run_analysis_tools(self):
        """Run analysis tools section"""
        st.markdown("### 🛠️ Analysis Tools")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Run Data Cleaning", use_container_width=True):
                with st.spinner("Cleaning data..."):
                    # Run data cleaning
                    exec(open('data_cleaning.py').read())
                    st.success("Data cleaning completed!")
                    st.experimental_rerun()
        
        with col2:
            if st.button("👥 Run Customer Segmentation", use_container_width=True):
                with st.spinner("Performing customer segmentation..."):
                    try:
                        segmentation = CustomerSegmentation()
                        segmentation.create_customer_features()
                        segmentation.perform_clustering()
                        segmentation.save_results()
                        st.success("Customer segmentation completed!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col3:
            if st.button("📈 Run Portfolio Analysis", use_container_width=True):
                with st.spinner("Analyzing portfolio..."):
                    try:
                        portfolio = PortfolioAnalysis()
                        portfolio.product_performance_analysis()
                        portfolio.regional_analysis()
                        portfolio.temporal_analysis()
                        portfolio.create_portfolio_matrix()
                        portfolio.save_results()
                        st.success("Portfolio analysis completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    def main(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">📊 Consumer Segmentation & Portfolio Tool</h1>', 
                   unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.error("⚠️ Data not found! Please run the data cleaning process first.")
            self.run_analysis_tools()
            return
        
        # Sidebar controls
        date_range, region, category, segment = self.sidebar_controls()
        
        # Filter data
        filtered_df = self.filter_data(date_range, region, category, segment)
        
        # Display KPIs
        st.markdown("### 📊 Key Performance Indicators")
        self.display_kpis(filtered_df)
        
        # Charts section
        st.markdown("### 📈 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales trend
            sales_chart = self.create_sales_trend_chart(filtered_df)
            st.plotly_chart(sales_chart, use_container_width=True)
        
        with col2:
            # Regional performance
            regional_chart = self.create_regional_map(filtered_df)
            st.plotly_chart(regional_chart, use_container_width=True)
        
        # Category performance
        category_chart = self.create_category_performance_chart(filtered_df)
        st.plotly_chart(category_chart, use_container_width=True)
        
        # Customer segments (if available)
        if self.segments_available:
            segment_chart = self.create_customer_segment_chart(filtered_df)
            if segment_chart:
                st.plotly_chart(segment_chart, use_container_width=True)
        
        # Product performance table
        st.markdown("### 📋 Product Performance Details")
        product_table = self.create_product_performance_table(filtered_df)
        st.dataframe(product_table, use_container_width=True)
        
        # Insights
        self.display_insights(filtered_df)
        
        # Analysis tools
        self.run_analysis_tools()
        
        # Footer
        st.markdown("---")
        st.markdown("**Consumer Segmentation & Portfolio Tool** | Built with Streamlit & Python")

def main():
    """Run the dashboard"""
    dashboard = Dashboard()
    dashboard.main()

if __name__ == "__main__":
    main()