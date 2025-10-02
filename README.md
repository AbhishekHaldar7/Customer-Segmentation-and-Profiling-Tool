# Consumer Segmentation and Portfolio Tool 📊

A comprehensive business intelligence tool for customer segmentation and portfolio analysis using machine learning and data visualization.

## 🚀 Features

### 1. Data Cleaning & Preprocessing
- Automated data cleaning and validation
- Missing value handling
- Outlier detection and removal
- Feature engineering
- Data quality reporting

### 2. Customer Segmentation
- **RFM Analysis** (Recency, Frequency, Monetary)
- **K-means Clustering** with optimal cluster selection
- **Customer Value Scoring**
- Segment profiling and naming
- Interactive visualizations

### 3. Portfolio Analysis
- Product performance analysis
- Regional performance insights
- Temporal trend analysis
- BCG-style portfolio matrix
- Customer segment portfolio analysis
- Actionable business insights

### 4. Interactive Dashboard
- Real-time filtering and analysis
- Key Performance Indicators (KPIs)
- Interactive charts and visualizations
- Segment-wise performance tracking
- Export capabilities

## 📁 Project Structure

```
Consumer Segmentation and Portfolio Tool/
├── data.csv                          # Raw dataset
├── data_cleaning.py                  # Data preprocessing module
├── customer_segmentation.py          # Customer segmentation analysis
├── portfolio_analysis.py             # Portfolio performance analysis
├── main_dashboard.py                 # Interactive Streamlit dashboard
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── outputs/                          # Generated reports and visualizations
    ├── cleaned_data.csv
    ├── customer_segments.csv
    ├── segment_summary.csv
    ├── portfolio_insights.json
    └── various visualization files
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd "Consumer Segmentation and Portfolio Tool"
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data file**
   - Ensure `data.csv` is in the project directory
   - The dataset should contain columns: Customer ID, Sales, Profit, Order Date, etc.

## 🚀 Usage Guide

### Option 1: Run Complete Analysis Pipeline

1. **Data Cleaning**
   ```bash
   python data_cleaning.py
   ```
   - Cleans and preprocesses the raw data
   - Generates `cleaned_data.csv`

2. **Customer Segmentation**
   ```bash
   python customer_segmentation.py
   ```
   - Performs RFM analysis and clustering
   - Generates customer segments and visualizations

3. **Portfolio Analysis**
   ```bash
   python portfolio_analysis.py
   ```
   - Analyzes product and regional performance
   - Generates business insights and recommendations

### Option 2: Interactive Dashboard

```bash
streamlit run main_dashboard.py
```

This launches an interactive web dashboard where you can:
- View real-time KPIs
- Filter data by date, region, category, and customer segment
- Explore interactive visualizations
- Run analysis modules directly from the interface

## 📊 Key Outputs

### Customer Segmentation Results
- **customer_segments.csv**: Detailed customer data with segment assignments
- **segment_summary.csv**: Segment-level statistics and characteristics
- **customer_segments_dashboard.png**: Comprehensive visualization dashboard

### Portfolio Analysis Results
- **product_performance_analysis.csv**: Product-level performance metrics
- **regional_performance_analysis.csv**: Regional analysis results
- **portfolio_matrix.csv**: BCG-style portfolio classification
- **portfolio_insights.json**: Actionable business insights
- **portfolio_analysis_dashboard.png**: Visual analytics dashboard

## 🎯 Customer Segments

The tool automatically identifies and names customer segments:

1. **Champions** 🏆
   - High value, recent customers
   - Best customers for retention and upselling

2. **Loyal Customers** 💎
   - High value customers with good engagement
   - Focus on loyalty programs

3. **Potential Loyalists** 🌟
   - Recent customers with growth potential
   - Target for conversion campaigns

4. **At Risk** ⚠️
   - Valuable customers who haven't purchased recently
   - Priority for win-back campaigns

5. **New Customers** 🆕
   - Recent first-time buyers
   - Focus on onboarding and engagement

6. **Lost Customers** 😞
   - Haven't purchased in a long time
   - Consider reactivation campaigns

## 📈 Business Insights

The tool provides actionable insights including:

- **Product Performance**: Best and worst performing categories
- **Regional Analysis**: Geographic performance patterns
- **Seasonal Trends**: Monthly and seasonal patterns
- **Customer Behavior**: Segment-specific purchasing patterns
- **Profitability Analysis**: Margin analysis by various dimensions
- **Growth Opportunities**: Data-driven recommendations

## 🔧 Customization

### Adding New Metrics
Modify the analysis modules to include additional business metrics:

```python
# In customer_segmentation.py
customer_features['Custom_Metric'] = calculate_custom_metric(data)

# In portfolio_analysis.py
product_analysis['New_KPI'] = calculate_new_kpi(data)
```

### Adjusting Segmentation Logic
Customize the clustering parameters or segment naming logic:

```python
# Modify clustering parameters
kmeans = KMeans(n_clusters=custom_k, random_state=42)

# Adjust segment naming criteria
def custom_segment_naming(profile):
    # Your custom logic here
    return segment_name
```

## 📋 Data Requirements

The input dataset should contain the following columns:
- **Customer ID**: Unique customer identifier
- **Order Date**: Transaction date
- **Sales**: Revenue amount
- **Profit**: Profit amount
- **Category**: Product category
- **Sub-Category**: Product sub-category
- **Region**: Geographic region
- **State**: Geographic state
- **Quantity**: Items purchased
- **Discount**: Discount applied

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For questions or issues:
1. Check the documentation
2. Review the code comments
3. Create an issue with detailed description

## 🔄 Updates & Maintenance

Regular updates include:
- New visualization features
- Additional segmentation algorithms
- Enhanced business insights
- Performance optimizations

---

**Built with ❤️ using Python, Pandas, Scikit-learn, and Streamlit**