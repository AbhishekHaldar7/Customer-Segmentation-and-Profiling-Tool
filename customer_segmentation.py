import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self, data_path='cleaned_data.csv'):
        """Initialize with cleaned data"""
        self.df = pd.read_csv(data_path)
        self.customer_features = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.segments = None
        
    def create_customer_features(self):
        """Create RFM and other customer-level features"""
        print("Creating customer features...")
        
        # Convert Order Date to datetime if not already
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
        
        # Calculate reference date (latest date + 1 day)
        reference_date = self.df['Order Date'].max() + pd.Timedelta(days=1)
        
        # Create customer-level aggregations
        customer_features = self.df.groupby('Customer ID').agg({
            'Order Date': ['count', 'max'],  # Frequency and last order date
            'Sales': ['sum', 'mean'],        # Total and average sales
            'Profit': ['sum', 'mean'],       # Total and average profit
            'Quantity': ['sum', 'mean'],     # Total and average quantity
            'Discount': 'mean',              # Average discount
            'Profit Margin': 'mean'          # Average profit margin
        }).round(2)
        
        # Flatten column names
        customer_features.columns = [
            'Frequency', 'Last_Order_Date', 'Total_Sales', 'Avg_Sales',
            'Total_Profit', 'Avg_Profit', 'Total_Quantity', 'Avg_Quantity',
            'Avg_Discount', 'Avg_Profit_Margin'
        ]
        
        # Calculate Recency (days since last order)
        customer_features['Recency'] = (reference_date - customer_features['Last_Order_Date']).dt.days
        
        # Calculate Monetary (using total sales)
        customer_features['Monetary'] = customer_features['Total_Sales']
        
        # Add customer segment and region info
        customer_info = self.df.groupby('Customer ID').agg({
            'Segment': 'first',
            'Region': 'first',
            'Category': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]  # Most frequent category
        })
        
        customer_features = customer_features.join(customer_info)
        
        # Create customer value score
        customer_features['Customer_Value_Score'] = (
            customer_features['Total_Sales'] * 0.4 +
            customer_features['Total_Profit'] * 0.4 +
            customer_features['Frequency'] * 0.2
        )
        
        self.customer_features = customer_features
        print(f"Created features for {len(customer_features)} customers")
        return customer_features
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("Finding optimal number of clusters...")
        
        # Select features for clustering (RFM + additional metrics)
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'Avg_Profit_Margin', 'Customer_Value_Score']
        X = self.customer_features[clustering_features].fillna(0)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate metrics for different cluster numbers
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def perform_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        print(f"Performing clustering with {n_clusters} clusters...")
        
        # Select features for clustering
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'Avg_Profit_Margin', 'Customer_Value_Score']
        X = self.customer_features[clustering_features].fillna(0)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to customer features
        self.customer_features['Cluster'] = cluster_labels
        
        # Create segment names based on characteristics
        self.segments = self.create_segment_profiles()
        
        print("Clustering completed!")
        return cluster_labels
    
    def create_segment_profiles(self):
        """Create meaningful segment names and profiles"""
        print("Creating segment profiles...")
        
        # Calculate cluster characteristics
        cluster_profiles = self.customer_features.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean',
            'Avg_Profit_Margin': 'mean',
            'Customer_Value_Score': 'mean',
            'Total_Sales': 'mean',
            'Total_Profit': 'mean'
        }).round(2)
        
        # Create segment names based on RFM characteristics
        segment_names = {}
        for cluster in cluster_profiles.index:
            profile = cluster_profiles.loc[cluster]
            
            # Determine segment name based on characteristics
            if profile['Customer_Value_Score'] > cluster_profiles['Customer_Value_Score'].quantile(0.75):
                if profile['Recency'] < cluster_profiles['Recency'].quantile(0.25):
                    segment_names[cluster] = "Champions"
                else:
                    segment_names[cluster] = "Loyal Customers"
            elif profile['Customer_Value_Score'] > cluster_profiles['Customer_Value_Score'].quantile(0.5):
                if profile['Recency'] < cluster_profiles['Recency'].quantile(0.5):
                    segment_names[cluster] = "Potential Loyalists"
                else:
                    segment_names[cluster] = "At Risk"
            else:
                if profile['Recency'] > cluster_profiles['Recency'].quantile(0.75):
                    segment_names[cluster] = "Lost Customers"
                else:
                    segment_names[cluster] = "New Customers"
        
        # Add segment names to customer features
        self.customer_features['Segment_Name'] = self.customer_features['Cluster'].map(segment_names)
        
        # Display segment profiles
        print("\nSegment Profiles:")
        print("="*80)
        for cluster in sorted(cluster_profiles.index):
            profile = cluster_profiles.loc[cluster]
            count = len(self.customer_features[self.customer_features['Cluster'] == cluster])
            percentage = (count / len(self.customer_features)) * 100
            
            print(f"\nCluster {cluster}: {segment_names[cluster]}")
            print(f"Size: {count} customers ({percentage:.1f}%)")
            print(f"Avg Recency: {profile['Recency']:.0f} days")
            print(f"Avg Frequency: {profile['Frequency']:.1f} orders")
            print(f"Avg Monetary: ${profile['Monetary']:,.2f}")
            print(f"Avg Profit Margin: {profile['Avg_Profit_Margin']:.1f}%")
            print(f"Customer Value Score: {profile['Customer_Value_Score']:.2f}")
        
        return segment_names
    
    def visualize_segments(self):
        """Create visualizations for customer segments"""
        print("Creating segment visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Segment distribution
        ax1 = plt.subplot(3, 3, 1)
        segment_counts = self.customer_features['Segment_Name'].value_counts()
        colors = sns.color_palette("husl", len(segment_counts))
        plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        
        # 2. RFM scatter plot
        ax2 = plt.subplot(3, 3, 2)
        scatter = plt.scatter(self.customer_features['Recency'], 
                            self.customer_features['Monetary'],
                            c=self.customer_features['Cluster'], 
                            cmap='viridis', alpha=0.6)
        plt.xlabel('Recency (days)')
        plt.ylabel('Monetary ($)')
        plt.title('Recency vs Monetary by Segment')
        plt.colorbar(scatter)
        
        # 3. Frequency vs Monetary
        ax3 = plt.subplot(3, 3, 3)
        scatter = plt.scatter(self.customer_features['Frequency'], 
                            self.customer_features['Monetary'],
                            c=self.customer_features['Cluster'], 
                            cmap='viridis', alpha=0.6)
        plt.xlabel('Frequency (orders)')
        plt.ylabel('Monetary ($)')
        plt.title('Frequency vs Monetary by Segment')
        plt.colorbar(scatter)
        
        # 4. Average sales by segment
        ax4 = plt.subplot(3, 3, 4)
        avg_sales = self.customer_features.groupby('Segment_Name')['Total_Sales'].mean().sort_values(ascending=False)
        bars = plt.bar(range(len(avg_sales)), avg_sales.values, color=colors)
        plt.xticks(range(len(avg_sales)), avg_sales.index, rotation=45, ha='right')
        plt.ylabel('Average Total Sales ($)')
        plt.title('Average Sales by Segment')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_sales)*0.01,
                    f'${avg_sales.iloc[i]:,.0f}', ha='center', va='bottom')
        
        # 5. Customer value score distribution
        ax5 = plt.subplot(3, 3, 5)
        for segment in self.customer_features['Segment_Name'].unique():
            segment_data = self.customer_features[self.customer_features['Segment_Name'] == segment]
            plt.hist(segment_data['Customer_Value_Score'], alpha=0.7, label=segment, bins=20)
        plt.xlabel('Customer Value Score')
        plt.ylabel('Frequency')
        plt.title('Customer Value Score Distribution')
        plt.legend()
        
        # 6. Profit margin by segment
        ax6 = plt.subplot(3, 3, 6)
        sns.boxplot(data=self.customer_features, x='Segment_Name', y='Avg_Profit_Margin')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Profit Margin (%)')
        plt.title('Profit Margin Distribution by Segment')
        
        # 7. Recency distribution
        ax7 = plt.subplot(3, 3, 7)
        sns.boxplot(data=self.customer_features, x='Segment_Name', y='Recency')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Recency (days)')
        plt.title('Recency Distribution by Segment')
        
        # 8. Segment characteristics heatmap
        ax8 = plt.subplot(3, 3, 8)
        segment_metrics = self.customer_features.groupby('Segment_Name')[
            ['Recency', 'Frequency', 'Monetary', 'Avg_Profit_Margin', 'Customer_Value_Score']
        ].mean()
        
        # Normalize for better visualization
        segment_metrics_norm = (segment_metrics - segment_metrics.min()) / (segment_metrics.max() - segment_metrics.min())
        sns.heatmap(segment_metrics_norm.T, annot=True, cmap='RdYlBu_r', fmt='.2f')
        plt.title('Segment Characteristics (Normalized)')
        plt.ylabel('Metrics')
        
        # 9. Geographic distribution
        ax9 = plt.subplot(3, 3, 9)
        region_segment = pd.crosstab(self.customer_features['Region'], self.customer_features['Segment_Name'])
        region_segment.plot(kind='bar', stacked=True, ax=ax9)
        plt.title('Segment Distribution by Region')
        plt.xlabel('Region')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('customer_segments_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'customer_segments_dashboard.png'")
    
    def save_results(self):
        """Save segmentation results"""
        # Save customer features with segments
        self.customer_features.to_csv('customer_segments.csv')
        
        # Create segment summary
        segment_summary = self.customer_features.groupby('Segment_Name').agg({
            'Customer ID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Total_Sales': ['mean', 'sum'],
            'Total_Profit': ['mean', 'sum'],
            'Avg_Profit_Margin': 'mean',
            'Customer_Value_Score': 'mean'
        }).round(2)
        
        segment_summary.columns = [
            'Customer_Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
            'Avg_Total_Sales', 'Sum_Total_Sales', 'Avg_Total_Profit', 'Sum_Total_Profit',
            'Avg_Profit_Margin', 'Avg_Customer_Value_Score'
        ]
        
        segment_summary.to_csv('segment_summary.csv')
        
        print("Results saved:")
        print("- customer_segments.csv: Detailed customer data with segments")
        print("- segment_summary.csv: Segment-level summary statistics")
        
        return segment_summary

def main():
    """Main execution function"""
    print("="*60)
    print("CUSTOMER SEGMENTATION ANALYSIS")
    print("="*60)
    
    # Initialize segmentation
    segmentation = CustomerSegmentation()
    
    # Create customer features
    segmentation.create_customer_features()
    
    # Perform clustering
    segmentation.perform_clustering()
    
    # Create visualizations
    segmentation.visualize_segments()
    
    # Save results
    summary = segmentation.save_results()
    
    print("\n" + "="*60)
    print("SEGMENTATION COMPLETE!")
    print("="*60)
    print(f"Total customers segmented: {len(segmentation.customer_features):,}")
    print(f"Number of segments: {segmentation.customer_features['Segment_Name'].nunique()}")
    
    return segmentation

if __name__ == "__main__":
    segmentation = main()