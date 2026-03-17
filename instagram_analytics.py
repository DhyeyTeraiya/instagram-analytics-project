"""
Instagram Data Analytics Project
===============================

A comprehensive data analysis project that analyzes Instagram post performance metrics
to uncover insights about content engagement, reach patterns, and audience behavior.

Author: Data Analyst
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InstagramAnalytics:
    """
    A comprehensive Instagram analytics class for analyzing post performance,
    engagement patterns, and predicting reach metrics.
    """
    
    def __init__(self, data_path):
        """Initialize the analytics class with data loading and preprocessing."""
        self.df = self.load_and_clean_data(data_path)
        self.setup_derived_metrics()
        
    def load_and_clean_data(self, data_path):
        """Load and clean the Instagram data."""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    print(f"✅ Data loaded successfully with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode the file with any encoding")
                
            # Basic data info
            print(f"📊 Dataset Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"🔍 Missing Values: {df.isnull().sum().sum()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def setup_derived_metrics(self):
        """Create derived metrics for better analysis."""
        if self.df is not None:
            # Engagement rate calculation
            self.df['Engagement_Rate'] = (
                (self.df['Likes'] + self.df['Comments'] + self.df['Shares']) / 
                self.df['Impressions'] * 100
            )
            
            # Conversion rate (Profile visits to follows)
            self.df['Conversion_Rate'] = np.where(
                self.df['Profile Visits'] > 0,
                (self.df['Follows'] / self.df['Profile Visits']) * 100,
                0
            )
            
            # Save rate
            self.df['Save_Rate'] = (self.df['Saves'] / self.df['Impressions']) * 100
            
            # Total reach sources
            self.df['Total_Reach_Sources'] = (
                self.df['From Home'] + self.df['From Hashtags'] + 
                self.df['From Explore'] + self.df['From Other']
            )
            
            print("✅ Derived metrics calculated successfully")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if self.df is None:
            return "❌ No data available"
        
        print("=" * 60)
        print("📈 INSTAGRAM ANALYTICS SUMMARY REPORT")
        print("=" * 60)
        
        # Basic statistics
        total_posts = len(self.df)
        total_impressions = self.df['Impressions'].sum()
        total_likes = self.df['Likes'].sum()
        total_comments = self.df['Comments'].sum()
        total_shares = self.df['Shares'].sum()
        total_saves = self.df['Saves'].sum()
        
        print(f"📊 Total Posts Analyzed: {total_posts:,}")
        print(f"👁️  Total Impressions: {total_impressions:,}")
        print(f"❤️  Total Likes: {total_likes:,}")
        print(f"💬 Total Comments: {total_comments:,}")
        print(f"🔄 Total Shares: {total_shares:,}")
        print(f"💾 Total Saves: {total_saves:,}")
        
        # Average metrics
        print(f"\n📊 AVERAGE METRICS PER POST:")
        print(f"👁️  Avg Impressions: {self.df['Impressions'].mean():.0f}")
        print(f"❤️  Avg Likes: {self.df['Likes'].mean():.0f}")
        print(f"💬 Avg Comments: {self.df['Comments'].mean():.1f}")
        print(f"🔄 Avg Shares: {self.df['Shares'].mean():.1f}")
        print(f"💾 Avg Saves: {self.df['Saves'].mean():.0f}")
        print(f"📈 Avg Engagement Rate: {self.df['Engagement_Rate'].mean():.2f}%")
        
        # Top performing posts
        print(f"\n🏆 TOP PERFORMING POSTS:")
        top_post = self.df.loc[self.df['Impressions'].idxmax()]
        print(f"Most Impressions: {top_post['Impressions']:,} impressions")
        print(f"Caption: {top_post['Caption'][:100]}...")
        
        # Reach source analysis
        home_total = self.df['From Home'].sum()
        hashtag_total = self.df['From Hashtags'].sum()
        explore_total = self.df['From Explore'].sum()
        other_total = self.df['From Other'].sum()
        
        print(f"\n🎯 REACH SOURCES:")
        print(f"🏠 From Home: {home_total:,} ({home_total/total_impressions*100:.1f}%)")
        print(f"# From Hashtags: {hashtag_total:,} ({hashtag_total/total_impressions*100:.1f}%)")
        print(f"🔍 From Explore: {explore_total:,} ({explore_total/total_impressions*100:.1f}%)")
        print(f"🌐 From Other: {other_total:,} ({other_total/total_impressions*100:.1f}%)")
        
        return "✅ Summary report generated successfully"
    
    def create_reach_analysis_chart(self):
        """Create an interactive reach source analysis chart."""
        # Calculate totals for each source
        sources = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
        totals = [self.df[source].sum() for source in sources]
        
        # Create pie chart
        fig = px.pie(
            values=totals,
            names=['Home', 'Hashtags', 'Explore', 'Other'],
            title='📊 Instagram Post Reach Sources Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Impressions: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            showlegend=True,
            height=500
        )
        
        fig.show()
        return fig
    
    def create_engagement_analysis(self):
        """Create comprehensive engagement analysis visualizations."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Impressions vs Likes', 'Engagement Rate Distribution', 
                          'Saves vs Impressions', 'Comments vs Shares'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Impressions vs Likes scatter
        fig.add_trace(
            go.Scatter(
                x=self.df['Impressions'],
                y=self.df['Likes'],
                mode='markers',
                name='Posts',
                marker=dict(size=8, opacity=0.6, color='blue'),
                hovertemplate='Impressions: %{x:,}<br>Likes: %{y:,}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Engagement rate histogram
        fig.add_trace(
            go.Histogram(
                x=self.df['Engagement_Rate'],
                name='Engagement Rate',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Saves vs Impressions
        fig.add_trace(
            go.Scatter(
                x=self.df['Impressions'],
                y=self.df['Saves'],
                mode='markers',
                name='Saves',
                marker=dict(size=8, opacity=0.6, color='red'),
                hovertemplate='Impressions: %{x:,}<br>Saves: %{y:,}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Comments vs Shares
        fig.add_trace(
            go.Scatter(
                x=self.df['Comments'],
                y=self.df['Shares'],
                mode='markers',
                name='Engagement',
                marker=dict(size=8, opacity=0.6, color='purple'),
                hovertemplate='Comments: %{x:,}<br>Shares: %{y:,}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📈 Instagram Engagement Analysis Dashboard",
            title_x=0.5,
            height=700,
            showlegend=False
        )
        
        fig.show()
        return fig
    
    def create_wordcloud_analysis(self):
        """Create word cloud from captions."""
        # Combine all captions
        text = " ".join(caption for caption in self.df['Caption'].astype(str))
        
        # Create word cloud
        stopwords = set(STOPWORDS)
        stopwords.update(['will', 'can', 'using', 'Python', 'data', 'learn', 'here'])
        
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            stopwords=stopwords,
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('☁️ Most Common Words in Instagram Captions', fontsize=20, pad=20)
        plt.tight_layout()
        plt.show()
        
        return wordcloud
    
    def correlation_analysis(self):
        """Perform correlation analysis on numerical features."""
        # Select numerical columns
        numeric_cols = ['Impressions', 'From Home', 'From Hashtags', 'From Explore', 
                       'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 
                       'Profile Visits', 'Follows', 'Engagement_Rate']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        plt.title('🔗 Instagram Metrics Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        # Print strongest correlations with Impressions
        print("\n🔗 STRONGEST CORRELATIONS WITH IMPRESSIONS:")
        impressions_corr = correlation_matrix['Impressions'].sort_values(ascending=False)
        for metric, corr in impressions_corr.items():
            if metric != 'Impressions':
                print(f"{metric}: {corr:.3f}")
        
        return correlation_matrix
    
    def build_prediction_model(self):
        """Build machine learning model to predict impressions."""
        # Prepare features
        feature_cols = ['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']
        X = self.df[feature_cols].fillna(0)
        y = self.df['Impressions']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print("\n🤖 MACHINE LEARNING MODEL PERFORMANCE:")
        print("=" * 50)
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name}:")
            print(f"  📊 R² Score: {r2:.4f}")
            print(f"  📈 RMSE: {np.sqrt(mse):.0f}")
            print()
        
        # Feature importance for Random Forest
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🎯 FEATURE IMPORTANCE (Random Forest):")
            for _, row in feature_importance.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return results
    
    def create_performance_dashboard(self):
        """Create a comprehensive performance dashboard."""
        # Calculate key metrics
        top_posts = self.df.nlargest(10, 'Impressions')
        
        # Create dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Top 10 Posts by Impressions',
                'Engagement Rate vs Impressions',
                'Monthly Performance Trend',
                'Reach Source Breakdown',
                'Save Rate Analysis',
                'Conversion Rate Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Top posts bar chart
        fig.add_trace(
            go.Bar(
                x=list(range(1, 11)),
                y=top_posts['Impressions'],
                name='Top Posts',
                marker_color='skyblue',
                hovertemplate='Post #%{x}<br>Impressions: %{y:,}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Engagement rate scatter
        fig.add_trace(
            go.Scatter(
                x=self.df['Impressions'],
                y=self.df['Engagement_Rate'],
                mode='markers',
                name='Engagement',
                marker=dict(size=6, opacity=0.6, color='green'),
                hovertemplate='Impressions: %{x:,}<br>Engagement Rate: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add more visualizations...
        fig.update_layout(
            title_text="📊 Instagram Performance Dashboard",
            title_x=0.5,
            height=1000,
            showlegend=False
        )
        
        fig.show()
        return fig

def main():
    """Main function to run the Instagram analytics."""
    print("🚀 Starting Instagram Analytics Project")
    print("=" * 50)
    
    # Initialize analytics
    analytics = InstagramAnalytics("Instagram data.csv")
    
    if analytics.df is not None:
        # Generate summary report
        analytics.generate_summary_report()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        analytics.create_reach_analysis_chart()
        analytics.create_engagement_analysis()
        analytics.create_wordcloud_analysis()
        
        # Perform correlation analysis
        print("\n🔍 Performing correlation analysis...")
        analytics.correlation_analysis()
        
        # Build prediction models
        print("\n🤖 Building prediction models...")
        analytics.build_prediction_model()
        
        # Create performance dashboard
        print("\n📈 Creating performance dashboard...")
        analytics.create_performance_dashboard()
        
        print("\n✅ Analysis completed successfully!")
        print("📋 Check the generated visualizations and insights above.")
        
    else:
        print("❌ Failed to load data. Please check the file path and format.")

if __name__ == "__main__":
    main()