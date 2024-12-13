import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import ccf
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class CryptoAnalyzer:
    def __init__(self, df):
        """
        Initialize the analyzer with a DataFrame containing Bitcoin and indicator data.
        
        Parameters:
        df (pandas.DataFrame): DataFrame with columns as specified in the requirements
        """
        self.df = df.copy()
        self.price_columns = [
            'BTC_Weekly_Avg_Price', 'BTC_Close_Price', 'BTC_Open_Price',
            'BTC_Weekly_Pct_Change', 'BTC_Prior_Week_Pct_Change', 
            'BTC_Weekly_Avg_Volume'
        ]
        self.indicator_columns = [
            'Weekly_Avg_Fear_Greed', 'Fear_Greed_Weekly_Change',
            'Fear_Greed_Prior_Week_Change', 'M2SL', 'M2_Monthly_Pct_Change',
            'Funding_Rate'
        ]
        
    def prepare_data(self):
        """Prepare the data for analysis by handling missing values and scaling."""
        # Create date index (keeping existing date creation code)
        self.df['Year'] = self.df['Year'].fillna(method='ffill').fillna(method='bfill')
        self.df['Week'] = self.df['Week'].fillna(method='ffill').fillna(method='bfill')
        
        self.df['Date'] = pd.to_datetime(
            self.df['Year'].round().astype(int).astype(str) + '-W' + 
            self.df['Week'].round().astype(int).astype(str).str.zfill(2) + '-1',
            format='%Y-W%W-%w'
        )
        self.df.set_index('Date', inplace=True)
        
        # Create lagged variables for BTC metrics
        for col in self.price_columns:
            self.df[f'{col}_Lag1'] = self.df[col].shift(1)  # 1-week lag
            self.df[f'{col}_Lag2'] = self.df[col].shift(2)  # 2-week lag
            self.df[f'{col}_Lag3'] = self.df[col].shift(3)  # 3-week lag
            self.df[f'{col}_Lag4'] = self.df[col].shift(4)  # 4-week lag
        
        # Update indicator columns list to include lagged variables
        self.lagged_price_columns = [f'{col}_Lag{i}' 
                                    for col in self.price_columns 
                                    for i in range(1, 5)]
        
        # Handle missing values for all columns
        all_columns = (self.price_columns + self.indicator_columns + 
                    self.lagged_price_columns)
        for col in all_columns:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Scale the data
        scaler = StandardScaler()
        scaled_cols = all_columns
        self.df_scaled = pd.DataFrame(
            scaler.fit_transform(self.df[scaled_cols]),
            columns=scaled_cols,
            index=self.df.index
        )
        
        return self
    
    def seasonal_analysis(self):
        """Analyze and decompose seasonal patterns in the price data."""
        # Perform seasonal decomposition on weekly average price
        decomp = seasonal_decompose(
            self.df['BTC_Weekly_Avg_Price'],
            period=52,  # 52 weeks in a year
            extrapolate_trend='freq'
        )
        
        self.seasonal_components = {
            'trend': decomp.trend,
            'seasonal': decomp.seasonal,
            'residual': decomp.resid
        }
        
        return self.seasonal_components
    
    def analyze_correlations(self):
        """Calculate and analyze various types of correlations including lagged variables."""
        # Update to include lagged variables in correlation analysis
        all_indicators = self.indicator_columns + self.lagged_price_columns
        
        # Linear correlations (Pearson)
        pearson_corr = pd.DataFrame(
            index=self.price_columns,
            columns=all_indicators
        )
        
        # Calculate correlations
        for price_col in self.price_columns:
            for indicator_col in all_indicators:
                # Skip if comparing current price to its own lags
                if indicator_col.startswith(price_col + '_Lag'):
                    continue
                    
                # Pearson correlation
                pearson = stats.pearsonr(
                    self.df_scaled[price_col],
                    self.df_scaled[indicator_col]
                )
                pearson_corr.loc[price_col, indicator_col] = pearson[0]
        
        self.correlation_results = {
            'pearson': pearson_corr
        }
        
        return self.correlation_results
    
    def analyze_lagged_relationships(self, max_lag=4):
        """Analyze leading/lagging relationships between prices and indicators."""
        lag_correlations = {}
        
        for price_col in ['BTC_Weekly_Avg_Price', 'BTC_Weekly_Pct_Change']:
            lag_correlations[price_col] = {}
            
            for indicator_col in self.indicator_columns:
                # Calculate cross-correlation function
                ccf_result = ccf(
                    self.df_scaled[price_col],
                    self.df_scaled[indicator_col],
                    adjusted=False
                )
                
                # Store relevant lags
                lag_correlations[price_col][indicator_col] = {
                    'ccf': ccf_result[:max_lag+1],
                    'optimal_lag': np.argmax(np.abs(ccf_result[:max_lag+1]))
                }
        
        self.lag_results = lag_correlations
        return self.lag_results
    
    def analyze_combined_effects(self):
        """Analyze combined effects of multiple indicators."""
        # Focus on price percentage changes
        target = 'BTC_Weekly_Pct_Change'
        
        # Calculate conditional correlations
        conditional_corr = {}
        
        for ind1 in self.indicator_columns:
            for ind2 in self.indicator_columns:
                if ind1 >= ind2:
                    continue
                    
                # Split data based on first indicator's values
                high_mask = self.df_scaled[ind1] > self.df_scaled[ind1].median()
                
                # Calculate correlations in each regime
                high_corr = stats.pearsonr(
                    self.df_scaled[target][high_mask],
                    self.df_scaled[ind2][high_mask]
                )[0]
                
                low_corr = stats.pearsonr(
                    self.df_scaled[target][~high_mask],
                    self.df_scaled[ind2][~high_mask]
                )[0]
                
                conditional_corr[f"{ind1}_high_{ind2}"] = high_corr
                conditional_corr[f"{ind1}_low_{ind2}"] = low_corr
        
        self.combined_effects = conditional_corr
        return self.combined_effects
    
    def visualize_results(self):
        """Create visualizations of the analysis results with lagged variables."""
        plt.figure(figsize=(15, 10))  # Increased figure size for more variables
        
        correlation_matrix = self.correlation_results['pearson'].to_numpy().astype(float)
        im = plt.imshow(
            correlation_matrix,
            cmap='RdBu',
            aspect='auto',
            vmin=-1,
            vmax=1
        )
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add labels with smaller font and increased rotation for readability
        plt.xticks(range(len(self.indicator_columns + self.lagged_price_columns)), 
                self.indicator_columns + self.lagged_price_columns, 
                rotation=90, 
                fontsize=8)
        plt.yticks(range(len(self.price_columns)), 
                self.price_columns, 
                fontsize=8)
        
        # Add correlation values as text (only for significant correlations)
        for i in range(len(self.price_columns)):
            for j in range(len(self.indicator_columns + self.lagged_price_columns)):
                value = correlation_matrix[i, j]
                # Only show text for correlations above 0.3 or below -0.3
                if abs(value) >= 0.3:
                    plt.text(j, i, f'{value:.2f}',
                            ha='center', va='center',
                            fontsize=6,
                            color='white' if abs(value) > 0.5 else 'black')
        
        plt.title('Pearson Correlations: Bitcoin Metrics vs Indicators (Including Lagged Variables)')
        plt.tight_layout()
        
        return plt

    def generate_report(self):
        """Generate a summary report of the analysis findings."""
        report = {
            'strongest_correlations': {},
            'leading_indicators': {},
            'combined_effects': {}
        }
        
        # Find strongest correlations
        pearson_corr = self.correlation_results['pearson']
        max_corr = pearson_corr.max().max()
        max_corr_pair = np.where(pearson_corr == max_corr)
        report['strongest_correlations']['linear'] = {
            'indicator': pearson_corr.columns[max_corr_pair[1][0]],
            'price_metric': pearson_corr.index[max_corr_pair[0][0]],
            'correlation': max_corr
        }
        
        # Remove the seasonal patterns section since we're not using it
        
        # Identify strongest combined effects
        if hasattr(self, 'combined_effects'):
            max_cond_corr = max(self.combined_effects.values())
            max_cond_pair = max(self.combined_effects.items(), key=lambda x: abs(x[1]))
            report['combined_effects']['strongest'] = {
                'pair': max_cond_pair[0],
                'correlation': max_cond_pair[1]
            }
        
        return report

def example_usage():
    """Example usage of the CryptoAnalyzer class."""
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='W')
    sample_data = pd.DataFrame({
        'Year': dates.year,
        'Week': dates.isocalendar().week,
        'BTC_Weekly_Avg_Price': np.random.normal(40000, 5000, len(dates)),
        'BTC_Close_Price': np.random.normal(40000, 5000, len(dates)),
        'BTC_Open_Price': np.random.normal(40000, 5000, len(dates)),
        'BTC_Weekly_Pct_Change': np.random.normal(0, 0.05, len(dates)),
        'BTC_Prior_Week_Pct_Change': np.random.normal(0, 0.05, len(dates)),
        'BTC_Weekly_Avg_Volume': np.random.normal(1000000, 100000, len(dates)),
        'Weekly_Avg_Fear_Greed': np.random.normal(50, 15, len(dates)),
        'Fear_Greed_Weekly_Change': np.random.normal(0, 5, len(dates)),
        'Fear_Greed_Prior_Week_Change': np.random.normal(0, 5, len(dates)),
        'M2SL': np.random.normal(20000, 1000, len(dates)),
        'M2_Monthly_Pct_Change': np.random.normal(0.002, 0.0005, len(dates)),
        'Funding_Rate': np.random.normal(0.001, 0.0002, len(dates))
    })
    
    # Initialize and run analysis
    analyzer = CryptoAnalyzer(sample_data)
    analyzer.prepare_data()
    analyzer.analyze_correlations()
    analyzer.analyze_lagged_relationships()
    analyzer.analyze_combined_effects()
    
    # Generate visualizations and report
    analyzer.visualize_results()
    report = analyzer.generate_report()
    
    return analyzer, report