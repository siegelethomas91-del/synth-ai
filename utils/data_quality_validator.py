import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import streamlit as st
from typing import Dict, Any, Optional

class DataQualityValidator:
    def __init__(self, synthetic_data: pd.DataFrame):
        self.synthetic_data = synthetic_data
    
    def compare_with_real_data(self, real_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Compare synthetic data with real data or benchmarks"""
        comparison_results = {}
        
        # Compare with uploaded real data
        if real_data is not None and not real_data.empty:
            comparison_results['uploaded'] = self._compare_with_real(real_data)
            
        # Compare with industry benchmarks
        comparison_results['benchmarks'] = self._compare_with_benchmarks()
        
        return comparison_results
    
    def _compare_with_real(self, real_data: pd.DataFrame) -> Dict[str, float]:
        """Compare synthetic data with uploaded real data"""
        metrics = {}
        
        # Compare numerical columns
        for col in self.synthetic_data.select_dtypes(include=[np.number]).columns:
            if col in real_data.columns:
                # KS test for distribution similarity
                statistic, pvalue = stats.ks_2samp(
                    self.synthetic_data[col], 
                    real_data[col]
                )
                metrics[f'{col}_ks_statistic'] = statistic
                metrics[f'{col}_ks_pvalue'] = pvalue
                
                # Mean and std differences
                metrics[f'{col}_mean_diff'] = abs(
                    self.synthetic_data[col].mean() - real_data[col].mean()
                )
                metrics[f'{col}_std_diff'] = abs(
                    self.synthetic_data[col].std() - real_data[col].std()
                )
        
        return metrics
    
    def _compare_with_benchmarks(self) -> Dict[str, float]:
        """Compare synthetic data with industry benchmarks"""
        metrics = {}
        
        # Define benchmark statistics
        benchmark_stats = {
            'transaction_amounts': {
                'mean': 500,
                'std': 1000
            },
            'fraud_rate': 0.002  # Industry standard ~0.2%
        }
        
        if 'amount' in self.synthetic_data.columns:
            metrics['amount_mean_diff'] = abs(
                self.synthetic_data['amount'].mean() - benchmark_stats['transaction_amounts']['mean']
            )
        
        if 'is_fraud' in self.synthetic_data.columns:
            synthetic_fraud_rate = self.synthetic_data['is_fraud'].mean()
            metrics['fraud_rate_diff'] = abs(synthetic_fraud_rate - benchmark_stats['fraud_rate'])
        
        return metrics
    
    def display_quality_report(self):
        """Display comprehensive quality report in Streamlit"""
        st.write("### Data Quality Metrics")
        
        # Basic statistics
        st.write("#### Basic Statistics")
        st.dataframe(self.synthetic_data.describe())
        
        # Distribution plots
        st.write("#### Distribution Analysis")
        for col in self.synthetic_data.select_dtypes(include=[np.number]).columns:
            fig = px.histogram(
                self.synthetic_data, 
                x=col,
                title=f'{col} Distribution'
            )
            st.plotly_chart(fig)
        
        # Correlation matrix
        st.write("#### Feature Correlations")
        corr_matrix = self.synthetic_data.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix"
        )
        st.plotly_chart(fig)