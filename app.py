import streamlit as st
import pandas as pd
import numpy as np
from synthetic_finance_data_generator import ParallelMemoryAugmentedCTGAN, generate_synthetic_finance_data
import plotly.express as px
from datetime import datetime
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os
import json
from time import sleep
import asyncio

# Instead of direct async calls, use this wrapper
def async_wrapper(async_func):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_func)
    loop.close()
    return result

def create_analysis_notebook(data, output_path="data_analysis_report.ipynb"):
    """Create a Jupyter notebook with data analysis"""
    nb = new_notebook()
    
    # Add markdown cells
    nb['cells'] = [
        new_markdown_cell("# Synthetic Financial Data Analysis Report"),
        new_markdown_cell("## Data Overview"),
        new_code_cell("import pandas as pd\nimport numpy as np\nimport plotly.express as px\n"
                     "df = pd.read_csv('synthetic_finance_data_ctgan.csv')"),
        new_code_cell("df.head()"),
        new_code_cell("df.describe()"),
        new_markdown_cell("## Distribution Analysis"),
        new_code_cell("px.histogram(df, x='amount', title='Transaction Amount Distribution').show()"),
        new_code_cell("px.box(df, x='transaction_type', y='amount', title='Amount by Transaction Type').show()"),
        new_markdown_cell("## Fraud Analysis"),
        new_code_cell("fraud_dist = df['is_fraud'].value_counts(normalize=True)\n"
                     "px.pie(values=fraud_dist.values, names=fraud_dist.index, "
                     "title='Fraud Distribution').show()"),
    ]
    
    # Save notebook
    with open(output_path, 'w') as f:
        nbformat.write(nb, f)

def main():
    st.set_page_config(
        page_title="SynthAI",
        page_icon="🤖",
        layout="wide"
    )
    
    # Title with custom styling
    st.markdown("""
        <h1 style='text-align: center; color: #2E86C1;'>
            SynthAI: Advanced Synthetic Data Generator
        </h1>
        <p style='text-align: center; color: #666666;'>
            Powered by Memory-Augmented CTGAN
        </p>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("Configuration")
    
    # Sample size selection
    num_samples = st.sidebar.number_input(
        "Number of samples to generate",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=1000
    )
    
    # Date range selection
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2020, 1, 1)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime(2023, 12, 31)
    )
    
    # Feature selection
    st.sidebar.header("Feature Selection")
    available_features = {
        'amount': True,
        'transaction_type': True,
        'merchant_category': True,
        'bank_type': True,
        'city': True,
        'customer_age': True,
        'customer_tenure': True,
        'transaction_frequency': True,
        'credit_score': True,
        'is_fraud': True
    }
    
    selected_features = {}
    for feature, default in available_features.items():
        selected_features[feature] = st.sidebar.checkbox(
            f"Include {feature}",
            value=default
        )
    
    # Advanced settings
    st.sidebar.header("Advanced Settings")
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    memory_size = st.sidebar.number_input(
        "Memory Size",
        min_value=1000,
        max_value=100000,
        value=50000,
        step=1000
    )
    
    if st.button("Generate Data"):
        # Create placeholder for progress bar and status
        progress_bar = st.progress(0)
        status_container = st.empty()
        info_container = st.empty()
        
        try:
            params = {
                'num_samples': num_samples,
                'batch_size': batch_size,
                'start_date': start_date,
                'end_date': end_date,
                'selected_features': selected_features,
                'memory_size': memory_size
            }
            
            # Custom progress callback
            def progress_callback(stage, progress, message):
                if stage == "training":
                    progress_bar.progress(progress * 0.6)  # Training takes 60% of progress
                    status_container.info(f"Training CTGAN: {progress*100:.1f}%")
                    info_container.markdown(f"""
                        **Current Status:**
                        - Stage: Training CTGAN Model
                        - Message: {message}
                        - Epochs Progress: {progress*100:.1f}%
                    """)
                elif stage == "generating":
                    progress_bar.progress(0.6 + progress * 0.4)  # Generation takes 40%
                    status_container.info(f"Generating Data: {progress*100:.1f}%")
                    info_container.markdown(f"""
                        **Current Status:**
                        - Stage: Generating Synthetic Data
                        - Message: {message}
                        - Generation Progress: {progress*100:.1f}%
                    """)
            
            # Add callback to params
            params['progress_callback'] = progress_callback
            
            # Generate data
            data = generate_synthetic_finance_data(**params)
            
            # Clear progress indicators
            progress_bar.empty()
            status_container.empty()
            info_container.empty()
            
            # Success message with stats
            st.success(f"""
                ✅ Successfully generated {len(data):,} records!
                - Features: {len(data.columns)} columns
                - Memory Used: {data.memory_usage().sum() / 1024**2:.1f} MB
            """)
            
            # Display sample of generated data
            st.subheader("Sample of Generated Data")
            st.dataframe(data.head())
            
            # Basic visualizations
            st.subheader("Data Visualizations")
            
            # Amount distribution
            fig_amount = px.histogram(data, x='amount', title='Transaction Amount Distribution')
            st.plotly_chart(fig_amount)
            
            # Fraud distribution
            fig_fraud = px.pie(data, names='is_fraud', title='Fraud Distribution')
            st.plotly_chart(fig_fraud)
            
            # Export options
            st.subheader("Export Options")
            
            if st.button("Generate Analysis Report"):
                create_analysis_notebook(data)
                st.success("Analysis report generated as 'data_analysis_report.ipynb'!")
            
            if st.button("Export to CSV"):
                data.to_csv("synthetic_finance_data_export.csv", index=False)
                st.success("Data exported to 'synthetic_finance_data_export.csv'!")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()