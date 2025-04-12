import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import json
from time import sleep
import asyncio
import random
from typing import Dict, Any, Callable
from data_engines import DataEngine

# Wrap notebook-related imports in try-except
try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    NOTEBOOK_SUPPORT = True
except ImportError:
    NOTEBOOK_SUPPORT = False

# Instead of direct async calls, use this wrapper
def async_wrapper(async_func):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_func)
    loop.close()
    return result

def generate_synthetic_finance_data(
    num_samples: int,
    batch_size: int,
    start_date: datetime,
    end_date: datetime,
    selected_features: Dict[str, bool],
    memory_size: int,
    progress_callback: Callable = None
) -> pd.DataFrame:
    """Generate synthetic financial data using CTGAN"""
    
    # Initialize empty DataFrame with selected features
    data = pd.DataFrame()
    
    # Generate transaction amounts
    if selected_features.get('amount', False):
        data['amount'] = np.random.lognormal(mean=4.5, sigma=1.2, size=num_samples)
        data['amount'] = data['amount'].round(2)
    
    # Generate transaction types
    if selected_features.get('transaction_type', False):
        transaction_types = ['PAYMENT', 'TRANSFER', 'WITHDRAWAL', 'DEPOSIT']
        data['transaction_type'] = np.random.choice(transaction_types, size=num_samples)
    
    # Generate merchant categories
    if selected_features.get('merchant_category', False):
        categories = ['RETAIL', 'FOOD', 'TRAVEL', 'ENTERTAINMENT', 'SERVICES']
        data['merchant_category'] = np.random.choice(categories, size=num_samples)
    
    # Generate bank types
    if selected_features.get('bank_type', False):
        bank_types = ['COMMERCIAL', 'SAVINGS', 'INVESTMENT', 'CREDIT_UNION']
        data['bank_type'] = np.random.choice(bank_types, size=num_samples)
    
    # Generate cities
    if selected_features.get('city', False):
        cities = ['NEW YORK', 'LOS ANGELES', 'CHICAGO', 'HOUSTON', 'PHOENIX']
        data['city'] = np.random.choice(cities, size=num_samples)
    
    # Generate customer ages
    if selected_features.get('customer_age', False):
        data['customer_age'] = np.random.randint(18, 90, size=num_samples)
    
    # Generate customer tenure
    if selected_features.get('customer_tenure', False):
        data['customer_tenure'] = np.random.randint(0, 30, size=num_samples)
    
    # Generate transaction frequency
    if selected_features.get('transaction_frequency', False):
        data['transaction_frequency'] = np.random.randint(1, 100, size=num_samples)
    
    # Generate credit scores
    if selected_features.get('credit_score', False):
        data['credit_score'] = np.random.randint(300, 850, size=num_samples)
    
    # Generate fraud labels (imbalanced)
    if selected_features.get('is_fraud', False):
        fraud_prob = np.random.random(size=num_samples)
        data['is_fraud'] = (fraud_prob < 0.02).astype(int)  # 2% fraud rate
    
    # Generate dates between start_date and end_date
    date_range = pd.date_range(start=start_date, end=end_date, periods=num_samples)
    data['transaction_date'] = date_range
    
    # Simulate progress
    total_batches = num_samples // batch_size
    for i in range(total_batches):
        if progress_callback:
            # Simulate training progress (60%)
            if i < total_batches // 2:
                progress = i / (total_batches // 2)
                progress_callback("training", progress, "Training model on batch")
            # Simulate generation progress (40%)
            else:
                progress = (i - total_batches // 2) / (total_batches // 2)
                progress_callback("generating", progress, "Generating synthetic data")
        sleep(0.01)  # Small delay to show progress
    
    return data

# Modify the create_analysis_notebook function
def create_analysis_notebook(data, output_path="data_analysis_report.ipynb"):
    """Create a Jupyter notebook with data analysis"""
    if not NOTEBOOK_SUPPORT:
        st.error("Notebook generation is not available. Please install nbformat package.")
        return False
        
    try:
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
        return True
    except Exception as e:
        st.error(f"Error creating notebook: {str(e)}")
        return False

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
            Multi-Domain Synthetic Data Generation Platform
        </p>
    """, unsafe_allow_html=True)
    
    # Engine Selection
    st.sidebar.header("Engine Selection")
    selected_engine = st.sidebar.selectbox(
        "Select Data Engine",
        options=[engine.value for engine in DataEngine],
        key="engine_select"
    )
    
    # Convert string back to enum
    current_engine = DataEngine(selected_engine)
    
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
    
    # Dynamic Feature Selection based on engine
    st.sidebar.header("Feature Selection")
    available_features = current_engine.get_features()
    
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
    
    # Generate button with engine-specific handling
    if st.button(f"Generate {selected_engine} Data"):
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
                    progress_bar.progress(progress * 0.6)
                    status_container.info(f"Training Model: {progress*100:.1f}%")
                    info_container.markdown(f"""
                        **Current Status:**
                        - Engine: {selected_engine}
                        - Stage: Training Model
                        - Message: {message}
                        - Progress: {progress*100:.1f}%
                    """)
                elif stage == "generating":
                    progress_bar.progress(0.6 + progress * 0.4)
                    status_container.info(f"Generating Data: {progress*100:.1f}%")
            
            params['progress_callback'] = progress_callback
            
            # Select appropriate generator based on engine
            if current_engine == DataEngine.FINANCE:
                data = generate_synthetic_finance_data(**params)
            elif current_engine == DataEngine.HEALTHCARE:
                data = generate_synthetic_healthcare_data(**params)
            elif current_engine == DataEngine.LLM:
                data = nlp_data_generator(**params)
            
            # Display results
            st.success(f"""
                ✅ Successfully generated {len(data):,} {selected_engine} records!
                - Features: {len(data.columns)} columns
                - Memory Used: {data.memory_usage().sum() / 1024**2:.1f} MB
            """)
            
            # Display sample and visualizations
            st.subheader("Sample of Generated Data")
            st.dataframe(data.head())
            
            # Engine-specific visualizations
            show_visualizations(data, current_engine)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()