import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime
import torch
import torch.nn as nn
from ctgan import CTGAN
from memory_profiler import profile
import gc
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import os
import joblib
from datetime import datetime
from ctgan.data_transformer import DataTransformer
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

def generate_batch(batch_params):
    """Helper function to generate a single batch of data with improved distributions"""
    batch_size, seed, features = batch_params
    np.random.seed(seed)
    
    # Define more realistic category weights
    transaction_types = {
        'Debit Card': 0.45,
        'Credit Card': 0.25,
        'UPI': 0.20,
        'Net Banking': 0.07,
        'RTGS/NEFT': 0.03
    }
    
    merchant_categories = {
        'Retail': 0.35,
        'Grocery': 0.25,
        'Online Shopping': 0.20,
        'Entertainment': 0.15,
        'Travel': 0.05
    }
    
    # Time-based patterns
    hour_of_day = np.random.normal(14, 4, batch_size)  # More transactions during business hours
    day_of_week = np.random.choice(range(7), size=batch_size, 
                                 p=[0.12, 0.15, 0.15, 0.15, 0.15, 0.18, 0.10])  # Weekend pattern
    
    data_batch = {
        'amount': generate_realistic_amounts(batch_size),
        'transaction_type': np.random.choice(
            list(transaction_types.keys()),
            size=batch_size,
            p=list(transaction_types.values())
        ),
        'merchant_category': np.random.choice(
            list(merchant_categories.keys()),
            size=batch_size,
            p=list(merchant_categories.values())
        ),
        'bank_type': generate_bank_types(batch_size),
        'city': generate_city_distribution(batch_size),
        'customer_age': generate_age_distribution(batch_size),
        'customer_tenure': None,  # Will be set based on age
        'transaction_frequency': None,  # Will be set based on age and credit score
        'credit_score': generate_credit_score_distribution(batch_size),
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_fraud': generate_fraud_cases(batch_size)
    }
    
    # Convert to DataFrame for relationship validation
    df = pd.DataFrame(data_batch)
    df = validate_relationships(df)
    
    if features:
        df = df[[k for k, v in features.items() if v]]
    
    return df

def generate_realistic_amounts(size):
    """Generate realistic transaction amounts with multiple distributions"""
    # Mix of different amount patterns
    small_transactions = np.random.lognormal(3, 0.5, size=int(size * 0.6))  # Regular purchases
    medium_transactions = np.random.lognormal(5, 0.7, size=int(size * 0.3))  # Larger purchases
    large_transactions = np.random.lognormal(8, 1.0, size=size - int(size * 0.9))  # Big tickets
    
    amounts = np.concatenate([small_transactions, medium_transactions, large_transactions])
    np.random.shuffle(amounts)
    return np.round(amounts, 2)[:size]

def generate_age_distribution(size):
    """Generate realistic age distribution"""
    # Multi-modal distribution for different age groups
    young = np.random.normal(25, 3, size=int(size * 0.3))
    middle = np.random.normal(40, 5, size=int(size * 0.5))
    senior = np.random.normal(60, 5, size=size - int(size * 0.8))
    
    ages = np.concatenate([young, middle, senior])
    np.random.shuffle(ages)
    return np.clip(ages, 18, 90).astype(int)[:size]

def validate_relationships(data):
    """Validate and adjust data relationships"""
    # Age-Credit Score Relationship
    data['credit_score'] = data.apply(lambda row: 
        adjust_credit_score(row['customer_age'], row['credit_score']), axis=1)
    
    # Age-Tenure Relationship
    data['customer_tenure'] = data.apply(lambda row:
        min(row['customer_tenure'], row['customer_age'] - 18), axis=1)
    
    # Amount-Merchant Category Relationship
    data['amount'] = data.apply(lambda row:
        adjust_amount_by_category(row['amount'], row['merchant_category']), axis=1)
    
    # Transaction Frequency based on Age and Credit Score
    data['transaction_frequency'] = data.apply(lambda row:
        adjust_transaction_frequency(row['customer_age'], row['credit_score']), axis=1)
    
    return data

def adjust_credit_score(age, score):
    """Adjust credit score based on age"""
    if age < 25:
        return min(score, 750)  # Young customers rarely have perfect scores
    elif age > 60:
        return max(score, 600)  # Older customers usually maintain baseline scores
    return score

def adjust_amount_by_category(amount, category):
    """Adjust transaction amount based on merchant category"""
    category_multipliers = {
        'Retail': 1.0,
        'Travel': 2.5,
        'Entertainment': 1.2,
        'Grocery': 0.5,
        'Online Shopping': 0.8
    }
    return amount * category_multipliers.get(category, 1.0)

def adjust_transaction_frequency(age, credit_score):
    """Adjust transaction frequency based on age and credit score"""
    base_frequency = np.random.poisson(5)
    age_factor = min(1.5, max(0.5, age / 40))
    credit_factor = min(1.5, max(0.5, credit_score / 700))
    return int(base_frequency * age_factor * credit_factor)

# Modify the ParallelMemoryAugmentedCTGAN class to handle training properly
class ParallelMemoryAugmentedCTGAN(CTGAN):
    def __init__(self, memory_size=1000, n_jobs=-1, model_path='trained_ctgan_model.pkl', **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_bank = None
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.model_path = model_path
        self._progress_callback = None

    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Override fit method to include progress tracking"""
        if epochs is None:
            epochs = self._epochs
        
        # Validate discrete columns
        self._validate_discrete_columns(train_data, discrete_columns)
        
        # Transform data for training
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        
        # Train the model
        for epoch in range(epochs):
            if self._progress_callback:
                progress = epoch / epochs
                self._progress_callback("training", progress, f"Training epoch {epoch}/{epochs}")
            
            # Call the parent class's fit method for each epoch
            super().fit(train_data, [], epochs=1)  # Already transformed data
        
        return self

    def sample(self, n):
        """Override sample method to handle transformed data"""
        sampled = super().sample(n)
        return self.transformer.inverse_transform(sampled)

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Validate that discrete columns exist in the data"""
        if len(discrete_columns) > 0:
            if not all(column in train_data.columns for column in discrete_columns):
                raise ValueError(
                    "All discrete columns must exist in the training data. "
                    f"Missing columns: {set(discrete_columns) - set(train_data.columns)}"
                )

    def set_progress_callback(self, callback):
        """Set the progress callback function"""
        self._progress_callback = callback
    
    def save_model(self, path=None):
        """Save the trained model to disk with error handling"""
        try:
            save_path = path or self.model_path
            # Create a copy of the model without the callback
            model_copy = ParallelMemoryAugmentedCTGAN(
                memory_size=self.memory_size,
                n_jobs=self.n_jobs,
                model_path=self.model_path
            )
            model_copy.__dict__.update({
                k: v for k, v in self.__dict__.items() 
                if k != '_progress_callback'
            })
            
            # Save the copy
            joblib.dump(model_copy, save_path, compress=3)
            print(f"Model saved successfully to {save_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False

    @classmethod
    def load_model(cls, path):
        """Load a trained model from disk with error handling"""
        try:
            if not os.path.exists(path):
                print(f"No model file found at {path}")
                return None
            
            model = joblib.load(path)
            print(f"Model loaded successfully from {path}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            if os.path.exists(path):
                print(f"Removing corrupted model file: {path}")
                os.remove(path)
            return None
        
    def parallel_sample(self, n, batch_size=1000):
        """Sample data in parallel"""
        num_batches = (n + batch_size - 1) // batch_size
        batch_sizes = [min(batch_size, n - i * batch_size) for i in range(num_batches)]
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self.sample, size) for size in batch_sizes]
            samples = [future.result() for future in futures]
        
        return pd.concat(samples, ignore_index=True)

@profile
def generate_synthetic_finance_data(
    num_samples=100_000,
    batch_size=5000,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    selected_features=None,
    memory_size=10000,
    n_jobs=-1,
    progress_callback=None,
    model_path='trained_ctgan_model.pkl',
    force_retrain=False,
    real_data_path=None  # Add this parameter
):
    """Modified function with robust model saving/loading capability"""
    if progress_callback is None:
        def progress_callback(stage, progress, message):
            if stage != "training":
                print(f"{stage.capitalize()}: {message} ({progress*100:.1f}%)")
    
    # Try to load existing model
    ctgan = None
    if not force_retrain:
        progress_callback("loading", 0.0, "Attempting to load pre-trained model...")
        ctgan = ParallelMemoryAugmentedCTGAN.load_model(model_path)
        if ctgan is not None:
            progress_callback("loading", 1.0, "Model loaded successfully!")

    if ctgan is None:
        # Training new model
        progress_callback("training", 0.0, "Starting new model training...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        initial_samples = 25_000  # Reduced from 100_000
        
        print(f"Generating initial training data using {n_jobs} processes...")
        
        # Prepare batch parameters for parallel processing
        batch_params = [
            (min(batch_size, initial_samples - i), 42 + i, selected_features)
            for i in range(0, initial_samples, batch_size)
        ]
        
        # Generate initial training data in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            data_batches = list(tqdm(
                executor.map(generate_batch, batch_params),
                total=len(batch_params)
            ))
        
        train_df = pd.concat(data_batches, ignore_index=True)
        del data_batches
        gc.collect()

        # During CTGAN training, update progress
        class ProgressCallback:
            def __init__(self, epochs):
                self.epochs = epochs
                self.current = 0
                
            def __call__(self, epoch, *args):
                self.current = epoch
                progress = epoch / self.epochs
                progress_callback("training", progress, f"Training epoch {epoch}/{self.epochs}")
        
        # Initialize and train Parallel Memory-Augmented CTGAN with fewer epochs
        print("Training Parallel Memory-Augmented CTGAN model...")
        epochs = 100
        progress_tracker = ProgressCallback(epochs)
        
        # Initialize CTGAN with smaller epochs for faster training
        ctgan = ParallelMemoryAugmentedCTGAN(
            memory_size=memory_size,
            epochs=50,  # Reduced epochs
            batch_size=500,
            verbose=True,
            n_jobs=n_jobs,
            model_path=model_path,
            embedding_dim=128,  # Added for better stability
            generator_dim=(256, 256),
            discriminator_dim=(256, 256)
        )
        
        # Set the progress callback
        ctgan.set_progress_callback(progress_callback)
        
        # Fit the model with discrete columns
        discrete_columns = [
            'transaction_type', 
            'merchant_category',
            'bank_type',
            'city',
            'is_fraud'
        ]
        
        try:
            ctgan.fit(train_df, discrete_columns=discrete_columns)
            print("Model training completed successfully!")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        
        # Save model with error handling
        if not ctgan.save_model(model_path):
            print("Warning: Failed to save model, but continuing with data generation...")

    # Generate synthetic data using loaded/trained model
    progress_callback("generating", 0.0, "Generating synthetic data...")
    
    # Generate synthetic data in parallel batches
    print(f"Generating {num_samples} synthetic samples in parallel...")
    total_batches = (num_samples + batch_size - 1) // batch_size
    for i in range(total_batches):
        current_batch = min(batch_size, num_samples - i * batch_size)
        progress = i / total_batches
        progress_callback("generating", progress, 
                        f"Generating batch {i+1}/{total_batches} ({current_batch} samples)")
        synthetic_data = ctgan.parallel_sample(num_samples, batch_size=batch_size)
    
    # Post-process the data
    print("Post-processing generated data...")
    synthetic_data = validate_relationships(synthetic_data)
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Post-process columns in parallel
        futures = []
        futures.append(executor.submit(lambda: synthetic_data['amount'].clip(lower=0).round(2)))
        futures.append(executor.submit(lambda: synthetic_data['customer_age'].clip(18, 90).astype(int)))
        futures.append(executor.submit(lambda: synthetic_data['customer_tenure'].clip(0, 30).astype(int)))
        futures.append(executor.submit(lambda: synthetic_data['credit_score'].clip(300, 900).astype(int)))
        futures.append(executor.submit(lambda: synthetic_data['transaction_frequency'].clip(0).astype(int)))
        
        # Update columns with processed results
        synthetic_data['amount'] = futures[0].result()
        synthetic_data['customer_age'] = futures[1].result()
        synthetic_data['customer_tenure'] = futures[2].result()
        synthetic_data['credit_score'] = futures[3].result()
        synthetic_data['transaction_frequency'] = futures[4].result()
    
    # Add transaction IDs and dates
    synthetic_data['transaction_id'] = [f'TXN{i:010d}' for i in range(len(synthetic_data))]
    synthetic_data['date'] = pd.date_range(start=start_date, end=end_date, periods=len(synthetic_data))
    
    # Save to CSV in parallel chunks
    print("Saving to CSV in parallel chunks...")
    chunk_size = 100000
    chunks = [(synthetic_data.iloc[i:i + chunk_size], i == 0) 
             for i in range(0, len(synthetic_data), chunk_size)]
    
    def save_chunk(chunk_data):
        chunk, is_first = chunk_data
        mode = 'w' if is_first else 'a'
        chunk.to_csv('synthetic_finance_data_ctgan.csv', 
                    mode=mode, 
                    header=is_first, 
                    index=False)
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        list(tqdm(executor.map(save_chunk, chunks), total=len(chunks)))
    
    # Initialize data quality validator
    validator = DataQualityValidator()
    if real_data_path and os.path.exists(real_data_path):
        real_data = pd.read_csv(real_data_path)
        validator.set_real_data(real_data)
        print("Loaded real data for quality validation")
    
    # Validate synthetic data quality
    if validator.real_data is not None:
        print("\nValidating synthetic data quality...")
        quality_metrics = validator.validate_distributions(synthetic_data)
        
        # Print quality report
        print("\nData Quality Report:")
        print("-------------------")
        print(f"Correlation Preservation Score: {quality_metrics['correlation_preservation']:.3f}")
        
        print("\nNumerical Column Similarities:")
        for col, metrics in quality_metrics['numerical_similarity'].items():
            print(f"\n{col}:")
            print(f"  KS Statistic: {metrics['ks_statistic']:.3f}")
            print(f"  P-value: {metrics['p_value']:.3f}")
            print(f"  Mean Difference: {metrics['mean_difference']:.3f}")
            print(f"  Std Difference: {metrics['std_difference']:.3f}")
        
        print("\nCategorical Column Similarities:")
        for col, metrics in quality_metrics['categorical_similarity'].items():
            print(f"\n{col}:")
            print(f"  Chi2 Statistic: {metrics['chi2_statistic']:.3f}")
            print(f"  P-value: {metrics['p_value']:.3f}")
            print(f"  Jensen-Shannon Divergence: {metrics['jsd']:.3f}")
        
        print("\nBias Metrics:")
        for metric, value in quality_metrics['bias_metrics'].items():
            print(f"  {metric}: {value:.3f}")
    
    progress_callback("generating", 1.0, "Finalizing data generation")
    
    print(f"Final data shape: {synthetic_data.shape}")
    return synthetic_data

if __name__ == "__main__":
    # Get the number of CPU cores
    num_cores = mp.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")
    
    # Generate data using parallel processing with smaller sample size
    data = generate_synthetic_finance_data(
        num_samples=100_000,  # Reduced from 1,000,000
        n_jobs=num_cores-1,  # Leave one core free for system processes
        real_data_path='real_finance_data.csv'  # Specify path to real data
    )
    print(f"Generated {len(data)} synthetic finance records")