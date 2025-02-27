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

def generate_batch(batch_params):
    """Helper function to generate a single batch of data"""
    batch_size, seed, features = batch_params
    np.random.seed(seed)
    
    transaction_types = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'RTGS/NEFT']
    merchant_categories = ['Retail', 'Travel', 'Entertainment', 'Grocery', 'Online Shopping']
    bank_types = ['Public', 'Private', 'International']
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
    
    data_batch = {
        'amount': np.random.lognormal(mean=8.5, sigma=1.2, size=batch_size),
        'transaction_type': np.random.choice(transaction_types, size=batch_size),
        'merchant_category': np.random.choice(merchant_categories, size=batch_size),
        'bank_type': np.random.choice(bank_types, size=batch_size),
        'city': np.random.choice(cities, size=batch_size),
        'customer_age': np.random.normal(35, 12, batch_size).astype(int),
        'customer_tenure': np.random.normal(5, 3, batch_size).astype(int),
        'transaction_frequency': np.random.poisson(5, size=batch_size),
        'credit_score': np.random.normal(700, 100, batch_size).clip(300, 900).astype(int),
        'is_fraud': np.random.choice([0, 1], size=batch_size, p=[0.995, 0.005])
    }
    
    if features:
        data_batch = {k: v for k, v in data_batch.items() if features.get(k, True)}
    
    return pd.DataFrame(data_batch)

class ParallelMemoryAugmentedCTGAN(CTGAN):
    def __init__(self, memory_size=1000, n_jobs=-1, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_bank = None
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
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
    progress_callback=None
):
    # Set default callback if none provided
    if progress_callback is None:
        progress_callback = lambda stage, progress, message: None
    
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
    
    ctgan = ParallelMemoryAugmentedCTGAN(
        memory_size=memory_size,
        epochs=epochs,
        batch_size=500,
        verbose=True,
        n_jobs=n_jobs
    )
    
    # Add progress tracking to CTGAN training
    ctgan._train_epochs = lambda *args, **kwargs: (
        progress_tracker(i) or original_train_epochs(*args, **kwargs)
        for i, original_train_epochs in enumerate(range(epochs))
    )
    
    discrete_columns = [
        'transaction_type', 
        'merchant_category',
        'bank_type',
        'city',
        'is_fraud'
    ]

    # Fit the model
    ctgan.fit(train_df, discrete_columns=discrete_columns)
    
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
        n_jobs=num_cores-1  # Leave one core free for system processes
    )
    print(f"Generated {len(data)} synthetic finance records")