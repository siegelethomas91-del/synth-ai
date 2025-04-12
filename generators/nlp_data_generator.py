from generators.ecommerce_nlp_generator import EcommerceNLPDataGenerator
import pandas as pd
import os
import json
from tqdm import tqdm
from datetime import datetime
import joblib
import logging
from pathlib import Path
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class ModelConfig:
    """Configuration for model training and generation"""
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    model_dir: str = 'trained_models'
    cache_dir: str = 'data_cache'
    n_jobs: int = mp.cpu_count() - 1

class EnhancedNLPDatasetGenerator:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.ecommerce = EcommerceNLPDataGenerator()
        self.setup_logging()
        self.setup_dirs()
        self.models = {}
        self.load_cached_models()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nlp_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_dirs(self):
        """Create necessary directories"""
        Path(self.config.model_dir).mkdir(exist_ok=True)
        Path(self.config.cache_dir).mkdir(exist_ok=True)

    def load_cached_models(self):
        """Load previously trained models"""
        try:
            model_files = Path(self.config.model_dir).glob('*.pkl')
            for model_file in model_files:
                model_name = model_file.stem
                self.logger.info(f"Loading cached model: {model_name}")
                self.models[model_name] = joblib.load(model_file)
        except Exception as e:
            self.logger.error(f"Error loading cached models: {str(e)}")

    def generate_training_data(
        self, 
        n_samples: int,
        force_retrain: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Generate training data and train models"""
        self.force_retrain = force_retrain
        cache_file = Path(self.config.cache_dir) / f'data_cache_{n_samples}.pkl'

        if not force_retrain and cache_file.exists():
            self.logger.info("Loading cached data...")
            datasets = joblib.load(cache_file)
        else:
            self.logger.info("Generating new datasets...")
            
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                # Generate base data in parallel
                future_products = executor.submit(
                    self.ecommerce.generate_product_data, n_samples
                )
                products_df = future_products.result()
                
                future_reviews = executor.submit(
                    self.ecommerce.generate_review_data, products_df
                )
                reviews_df = future_reviews.result()

            # Create datasets with progress bars
            nlp_datasets = {}
            dataset_creators = {
                'sentiment_analysis': (self._create_sentiment_dataset, reviews_df),
                'product_classification': (self._create_classification_dataset, products_df),
                'text_generation': (self._create_text_generation_dataset, (products_df, reviews_df))
            }

            for name, (creator_func, data) in tqdm(
                dataset_creators.items(), 
                desc="Creating datasets"
            ):
                nlp_datasets[name] = creator_func(data)

            # Cache the generated data
            joblib.dump(nlp_datasets, cache_file)
            datasets = nlp_datasets
        
        # Train and save models
        self.train_and_save_models(datasets)
        
        return datasets

    def _create_sentiment_dataset(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment analysis dataset from reviews"""
        try:
            sentiment_data = pd.DataFrame({
                'text': reviews_df['review_text'],
                'rating': reviews_df['rating'],
                'sentiment': reviews_df['rating'].apply(
                    lambda x: 'positive' if x >= 4 
                    else 'negative' if x <= 2 
                    else 'neutral'
                )
            })
            
            # Split into train/test sets
            train_data, test_data = train_test_split(
                sentiment_data, 
                test_size=0.2, 
                random_state=42
            )
            
            return pd.concat([
                train_data.assign(split='train'),
                test_data.assign(split='test')
            ])
        
        except Exception as e:
            self.logger.error(f"Error creating sentiment dataset: {str(e)}")
            raise

    def _create_classification_dataset(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Create product classification dataset"""
        try:
            classification_data = pd.DataFrame({
                'text': products_df['description'],
                'category': products_df['category'],
                'subcategory': products_df['subcategory']
            })
            
            # Split into train/test sets
            train_data, test_data = train_test_split(
                classification_data, 
                test_size=0.2, 
                random_state=42
            )
            
            return pd.concat([
                train_data.assign(split='train'),
                test_data.assign(split='test')
            ])
        
        except Exception as e:
            self.logger.error(f"Error creating classification dataset: {str(e)}")
            raise

    def _create_text_generation_dataset(
        self, 
        data_tuple: tuple[pd.DataFrame, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create text generation dataset from products and reviews"""
        try:
            products_df, reviews_df = data_tuple
            
            # Combine product info with reviews
            generation_data = pd.merge(
                reviews_df[['product_id', 'review_text']],
                products_df[['product_id', 'description']],
                on='product_id'
            )
            
            # Prepare input-output pairs
            generation_data = pd.DataFrame({
                'input_text': generation_data['description'],
                'target_text': generation_data['review_text']
            })
            
            # Split into train/test sets
            train_data, test_data = train_test_split(
                generation_data, 
                test_size=0.2, 
                random_state=42
            )
            
            return pd.concat([
                train_data.assign(split='train'),
                test_data.assign(split='test')
            ])
        
        except Exception as e:
            self.logger.error(f"Error creating text generation dataset: {str(e)}")
            raise

    def save_datasets(
        self, 
        datasets: Dict[str, pd.DataFrame], 
        output_dir: str = 'synthetic_data'
    ) -> None:
        """Save datasets with validation and error handling"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            for name, data in datasets.items():
                # Validate data before saving
                if self._validate_dataset(name, data):
                    # Save both CSV and JSON versions
                    futures.append(
                        executor.submit(self._save_dataset, name, data, output_path)
                    )
                else:
                    self.logger.error(f"Validation failed for dataset: {name}")

            # Wait for all saves to complete
            for future in tqdm(futures, desc="Saving datasets"):
                future.result()

    def _validate_dataset(self, name: str, data: pd.DataFrame) -> bool:
        """Validate dataset before saving"""
        try:
            if data.empty:
                return False
            if data.isnull().any().any():
                self.logger.warning(f"Dataset {name} contains null values")
            return True
        except Exception as e:
            self.logger.error(f"Error validating dataset {name}: {str(e)}")
            return False

    def _save_dataset(
        self, 
        name: str, 
        data: pd.DataFrame, 
        output_path: Path
    ) -> None:
        """Save individual dataset with error handling"""
        try:
            # Save CSV
            data.to_csv(output_path / f"{name}.csv", index=False)
            
            # Save JSON
            data.to_json(
                output_path / f"{name}.json",
                orient='records',
                lines=True
            )
            
            # Generate dataset statistics
            stats = self._generate_dataset_stats(data)
            with open(output_path / f"{name}_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving dataset {name}: {str(e)}")

    def _generate_dataset_stats(self, data: pd.DataFrame) -> dict:
        """Generate statistical summary of dataset"""
        return {
            'rows': len(data),
            'columns': list(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'null_counts': data.isnull().sum().to_dict(),
            'sample': data.head(5).to_dict(orient='records')
        }

    def save_model(self, model_name: str, model: object) -> bool:
        """Save trained model to disk"""
        try:
            model_path = Path(self.config.model_dir) / f"{model_name}_model.pkl"
            joblib.dump(model, model_path, compress=3)
            self.logger.info(f"Model {model_name} saved successfully to {model_path}")
            self.models[model_name] = model
            return True
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            return False

    def train_and_save_models(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Train and save models for each dataset type"""
        for name, data in datasets.items():
            if name not in self.models or self.force_retrain:
                try:
                    self.logger.info(f"Training model for {name}...")
                    
                    if name == 'sentiment_analysis':
                        from sklearn.pipeline import Pipeline
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.naive_bayes import MultinomialNB
                        
                        model = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', MultinomialNB())
                        ])
                        
                        train_data = data[data['split'] == 'train']
                        model.fit(train_data['text'], train_data['sentiment'])
                        
                    elif name == 'product_classification':
                        from sklearn.pipeline import Pipeline
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.ensemble import RandomForestClassifier
                        
                        model = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', RandomForestClassifier())
                        ])
                        
                        train_data = data[data['split'] == 'train']
                        model.fit(train_data['text'], train_data['category'])
                    
                    elif name == 'text_generation':
                        from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
                        
                        model = GPT2LMHeadModel.from_pretrained('gpt2')
                        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                        model.save_pretrained(Path(self.config.model_dir) / f"{name}_transformer")
                        tokenizer.save_pretrained(Path(self.config.model_dir) / f"{name}_transformer")
                    
                    # Save the trained model
                    if name != 'text_generation':  # Handle transformer models separately
                        self.save_model(name, model)
                    
                    self.logger.info(f"Model for {name} trained and saved successfully!")
                    
                except Exception as e:
                    self.logger.error(f"Error training model for {name}: {str(e)}")
                    continue

def main():
    config = ModelConfig(
        batch_size=128,
        epochs=15,
        learning_rate=0.001,
        n_jobs=mp.cpu_count() - 1
    )
    
    generator = EnhancedNLPDatasetGenerator(config)
    n_samples = 1000

    try:
        generator.logger.info("Starting data generation and model training process...")
        
        # Generate data and train models
        datasets = generator.generate_training_data(
            n_samples=n_samples,
            force_retrain=False  # Set to True to force model retraining
        )
        
        # Save datasets with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'synthetic_data_{timestamp}'
        generator.save_datasets(datasets, output_dir)
        
        # Log completion status
        generator.logger.info(f"Generation and training completed successfully!")
        generator.logger.info(f"Data saved to: {output_dir}")
        generator.logger.info(f"Models saved to: {config.model_dir}")
        
        # Print statistics
        for name, data in datasets.items():
            generator.logger.info(f"{name}: {len(data)} samples generated")
            if name in generator.models:
                generator.logger.info(f"{name} model trained and saved successfully")

    except Exception as e:
        generator.logger.error(f"Error during process: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()