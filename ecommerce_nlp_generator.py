import pandas as pd
import numpy as np
from faker import Faker
import torch
from datetime import datetime, timedelta
from ctgan import CTGAN
import gc
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import joblib
from transformers import GPT2Tokenizer
import random
import json

class EcommerceNLPDataGenerator:
    def __init__(self):
        self.faker = Faker(['en_IN', 'hi_IN'])
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Set a fixed end date to avoid future dates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=730)  # 2 years ago
        
        # Categories and descriptions
        self.categories = {
            'Electronics': ['Smartphones', 'Laptops', 'Accessories'],
            'Fashion': ['Men', 'Women', 'Kids'],
            'Home': ['Furniture', 'Decor', 'Kitchen'],
            'Beauty': ['Skincare', 'Makeup', 'Haircare']
        }
        
        # Templates
        self.review_templates = [
            "The {product} is {sentiment}. {detail}",
            "I {sentiment_verb} this {product}. {detail}",
            "{sentiment_start} product. {detail}"
        ]
        
        # Sentiment mappings
        self.sentiment_mappings = {
            'positive': {
                'words': ['excellent', 'amazing', 'great'],
                'verbs': ['love', 'recommend', 'enjoy'],
                'starts': ['Excellent', 'Amazing', 'Great']
            },
            'negative': {
                'words': ['poor', 'terrible', 'disappointing'],
                'verbs': ['dislike', 'regret', 'avoid'],
                'starts': ['Poor', 'Terrible', 'Disappointing']
            },
            'neutral': {
                'words': ['okay', 'decent', 'average'],
                'verbs': ['think', 'feel', 'believe'],
                'starts': ['Okay', 'Decent', 'Average']
            }
        }

    def generate_product_description(self, category, subcategory):
        """Generate product description"""
        features = [
            self.faker.word(),
            self.faker.word(),
            self.faker.word()
        ]
        return f"A premium {subcategory} from {category} category. Features include: {', '.join(features)}."

    def generate_review_text(self, product_name, sentiment, category):
        """Generate review text"""
        template = random.choice(self.review_templates)
        sentiment_dict = self.sentiment_mappings[sentiment]
        
        detail = f"Perfect for {category.lower()} enthusiasts." if sentiment == 'positive' else \
                f"Needs improvement in quality." if sentiment == 'negative' else \
                f"Serves the purpose."
        
        return template.format(
            product=product_name,
            sentiment=random.choice(sentiment_dict['words']),
            sentiment_verb=random.choice(sentiment_dict['verbs']),
            sentiment_start=random.choice(sentiment_dict['starts']),
            detail=detail
        )

    def generate_product_data(self, n_samples):
        """Generate synthetic product data"""
        data = []
        for _ in tqdm(range(n_samples), desc="Generating product data"):
            category = random.choice(list(self.categories.keys()))
            subcategory = random.choice(self.categories[category])
            
            product = {
                'product_id': f"PROD{str(random.randint(10000, 99999))}",
                'name': f"{self.faker.company()} {subcategory}",
                'category': category,
                'subcategory': subcategory,
                'price': round(random.uniform(100, 50000), 2),
                'description': self.generate_product_description(category, subcategory),
                'rating': round(random.uniform(3.5, 5.0), 1),
                'stock': random.randint(0, 1000),
                'seller': self.faker.company(),
                'created_at': self.faker.date_time_between(
                    start_date=self.start_date,
                    end_date=self.end_date
                ).strftime('%Y-%m-%d %H:%M:%S')  # Format date as string
            }
            data.append(product)
        return pd.DataFrame(data)
    
    def generate_review_data(self, products_df, n_reviews_per_product=5):
        """Generate synthetic review data"""
        reviews = []
        for _, product in tqdm(products_df.iterrows(), desc="Generating reviews"):
            product_date = datetime.strptime(product['created_at'], '%Y-%m-%d %H:%M:%S')
            for _ in range(random.randint(1, n_reviews_per_product)):
                sentiment = random.choice(['positive', 'negative', 'neutral'])
                review = {
                    'review_id': f"REV{str(random.randint(100000, 999999))}",
                    'product_id': product['product_id'],
                    'user_id': f"USER{str(random.randint(1000, 9999))}",
                    'rating': random.randint(1, 5),
                    'review_text': self.generate_review_text(
                        product['name'], 
                        sentiment,
                        product['category']
                    ),
                    'language': random.choice(['en', 'hi']),
                    'helpful_votes': random.randint(0, 100),
                    'verified_purchase': random.choice([True, False]),
                    'created_at': self.faker.date_time_between(
                        start_date=product_date,
                        end_date=self.end_date
                    ).strftime('%Y-%m-%d %H:%M:%S')  # Format date as string
                }
                reviews.append(review)
        return pd.DataFrame(reviews)