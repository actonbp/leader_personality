# -*- coding: utf-8 -*-
"""ceo_personality.py

This script analyzes CEO personality traits using a BERT model.
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict
import torch
from transformers import BertTokenizer, BertForSequenceClassification

"""CONFIGURATION & SETUP"""

logging.basicConfig(level=logging.INFO)

# Use local paths instead of Google Drive
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories for your data
personality_dir = os.path.join(base_path, "data", "speeches", "ceos")

# Output file paths for results
results_dir = os.path.join(base_path, "results")
speeches_output = os.path.join(results_dir, "speeches_personality_analysis.csv")

# Ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)

"""Personality BERT analysis"""

class PersonalityAnalyzer:
    """
    Analyzes text to estimate Big Five personality traits using a fine-tuned BERT model.
    """
    def __init__(self, model_name: str = "Minej/bert-base-personality"):
        # Load the tokenizer and model for personality analysis
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.label_names = ['Extroversion', 'Neuroticism', 'Agreeableness',
                            'Conscientiousness', 'Openness']
        logging.info(f"Initialized PersonalityAnalyzer with model: {model_name}")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze a single text string for personality traits.
        Returns a dictionary of trait scores.
        """
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits).squeeze().detach().numpy()
        results = {self.label_names[i]: float(predictions[i]) for i in range(len(self.label_names))}
        return results

    def analyze_text_chunks(self, text: str, chunk_size: int = 512) -> Dict[str, float]:
        """
        Analyze longer text by splitting into chunks and averaging the results.
        """
        paragraphs = text.split('\n\n')
        current_chunk = []
        chunks = []
        current_length = 0

        for para in paragraphs:
            para_words = para.split()
            para_len = len(para_words)
            if current_length + para_len <= chunk_size:
                current_chunk.append(para)
                current_length += para_len
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_len

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        # Analyze each chunk
        results = []
        for chunk in chunks:
            chunk_result = self.analyze_text(chunk)
            results.append(chunk_result)

        # Average across all chunks
        avg_results = {}
        for trait in self.label_names:
            avg_results[trait] = sum(r[trait] for r in results) / len(results)
        return avg_results

"""Analysis functions"""

def analyze_file(file_path: str, personality_analyzer: PersonalityAnalyzer) -> Dict[str, any]:
    """
    Analyze personality traits for a single file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract CEO name
        first_line = content.split('\n')[0]
        name = first_line.strip() if first_line else "Unknown"

        # Personality Analysis
        personality_results = personality_analyzer.analyze_text_chunks(content)

        # Combine results
        combined_results = {
            'file_path': file_path,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            **personality_results
        }
        return combined_results
    except Exception as e:
        logging.error(f"Error analyzing file {file_path}: {str(e)}")
        return None

def analyze_directory(directory_path: str, personality_analyzer: PersonalityAnalyzer, output_file: str = None) -> pd.DataFrame:
    """
    Analyze personality for all .txt files in a directory.
    Returns a DataFrame of results and optionally saves it as a CSV.
    """
    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            result = analyze_file(file_path, personality_analyzer)
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    if output_file and len(df) > 0:
        df.to_csv(output_file, index=False)
    return df

"""MAIN EXECUTION BLOCK"""

if __name__ == "__main__":
    # Initialize the personality analyzer
    analyzer = PersonalityAnalyzer()

    # Analyze personality for speeches
    logging.info("Analyzing personality for CEO speeches...")
    speeches_df = analyze_directory(personality_dir, analyzer, speeches_output)
    logging.info(f"Personality analysis complete for speeches. Results saved to {speeches_output}")
    print("\n=== Personality Analysis (Speeches) ===")
    print(speeches_df.head()) 