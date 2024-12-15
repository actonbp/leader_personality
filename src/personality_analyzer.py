from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from typing import Dict, List
import logging
import os
import json
from datetime import datetime

class PersonalityAnalyzer:
    def __init__(self, model_name: str = "Minej/bert-base-personality"):
        """Initialize the personality analyzer with a specific model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 
                           'Conscientiousness', 'Openness']
        
        logging.info(f"Initialized PersonalityAnalyzer with model: {model_name}")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze a piece of text for personality traits."""
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits).squeeze().detach().numpy()
        
        results = {self.label_names[i]: float(predictions[i]) 
                  for i in range(len(self.label_names))}
        
        return results

    def analyze_text_chunks(self, text: str, chunk_size: int = 512) -> Dict[str, float]:
        """Analyze a long text by breaking it into chunks and averaging results."""
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) 
                 for i in range(0, len(words), chunk_size)]
        
        results = []
        for chunk in chunks:
            chunk_result = self.analyze_text(chunk)
            results.append(chunk_result)
        
        avg_results = {}
        for trait in self.label_names:
            avg_results[trait] = sum(r[trait] for r in results) / len(results)
            
        return avg_results

    def analyze_file(self, file_path: str) -> Dict[str, any]:
        """Analyze a single file and return results with metadata."""
        try:
            with open(file_path, 'r') as file:
                text = file.read()
            
            results = self.analyze_text_chunks(text)
            
            # Add metadata
            metadata = {
                'file_name': os.path.basename(file_path),
                'analysis_date': datetime.now().isoformat(),
                'file_path': file_path,
                'results': results
            }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {str(e)}")
            return None

    def analyze_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """Analyze all text files in a directory."""
        results = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(directory_path, file_name)
                result = self.analyze_file(file_path)
                if result:
                    results.append(result)
        return results

def save_results(results: List[Dict[str, any]], output_file: str):
    """Save analysis results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def print_results(results: List[Dict[str, any]], focus_trait: str = 'Neuroticism'):
    """Print analysis results with focus on a specific trait."""
    print(f"\nPersonality Analysis Results (Focus on {focus_trait}):")
    print("-" * 50)
    
    # Sort results by the focus trait
    sorted_results = sorted(results, 
                          key=lambda x: x['results'][focus_trait], 
                          reverse=True)
    
    for result in sorted_results:
        print(f"\nFile: {result['file_name']}")
        print(f"{focus_trait} Score: {result['results'][focus_trait]:.3f}")
        print("\nAll Traits:")
        for trait, score in result['results'].items():
            print(f"{trait}: {score:.3f}")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Create output directories if they don't exist
    os.makedirs('data/outputs/analysis', exist_ok=True)
    
    # Analyze speeches in the CEOs directory
    try:
        results = analyzer.analyze_directory('data/speeches/ceos')
        
        # Save results
        save_results(results, 'data/outputs/analysis/analysis_results.json')
        
        # Print results
        print_results(results, focus_trait='Neuroticism')
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 