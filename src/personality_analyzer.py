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
        # Split on paragraphs to maintain context
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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract CEO name and company from standardized format
            first_line = content.split('\n')[0]
            name = first_line.split('-')[0].strip()
            
            # Analyze the content
            results = self.analyze_text_chunks(content)
            
            # Add metadata
            metadata = {
                'file_path': file_path,
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'analysis_results': results
            }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {str(e)}")
            return None

    def analyze_directory(self, directory_path: str, output_file: str = None) -> List[Dict]:
        """Analyze all text files in a directory and optionally save results."""
        results = []
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                result = self.analyze_file(file_path)
                if result:
                    results.append(result)
        
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'Name': r['name'],
                'File': os.path.basename(r['file_path']),
                'Timestamp': r['timestamp'],
                **r['analysis_results']
            }
            for r in results
        ])
        
        # Save results if output file specified
        if output_file:
            if output_file.endswith('.csv'):
                df.to_csv(output_file, index=False)
            elif output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
        
        return df

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Process all CEO speeches
    ceo_directory = "data/speeches/ceos"
    output_file = "results/personality_analysis.csv"
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Run analysis
    logging.info("Starting personality analysis of CEO speeches...")
    results_df = analyzer.analyze_directory(ceo_directory, output_file)
    logging.info(f"Analysis complete. Results saved to {output_file}")
    
    # Display summary statistics
    print("\nPersonality Analysis Summary:")
    print("============================")
    print(results_df.describe()) 