"""
Enhanced CEO Personality Analysis using BERT-based Models with Confidence Weighting
----------------------------------------------------------------------------------
This module extends the standard personality analyzer with confidence-weighted averaging
to provide more accurate measurements of CEO personality traits. It weights each text
chunk based on the model's confidence level in its predictions.

The enhanced analysis pipeline includes:
1. Text preprocessing and chunking
2. BERT-based trait prediction with confidence estimation
3. Confidence-weighted result aggregation
4. Statistical analysis and output generation
"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import os
import json
from datetime import datetime

class EnhancedPersonalityAnalyzer:
    def __init__(self, model_name: str = "Minej/bert-base-personality"):
        """
        Initialize the enhanced personality analyzer with a pre-trained BERT model.
        
        Args:
            model_name (str): HuggingFace model identifier for the personality analysis model.
                            Default is "Minej/bert-base-personality" which is fine-tuned for
                            Big Five personality trait detection.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 
                           'Conscientiousness', 'Openness']
        
        logging.info(f"Initialized EnhancedPersonalityAnalyzer with model: {model_name}")

    def analyze_text_with_confidence(self, text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Analyze a single piece of text for personality traits and return confidence scores.
        
        Args:
            text (str): The input text to analyze (should be preprocessed and cleaned)
        
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: 
                - First dict: Mapping each personality trait to its score (0-1)
                - Second dict: Mapping each trait to its confidence score
        """
        # Tokenize and prepare input for BERT
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        
        # Get model predictions
        outputs = self.model(**inputs)
        
        # Get raw logits before sigmoid
        raw_logits = outputs.logits.squeeze().detach().numpy()
        
        # Convert logits to probabilities using sigmoid
        predictions = torch.sigmoid(outputs.logits).squeeze().detach().numpy()
        
        # Map predictions to trait names
        results = {self.label_names[i]: float(predictions[i]) 
                  for i in range(len(self.label_names))}
        
        # Calculate confidence as the absolute distance from decision boundary
        # The further from 0.5, the more confident the prediction
        confidence_scores = {}
        for i, trait in enumerate(self.label_names):
            # Normalize confidence to be between 0 and 1
            # A prediction close to 0 or 1 indicates high confidence
            confidence = 2 * abs(predictions[i] - 0.5)
            confidence_scores[trait] = float(confidence)
        
        return results, confidence_scores

    def analyze_text_chunks_weighted(self, text: str, chunk_size: int = 512) -> Dict[str, float]:
        """
        Analyze long text by breaking it into manageable chunks with confidence-weighted averaging.
        
        Args:
            text (str): The full text to analyze
            chunk_size (int): Maximum number of words per chunk (default: 512)
        
        Returns:
            Dict[str, float]: Confidence-weighted personality trait scores across all chunks
        """
        # Split on paragraphs to maintain context
        paragraphs = text.split('\n\n')
        current_chunk = []
        chunks = []
        current_length = 0
        
        # Build chunks while respecting paragraph boundaries
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
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # No chunks found (empty text)
        if not chunks:
            return {trait: 0.5 for trait in self.label_names}
        
        # Analyze each chunk with confidence scores
        chunk_results = []
        chunk_confidences = []
        
        for chunk in chunks:
            result, confidence = self.analyze_text_with_confidence(chunk)
            chunk_results.append(result)
            chunk_confidences.append(confidence)
        
        # Weighted averaging based on confidence scores
        weighted_results = {}
        
        for trait in self.label_names:
            # Extract trait values and their confidence scores
            trait_values = [r[trait] for r in chunk_results]
            trait_confidences = [c[trait] for c in chunk_confidences]
            
            # Normalize weights to sum to 1
            total_confidence = sum(trait_confidences)
            if total_confidence > 0:
                normalized_weights = [c/total_confidence for c in trait_confidences]
            else:
                # If all confidences are zero (unlikely), use equal weights
                normalized_weights = [1.0/len(trait_confidences)] * len(trait_confidences)
            
            # Calculate weighted average
            weighted_results[trait] = sum(v * w for v, w in zip(trait_values, normalized_weights))
            
        return weighted_results

    def analyze_file(self, file_path: str) -> Dict[str, any]:
        """
        Analyze a single file containing a CEO's speech with confidence weighting.
        
        Args:
            file_path (str): Path to the text file containing the speech
        
        Returns:
            Dict[str, any]: Analysis results with metadata
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract CEO name from standardized format (Name - Company CEO)
            first_line = content.split('\n')[0]
            name = first_line.split('-')[0].strip()
            
            # Analyze the content with weighted averaging
            results = self.analyze_text_chunks_weighted(content)
            
            # Compile metadata with results
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

    def analyze_directory(self, directory_path: str, output_file: str = None) -> pd.DataFrame:
        """
        Analyze all speech files in a directory with enhanced confidence weighting.
        
        Args:
            directory_path (str): Path to directory containing speech files
            output_file (str, optional): Path to save results (supports CSV or JSON)
        
        Returns:
            pd.DataFrame: DataFrame containing analysis results for all speeches
        """
        results = []
        
        # Process each file in directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                result = self.analyze_file(file_path)
                if result:
                    results.append(result)
                    logging.info(f"Processed {filename} with confidence-weighted analysis")
        
        # Convert results to DataFrame for analysis
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
                logging.info(f"Results saved to {output_file}")
            elif output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logging.info(f"Results saved to {output_file}")
        
        return df

if __name__ == "__main__":
    """
    Main execution block for running enhanced personality analysis on CEO speeches.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer with default model
    analyzer = EnhancedPersonalityAnalyzer()
    
    # Define input/output paths
    ceo_directory = "data/speeches/ceos"
    output_file = "results/enhanced_personality_analysis.csv"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    print(f"Starting enhanced analysis with confidence weighting on {ceo_directory}")
    results = analyzer.analyze_directory(ceo_directory, output_file)
    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Processed {len(results)} CEO files") 