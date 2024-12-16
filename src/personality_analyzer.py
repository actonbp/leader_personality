"""
CEO Personality Analysis using BERT-based Models
----------------------------------------------
This module implements personality trait analysis for CEO speeches using a fine-tuned BERT model.
It processes text data to extract Big Five personality traits: Extroversion, Neuroticism,
Agreeableness, Conscientiousness, and Openness.

The analysis pipeline includes:
1. Text preprocessing and tokenization
2. BERT-based trait prediction
3. Result aggregation and normalization
4. Statistical analysis and output generation
"""

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
        """
        Initialize the personality analyzer with a pre-trained BERT model.
        
        Args:
            model_name (str): HuggingFace model identifier for the personality analysis model.
                            Default is "Minej/bert-base-personality" which is fine-tuned for
                            Big Five personality trait detection.
        
        The initialization process:
        1. Loads the BERT tokenizer for text preprocessing
        2. Loads the pre-trained model for personality prediction
        3. Sets up trait labels for result mapping
        4. Configures logging for tracking analysis progress
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 
                           'Conscientiousness', 'Openness']
        
        logging.info(f"Initialized PersonalityAnalyzer with model: {model_name}")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze a single piece of text for personality traits.
        
        Args:
            text (str): The input text to analyze (should be preprocessed and cleaned)
        
        Returns:
            Dict[str, float]: Dictionary mapping each personality trait to its score (0-1)
        
        Process:
        1. Tokenize text using BERT tokenizer (handles max length and padding)
        2. Pass tokens through the model
        3. Apply sigmoid activation to get probability scores
        4. Map scores to trait names
        """
        # Tokenize and prepare input for BERT
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        
        # Get model predictions
        outputs = self.model(**inputs)
        
        # Convert logits to probabilities using sigmoid
        predictions = torch.sigmoid(outputs.logits).squeeze().detach().numpy()
        
        # Map predictions to trait names
        results = {self.label_names[i]: float(predictions[i]) 
                  for i in range(len(self.label_names))}
        
        return results

    def analyze_text_chunks(self, text: str, chunk_size: int = 512) -> Dict[str, float]:
        """
        Analyze long text by breaking it into manageable chunks while preserving context.
        
        Args:
            text (str): The full text to analyze
            chunk_size (int): Maximum number of words per chunk (default: 512 to match BERT's limit)
        
        Returns:
            Dict[str, float]: Averaged personality trait scores across all chunks
        
        Process:
        1. Split text into paragraphs to maintain semantic coherence
        2. Group paragraphs into chunks of appropriate size
        3. Analyze each chunk separately
        4. Average results across all chunks
        
        Note: This method preserves context better than simple truncation
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
        
        # Analyze each chunk
        results = []
        for chunk in chunks:
            chunk_result = self.analyze_text(chunk)
            results.append(chunk_result)
        
        # Average results across chunks
        avg_results = {}
        for trait in self.label_names:
            avg_results[trait] = sum(r[trait] for r in results) / len(results)
            
        return avg_results

    def analyze_file(self, file_path: str) -> Dict[str, any]:
        """
        Analyze a single file containing a CEO's speech.
        
        Args:
            file_path (str): Path to the text file containing the speech
        
        Returns:
            Dict[str, any]: Analysis results with metadata including:
                           - file_path: Original file location
                           - name: Extracted CEO name
                           - timestamp: Analysis timestamp
                           - analysis_results: Personality trait scores
        
        Process:
        1. Read and validate file content
        2. Extract metadata (CEO name, context)
        3. Analyze text in chunks
        4. Compile results with metadata
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract CEO name from standardized format (Name - Company CEO)
            first_line = content.split('\n')[0]
            name = first_line.split('-')[0].strip()
            
            # Analyze the content
            results = self.analyze_text_chunks(content)
            
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

    def analyze_directory(self, directory_path: str, output_file: str = None) -> List[Dict]:
        """
        Analyze all speech files in a directory and generate comprehensive results.
        
        Args:
            directory_path (str): Path to directory containing speech files
            output_file (str, optional): Path to save results (supports CSV or JSON)
        
        Returns:
            pd.DataFrame: DataFrame containing analysis results for all speeches
        
        Process:
        1. Scan directory for text files
        2. Process each file individually
        3. Aggregate results into a DataFrame
        4. Save results in specified format
        5. Generate summary statistics
        """
        results = []
        
        # Process each file in directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                result = self.analyze_file(file_path)
                if result:
                    results.append(result)
        
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
            elif output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
        
        return df

if __name__ == "__main__":
    """
    Main execution block for running personality analysis on CEO speeches.
    
    Process:
    1. Set up logging configuration
    2. Initialize the personality analyzer
    3. Process all speeches in the specified directory
    4. Save results and display summary statistics
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer with default model
    analyzer = PersonalityAnalyzer()
    
    # Define input/output paths
    ceo_directory = "data/speeches/ceos"
    output_file = "results/personality_analysis.csv"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Run analysis pipeline
    logging.info("Starting personality analysis of CEO speeches...")
    results_df = analyzer.analyze_directory(ceo_directory, output_file)
    logging.info(f"Analysis complete. Results saved to {output_file}")
    
    # Display summary statistics
    print("\nPersonality Analysis Summary:")
    print("============================")
    print(results_df.describe()) 