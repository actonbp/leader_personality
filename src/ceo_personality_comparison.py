#!/usr/bin/env python3
"""
CEO Personality Analysis: BERT vs LIWC Comparison
-------------------------------------------------
This script analyzes CEO personality traits using both:
1. A BERT-based model for contextual understanding
2. A simplified LIWC-like approach for word-level analysis

It processes speech transcripts, extracts Big Five personality traits,
compares results from both approaches, and generates visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from pathlib import Path
import argparse
import random
import requests
from huggingface_hub import list_models
import logging
from datetime import datetime

# Import the enhanced analyzer
from enhanced_personality_analyzer import EnhancedPersonalityAnalyzer

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PersonalityAnalyzer:
    def __init__(self, model_name="Minej/bert-base-personality", use_demo_mode=False):
        """Initialize the analyzer with a pre-trained BERT model
        
        Args:
            model_name: HuggingFace model name
            use_demo_mode: If True, use random values instead of actual model predictions
                          (for demonstration/testing only)
        """
        self.use_demo_mode = use_demo_mode
        
        if not use_demo_mode:
            # For actual model predictions, load the model
            if "bert-base-personality" in model_name:
                # Use the specific BERT personality model
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.model = BertForSequenceClassification.from_pretrained(model_name)
            else:
                # Use a generic model for sequence classification
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
        
        self.label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 
                           'Conscientiousness', 'Openness']
        
    def analyze_text(self, text):
        """Analyze a single text for personality traits"""
        # For demo mode, just return random values in the 0.4-0.6 range
        if self.use_demo_mode:
            return {trait: 0.4 + (random.random() * 0.2) for trait in self.label_names}
            
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
    
    def analyze_text_chunks(self, text, chunk_size=512):
        """Analyze long text by breaking it into manageable chunks"""
        # For demo mode, just return random values directly
        if self.use_demo_mode:
            return {trait: 0.4 + (random.random() * 0.2) for trait in self.label_names}
            
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
            avg_results[trait] = sum(r[trait] for r in results) / len(results) if results else 0.5
            
        return avg_results


# Enhanced LIWC-like dictionaries for personality traits
enhanced_trait_dictionaries = {
    'Extroversion': [
        # Communication and socializing
        'social', 'outgoing', 'energetic', 'active', 'enthusiastic', 'excited', 'talkative', 
        'assertive', 'gregarious', 'sociable', 'confident', 'expressive', 'bold', 'engaging',
        'charismatic', 'dynamic', 'lively', 'spirited', 'vibrant', 'passionate', 'animated',
        # Group-related words
        'team', 'group', 'meeting', 'collaborate', 'partnership', 'together', 'interaction',
        'engage', 'network', 'community', 'conference', 'gathering', 'event', 'audience',  
        # Communication verbs
        'talk', 'discuss', 'speak', 'tell', 'share', 'announce', 'convey', 'explain', 'present',
        'address', 'communicate', 'express', 'articulate', 'converse', 'say', 'stated', 'declare'
    ],
    
    'Neuroticism': [
        # Anxiety and worry
        'worry', 'nervous', 'anxious', 'tense', 'stress', 'upset', 'fear', 'afraid', 
        'distress', 'concern', 'risk', 'problem', 'trouble', 'challenge', 'difficult',
        'uncertain', 'hesitant', 'cautious', 'concerned', 'apprehensive', 'uneasy',
        # Negative emotions
        'frustrate', 'disappoint', 'regret', 'fail', 'loss', 'decline', 'decrease', 
        'pressure', 'urgent', 'critical', 'serious', 'sensitive', 'vulnerable',
        # Business risk terms
        'threat', 'disruption', 'crisis', 'volatile', 'uncertainty', 'complex', 'ambiguous',
        'liability', 'exposure', 'adverse', 'negative', 'downside', 'constraint', 'limitation',
        # Hedging language
        'perhaps', 'maybe', 'might', 'could', 'possibly', 'potentially', 'somewhat', 
        'rather', 'approximately', 'around', 'estimated', 'projected', 'forecasted'
    ],
    
    'Agreeableness': [
        # Cooperation and support
        'agree', 'trust', 'cooperate', 'warm', 'helpful', 'friendly', 'kind', 'support',
        'appreciate', 'thank', 'collaborate', 'assist', 'help', 'serve', 'empathize',
        'understand', 'care', 'compassion', 'generous', 'giving', 'considerate', 'gentle',
        # Team and relationship words
        'partner', 'team', 'together', 'alliance', 'relationship', 'collaboration', 
        'community', 'stakeholder', 'client', 'customer', 'colleague', 'associate',
        # Positive acknowledgment
        'gratitude', 'recognition', 'acknowledge', 'respect', 'value', 'honor', 'admire',
        'praise', 'compliment', 'congratulate', 'celebrate', 'recognize', 'appreciate'
    ],
    
    'Conscientiousness': [
        # Organization and planning
        'careful', 'diligent', 'precise', 'thorough', 'organized', 'plan', 'prepare',
        'achieve', 'focus', 'goal', 'complete', 'responsible', 'disciplined', 'efficient',
        'methodical', 'structured', 'systematic', 'orderly', 'punctual', 'reliable',
        # Achievement words
        'accomplish', 'achieve', 'success', 'execute', 'deliver', 'performance', 'result',
        'outcome', 'effective', 'productive', 'efficient', 'quality', 'excellence', 'standard',
        # Process-oriented
        'process', 'procedure', 'protocol', 'guideline', 'framework', 'method', 'approach',
        'system', 'strategy', 'tactic', 'implement', 'monitor', 'measure', 'evaluate',
        # Time-oriented
        'schedule', 'deadline', 'timeline', 'milestone', 'promptly', 'timely', 'punctual'
    ],
    
    'Openness': [
        # Innovation and creativity
        'creative', 'innovative', 'curious', 'imagine', 'idea', 'explore', 'discover',
        'insight', 'perspective', 'vision', 'opportunity', 'learn', 'adapt', 'flexible',
        'novel', 'unique', 'original', 'inventive', 'ingenious', 'resourceful', 'clever',
        # Growth and exploration
        'grow', 'expand', 'develop', 'advance', 'progress', 'evolve', 'transform',
        'improve', 'enhance', 'upgrade', 'revolution', 'breakthrough', 'pioneer',
        # Future-oriented
        'future', 'forward', 'prospect', 'outlook', 'horizon', 'potential', 'possibility',
        'opportunity', 'aspiration', 'ambition', 'vision', 'foresight', 'anticipate',
        # Learning-oriented
        'learn', 'knowledge', 'skill', 'expertise', 'insight', 'understanding', 'wisdom',
        'information', 'education', 'training', 'development', 'growth', 'mindset'
    ]
}

# Original simplified dictionaries
simplified_trait_dictionaries = {
    'Extroversion': ['social', 'outgoing', 'energetic', 'active', 'enthusiastic', 'excited', 
                     'talkative', 'assertive', 'gregarious', 'sociable'],
    'Neuroticism': ['worry', 'nervous', 'anxious', 'tense', 'stress', 'upset', 'fear', 
                   'distress', 'concern', 'risk', 'problem', 'trouble', 'challenge'],
    'Agreeableness': ['agree', 'trust', 'cooperate', 'warm', 'helpful', 'friendly', 'kind', 
                      'support', 'appreciate', 'thank', 'collaborate'],
    'Conscientiousness': ['careful', 'diligent', 'precise', 'thorough', 'organized', 'plan', 
                          'prepare', 'achieve', 'focus', 'goal', 'complete', 'responsible'],
    'Openness': ['creative', 'innovative', 'curious', 'imagine', 'idea', 'explore', 'discover', 
                'insight', 'perspective', 'vision', 'opportunity', 'learn']
}

def find_personality_models():
    """Find personality-related models on HuggingFace Hub"""
    try:
        # Search for personality-related models
        models = list_models(filter="text-classification", search="personality")
        personality_models = [model.id for model in models]
        return personality_models
    except Exception as e:
        print(f"Error searching for personality models: {str(e)}")
        return []

def check_model_validity(model_name):
    """Check if a model is appropriate for personality analysis"""
    if "personality" in model_name.lower():
        return True, "Model seems to be specifically trained for personality analysis"
    
    try:
        # Try to get model info from Hugging Face API
        response = requests.get(f"https://huggingface.co/api/models/{model_name}")
        if response.status_code == 200:
            model_info = response.json()
            tags = model_info.get("tags", [])
            
            # Check for relevant tags
            personality_tags = ["personality", "psychology", "trait", "big-five", "ocean"]
            for tag in personality_tags:
                if any(tag in t.lower() for t in tags):
                    return True, f"Model has relevant tag: {tag}"
            
            # If it's a classification model, it might work but needs fine-tuning
            if "text-classification" in tags:
                return True, "General classification model; may need fine-tuning for personality"
            
            return False, "Model doesn't have personality-related tags"
        else:
            return False, f"Couldn't retrieve model info (status code: {response.status_code})"
    except Exception as e:
        return False, f"Error checking model: {str(e)}"

def simple_liwc_analysis(text, use_enhanced_dict=True):
    """A simplified LIWC-like analysis based on word counting
    
    Args:
        text: The text to analyze
        use_enhanced_dict: Whether to use the enhanced dictionary with more words
    """
    # Choose dictionary based on parameter
    trait_dictionaries = enhanced_trait_dictionaries if use_enhanced_dict else simplified_trait_dictionaries
    
    # Convert to lowercase and split into words
    words = text.lower().split()
    
    # Count words for each trait
    counts = {trait: 0 for trait in trait_dictionaries}
    total_words = len(words)
    
    for word in words:
        # Remove punctuation from word
        word = ''.join(c for c in word if c.isalpha())
        
        # Check each trait dictionary
        for trait, dictionary in trait_dictionaries.items():
            if word in dictionary:
                counts[trait] += 1
    
    # Convert to percentages
    percentages = {trait: count/total_words for trait, count in counts.items()}
    
    # Normalize to similar scale as BERT (0.4-0.6 range)
    normalized = {trait: 0.4 + (val * 0.2 / max(0.001, max(percentages.values()))) 
                 for trait, val in percentages.items()}
    
    return normalized


def process_speeches(data_path, results_path, limit=None, use_demo_mode=False, 
                    model_name="distilbert/distilbert-base-uncased", use_enhanced_liwc=True,
                    use_weighted_confidence=False):
    """Process all speech files in the data path"""
    os.makedirs(results_path, exist_ok=True)
    
    # Get list of all speech files
    speech_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    print(f"Found {len(speech_files)} speech files")
    
    # Limit files if requested
    if limit and limit > 0:
        speech_files = speech_files[:limit]
        print(f"Limited to processing {limit} files")
    
    # Check model validity if not in demo mode
    if not use_demo_mode:
        is_valid, message = check_model_validity(model_name)
        print(f"Model check: {message}")
        if not is_valid:
            print("Warning: The selected model may not be suitable for personality analysis")
            print("Consider using a dedicated personality model or enabling demo mode")
    
    # Process files with BERT - choose between standard and enhanced analyzers    
    bert_results = []
    
    if use_weighted_confidence:
        print("Using enhanced confidence-weighted analyzer for improved accuracy")
        analyzer = EnhancedPersonalityAnalyzer(model_name=model_name)
        
        # Process each file with enhanced analysis
        for filename in speech_files:
            file_path = os.path.join(data_path, filename)
            
            # Extract CEO name from filename
            name = filename.split('.')[0]
            if ' - ' in name:
                name = name.split(' - ')[0]
            name = ''.join([c for c in name if not c.isdigit()]).strip()
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze with enhanced weighted confidence
                trait_results = analyzer.analyze_text_chunks_weighted(content)
                
                # Store results
                bert_results.append({
                    'Name': name,
                    'File': filename,
                    **trait_results
                })
                
                print(f"Enhanced BERT processed: {name}")
            except Exception as e:
                print(f"Error processing {filename} with Enhanced BERT: {str(e)}")
    else:
        # Use original analyzer
        analyzer = PersonalityAnalyzer(model_name=model_name, use_demo_mode=use_demo_mode)
        
        # Process with standard analyzer
        for filename in speech_files:
            file_path = os.path.join(data_path, filename)
            
            # Extract CEO name from filename
            name = filename.split('.')[0]
            if ' - ' in name:
                name = name.split(' - ')[0]
            name = ''.join([c for c in name if not c.isdigit()]).strip()
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze with BERT
                trait_results = analyzer.analyze_text_chunks(content)
                
                # Store results
                bert_results.append({
                    'Name': name,
                    'File': filename,
                    **trait_results
                })
                
                print(f"BERT processed: {name}")
            except Exception as e:
                print(f"Error processing {filename} with BERT: {str(e)}")
    
    # Convert to DataFrame and save
    bert_df = pd.DataFrame(bert_results)
    
    # Save to appropriate file based on analysis method
    if use_weighted_confidence:
        bert_path = os.path.join(results_path, "bert_enhanced_personality_analysis.csv")
    else:
        bert_path = os.path.join(results_path, "bert_personality_analysis.csv")
    
    bert_df.to_csv(bert_path, index=False)
    print(f"BERT results saved to {bert_path}")
    
    # Process files with simplified LIWC
    liwc_results = []
    liwc_name = "enhanced_LIWC" if use_enhanced_liwc else "simple_LIWC"
    print(f"Using {liwc_name} dictionary for word-based analysis")
    
    for filename in speech_files:
        file_path = os.path.join(data_path, filename)
        
        # Extract CEO name
        name = filename.split('.')[0]
        if ' - ' in name:
            name = name.split(' - ')[0]
        name = ''.join([c for c in name if not c.isdigit()]).strip()
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze with simplified LIWC
            trait_results = simple_liwc_analysis(content, use_enhanced_dict=use_enhanced_liwc)
            
            # Store results
            liwc_results.append({
                'Name': name,
                'File': filename,
                **trait_results
            })
            
            print(f"LIWC processed: {name}")
        except Exception as e:
            print(f"Error processing {filename} with LIWC: {str(e)}")
    
    # Convert to DataFrame and save
    liwc_df = pd.DataFrame(liwc_results)
    liwc_path = os.path.join(results_path, f"liwc_personality_analysis_{liwc_name}.csv")
    liwc_df.to_csv(liwc_path, index=False)
    print(f"LIWC results saved to {liwc_path}")
    
    return bert_df, liwc_df


def analyze_correlations(bert_df, liwc_df, results_path, use_enhanced_liwc=True):
    """Analyze correlations between BERT and LIWC results"""
    # Determine which dictionary was used
    liwc_name = "enhanced_LIWC" if use_enhanced_liwc else "simple_LIWC"
    trait_dictionaries = enhanced_trait_dictionaries if use_enhanced_liwc else simplified_trait_dictionaries
    
    # Rename columns for clarity
    bert_renamed = bert_df.rename(columns={trait: f"BERT_{trait}" for trait in trait_dictionaries.keys()})
    liwc_renamed = liwc_df.rename(columns={trait: f"LIWC_{trait}" for trait in trait_dictionaries.keys()})
    
    # Merge the results
    comparison_df = pd.merge(bert_renamed, liwc_renamed, on=['Name', 'File'])
    
    # Calculate correlations
    correlations = {}
    for trait in trait_dictionaries.keys():
        bert_col = f"BERT_{trait}"
        liwc_col = f"LIWC_{trait}"
        corr = comparison_df[[bert_col, liwc_col]].corr().iloc[0, 1]
        correlations[trait] = corr
    
    # Save comparison DataFrame
    comparison_path = os.path.join(results_path, f"bert_liwc_comparison_{liwc_name}.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison results saved to {comparison_path}")
    
    # Save correlations
    correlations_df = pd.Series(correlations).rename("Correlation").to_frame()
    correlations_path = os.path.join(results_path, f"trait_correlations_{liwc_name}.csv")
    correlations_df.to_csv(correlations_path)
    print(f"Correlation results saved to {correlations_path}")
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(correlations.keys()), y=list(correlations.values()))
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'Correlation between BERT and {liwc_name} Personality Scores')
    plt.ylabel('Pearson Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    correlation_plot_path = os.path.join(results_path, f"bert_liwc_correlations_{liwc_name}.png")
    plt.savefig(correlation_plot_path)
    print(f"Correlation plot saved to {correlation_plot_path}")
    
    # Plot neuroticism specifically
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=comparison_df, x='BERT_Neuroticism', y='LIWC_Neuroticism')
    plt.title(f'Neuroticism Scores: BERT vs {liwc_name}')
    plt.xlabel('BERT Neuroticism Score')
    plt.ylabel(f'{liwc_name} Neuroticism Score')
    
    # Add regression line
    sns.regplot(data=comparison_df, x='BERT_Neuroticism', y='LIWC_Neuroticism', 
                scatter=False, line_kws={"color":"red"})
    
    # Save figure
    neuroticism_plot_path = os.path.join(results_path, f"neuroticism_comparison_{liwc_name}.png")
    plt.savefig(neuroticism_plot_path)
    print(f"Neuroticism comparison plot saved to {neuroticism_plot_path}")
    
    return correlations


def main():
    """Main function to parse arguments and run analysis"""
    parser = argparse.ArgumentParser(description="CEO Personality Analysis")
    parser.add_argument("--limit", type=int, default=30, help="Limit number of files to process")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without downloading models")
    parser.add_argument("--full", action="store_true", help="Run full analysis on all files")
    parser.add_argument("--simple-liwc", action="store_true", help="Use simple LIWC dictionary")
    parser.add_argument("--enhanced-liwc", action="store_true", help="Use enhanced LIWC dictionary (default)")
    parser.add_argument("--personality", action="store_true", help="Use dedicated personality model")
    parser.add_argument("--distilbert", action="store_true", help="Use DistilBERT (faster but less accurate)")
    parser.add_argument("--weighted", action="store_true", help="Use confidence-weighted analysis for improved accuracy")
    parser.add_argument("--list-models", action="store_true", help="List available personality models")
    parser.add_argument("--render", action="store_true", help="Render Quarto document")
    parser.add_argument("--data-dir", type=str, help="Custom data directory to process")
    
    args = parser.parse_args()
    
    # Display available models if requested
    if args.list_models:
        list_personality_models()
        return
    
    # Set data and results paths
    data_path = args.data_dir if args.data_dir else os.path.join("data", "282 ceo data  2")
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    
    # Determine model to use
    model_name = "distilbert/distilbert-base-uncased"  # Default
    if args.personality:
        model_name = "Minej/bert-base-personality"
    if args.distilbert:
        model_name = "distilbert/distilbert-base-uncased"
    
    # Handle --full flag
    limit = None if args.full else args.limit
    
    # Display configuration
    print("\nCEO Personality Analysis Configuration:")
    print(f"Data directory: {data_path}")
    print(f"Files to process: {'All' if limit is None else limit}")
    print(f"Model: {model_name}")
    print(f"LIWC dictionary: {'Enhanced' if not args.simple_liwc else 'Simple'}")
    print(f"Demo mode: {'Yes' if args.demo else 'No'}")
    print(f"Confidence weighting: {'Yes' if args.weighted else 'No'}")
    print(f"Results will be saved to: {os.path.abspath(results_path)}\n")
    
    # Process speeches
    process_speeches(
        data_path, 
        results_path, 
        limit=limit, 
        use_demo_mode=args.demo,
        model_name=model_name,
        use_enhanced_liwc=not args.simple_liwc,
        use_weighted_confidence=args.weighted
    )
    
    # Compare BERT and LIWC
    compare_bert_liwc(results_path, use_enhanced_liwc=not args.simple_liwc)
    
    # Render Quarto document if requested
    if args.render:
        render_quarto_document(results_path)


if __name__ == "__main__":
    main() 