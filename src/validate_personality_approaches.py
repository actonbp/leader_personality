"""
Validation script to compare LIWC and BERT approaches for personality analysis
using the IPIP dataset as ground truth.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
import json
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our personality analyzers
from src.personality_analyzer import PersonalityAnalyzer
from src.enhanced_personality_analyzer import EnhancedPersonalityAnalyzer

# LIWC trait mapping dictionary
# This maps LIWC categories to Big Five traits based on established research
LIWC_TRAIT_MAPPING = {
    'Extroversion': ['social', 'family', 'friend', 'posemo', 'affect', 'we', 'they'],
    'Neuroticism': ['negemo', 'anx', 'anger', 'sad', 'affect', 'i', 'swear'],
    'Agreeableness': ['social', 'affiliation', 'we', 'posemo', 'friend'],
    'Conscientiousness': ['work', 'achieve', 'time', 'certain', 'quant', 'number'],
    'Openness': ['insight', 'cause', 'discrep', 'tentat', 'differ', 'percept', 'see', 'hear', 'feel']
}

# Function to simulate LIWC analysis
def simple_liwc_classifier(text, liwc_mapping=LIWC_TRAIT_MAPPING):
    """
    A simplified LIWC approach that uses keyword matching to classify text.
    This simulates the LIWC approach without requiring the actual LIWC software.
    
    Args:
        text (str): The text to analyze
        liwc_mapping (dict): Mapping of LIWC categories to keywords
    
    Returns:
        str: The predicted Big Five trait
    """
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    # Count occurences of keywords for each trait
    trait_scores = defaultdict(int)
    
    for trait, keywords in liwc_mapping.items():
        for keyword in keywords:
            # Simple word matching (in real LIWC, this would be more sophisticated)
            if keyword.lower() in text:
                trait_scores[trait] += 1
    
    # If no traits were found, return None
    if len(trait_scores) == 0:
        return None
    
    # Return the trait with the highest score
    return max(trait_scores.items(), key=lambda x: x[1])[0]

def main():
    # Load IPIP dataset with error handling for encoding
    try:
        # Try with UTF-8 encoding first
        ipip_df = pd.read_csv("data/IPIP.csv", encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 encoding
        ipip_df = pd.read_csv("data/IPIP.csv", encoding="latin-1")
    
    # Map IPIP labels to Big Five traits
    # This mapping is approximate and based on common understanding
    big_five_mapping = {
        'Extraversion': 'Extroversion',
        'Neuroticism': 'Neuroticism',
        'Agreeableness': 'Agreeableness',
        'Conscientiousness': 'Conscientiousness',
        'Intellect': 'Openness',  # Intellect is often used as a proxy for Openness
        'Openness': 'Openness',
        'Emotional Stability': 'Neuroticism'  # Inverse of Neuroticism
    }
    
    # Create a new column for mapped Big Five traits
    # Handle NaN values
    def map_to_big_five(label):
        if pd.isna(label):
            return 'Other'
        try:
            # Try to find a matching Big Five trait
            return next((big_five_mapping[key] for key in big_five_mapping if key in label), 'Other')
        except TypeError:
            # Handle any other type errors
            return 'Other'
    
    ipip_df['big_five_trait'] = ipip_df['label'].apply(map_to_big_five)
    
    # Filter to only include items that map to Big Five traits
    big_five_df = ipip_df[ipip_df['big_five_trait'] != 'Other'].copy()
    
    print(f"Total IPIP items: {len(ipip_df)}")
    print(f"Big Five items: {len(big_five_df)}")
    print(f"Distribution of Big Five traits:")
    print(big_five_df['big_five_trait'].value_counts())
    
    # Initialize BERT personality analyzer
    bert_analyzer = EnhancedPersonalityAnalyzer()
    
    # Add columns for predictions with error handling
    print("\nApplying LIWC-style analysis...")
    big_five_df['liwc_prediction'] = big_five_df['text'].apply(simple_liwc_classifier)
    
    print("Applying BERT analysis (this may take some time)...")
    
    # Function to safely apply BERT analysis with error handling
    def safe_bert_analysis(text, analyzer=bert_analyzer):
        try:
            if pd.isna(text):
                return None
            # Use the correct method from the EnhancedPersonalityAnalyzer class
            results = analyzer.analyze_text_chunks_weighted(text)
            # Find the most expressed trait
            if results:
                most_expressed = max(results, key=results.get)
                return most_expressed
            return None
        except Exception as e:
            print(f"Error analyzing text with BERT: {e}")
            return None
    
    # Process in batches to show progress
    batch_size = 50
    total_rows = len(big_five_df)
    bert_predictions = []
    
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        print(f"Processing items {i+1}-{end_idx} of {total_rows}...")
        
        batch_texts = big_five_df['text'].iloc[i:end_idx]
        batch_predictions = [safe_bert_analysis(text) for text in batch_texts]
        bert_predictions.extend(batch_predictions)
    
    big_five_df['bert_prediction'] = bert_predictions
    
    # Calculate accuracy for both methods
    # Filter out None predictions from both methods
    liwc_mask = big_five_df['liwc_prediction'].notna()
    bert_mask = big_five_df['bert_prediction'].notna()
    
    liwc_accuracy = 0
    if liwc_mask.sum() > 0:
        liwc_accuracy = accuracy_score(
            big_five_df.loc[liwc_mask, 'big_five_trait'], 
            big_five_df.loc[liwc_mask, 'liwc_prediction']
        )
    
    bert_accuracy = 0
    if bert_mask.sum() > 0:
        bert_accuracy = accuracy_score(
            big_five_df.loc[bert_mask, 'big_five_trait'], 
            big_five_df.loc[bert_mask, 'bert_prediction']
        )
    
    print(f"\nLIWC Accuracy: {liwc_accuracy:.4f} (on {liwc_mask.sum()} items)")
    print(f"BERT Accuracy: {bert_accuracy:.4f} (on {len(big_five_df)} items)")
    
    # Generate classification reports
    if liwc_mask.sum() > 0:
        print("\nLIWC Classification Report:")
        try:
            print(classification_report(
                big_five_df.loc[liwc_mask, 'big_five_trait'], 
                big_five_df.loc[liwc_mask, 'liwc_prediction']
            ))
        except Exception as e:
            print(f"Could not generate LIWC classification report: {e}")
    
    if bert_mask.sum() > 0:
        print("\nBERT Classification Report:")
        try:
            print(classification_report(
                big_five_df.loc[bert_mask, 'big_five_trait'], 
                big_five_df.loc[bert_mask, 'bert_prediction']
            ))
        except Exception as e:
            print(f"Could not generate BERT classification report: {e}")
    
    # Create confusion matrices if we have predictions
    try:
        plt.figure(figsize=(15, 6))
        
        # Determine the unique labels across both datasets
        unique_traits = sorted(big_five_df['big_five_trait'].unique())
        
        # LIWC confusion matrix
        if liwc_mask.sum() > 0:
            plt.subplot(1, 2, 1)
            try:
                liwc_cm = confusion_matrix(
                    big_five_df.loc[liwc_mask, 'big_five_trait'], 
                    big_five_df.loc[liwc_mask, 'liwc_prediction'],
                    labels=unique_traits
                )
                sns.heatmap(liwc_cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=unique_traits,
                            yticklabels=unique_traits)
                plt.title(f'LIWC Confusion Matrix\nAccuracy: {liwc_accuracy:.4f}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
            except Exception as e:
                print(f"Could not generate LIWC confusion matrix: {e}")
                plt.text(0.5, 0.5, "Error generating LIWC confusion matrix", 
                        ha='center', va='center', fontsize=12)
        else:
            plt.subplot(1, 2, 1)
            plt.text(0.5, 0.5, "No LIWC predictions available", 
                    ha='center', va='center', fontsize=12)
        
        # BERT confusion matrix
        if bert_mask.sum() > 0:
            plt.subplot(1, 2, 2)
            try:
                bert_cm = confusion_matrix(
                    big_five_df.loc[bert_mask, 'big_five_trait'], 
                    big_five_df.loc[bert_mask, 'bert_prediction'],
                    labels=unique_traits
                )
                sns.heatmap(bert_cm, annot=True, fmt='d', cmap='Greens',
                            xticklabels=unique_traits,
                            yticklabels=unique_traits)
                plt.title(f'BERT Confusion Matrix\nAccuracy: {bert_accuracy:.4f}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
            except Exception as e:
                print(f"Could not generate BERT confusion matrix: {e}")
                plt.text(0.5, 0.5, "Error generating BERT confusion matrix", 
                        ha='center', va='center', fontsize=12)
        else:
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, "No BERT predictions available", 
                    ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/liwc_bert_validation.png')
    
    # Analyze discrepancies only if we have predictions from both methods
    try:
        valid_predictions = big_five_df[big_five_df['liwc_prediction'].notna() & big_five_df['bert_prediction'].notna()]
        
        if not valid_predictions.empty:
            discrepancies = valid_predictions[valid_predictions['liwc_prediction'] != valid_predictions['bert_prediction']].copy()
            
            print(f"\nFound {len(discrepancies)} items where LIWC and BERT disagree")
            
            # Save the full results and discrepancies to CSV for further analysis
            big_five_df.to_csv('results/validation_results.csv', index=False)
            if not discrepancies.empty:
                discrepancies.to_csv('results/validation_discrepancies.csv', index=False)
        else:
            print("\nNo items with both LIWC and BERT predictions - cannot analyze discrepancies")
            big_five_df.to_csv('results/validation_results.csv', index=False)
    except Exception as e:
        print(f"Error analyzing discrepancies: {e}")
    
    # Identify which approach is better for each trait
    trait_performance = {}
    try:
        for trait in sorted(big_five_df['big_five_trait'].unique()):
            trait_df = big_five_df[big_five_df['big_five_trait'] == trait]
            trait_liwc_mask = trait_df['liwc_prediction'].notna()
            trait_bert_mask = trait_df['bert_prediction'].notna()
            
            liwc_trait_acc = 0
            bert_trait_acc = 0
            
            # Calculate accuracy only if we have predictions
            if trait_liwc_mask.sum() > 0:
                liwc_trait_acc = accuracy_score(
                    trait_df.loc[trait_liwc_mask, 'big_five_trait'], 
                    trait_df.loc[trait_liwc_mask, 'liwc_prediction']
                )
            
            if trait_bert_mask.sum() > 0:
                bert_trait_acc = accuracy_score(
                    trait_df.loc[trait_bert_mask, 'big_five_trait'], 
                    trait_df.loc[trait_bert_mask, 'bert_prediction']
                )
            
            better_approach = "BERT" if bert_trait_acc > liwc_trait_acc else "LIWC" if liwc_trait_acc > bert_trait_acc else "Tie"
            
            trait_performance[trait] = {
                "liwc_accuracy": liwc_trait_acc,
                "bert_accuracy": bert_trait_acc,
                "better_approach": better_approach,
                "liwc_sample_size": trait_liwc_mask.sum(),
                "bert_sample_size": trait_bert_mask.sum()
            }
    except Exception as e:
        print(f"Error calculating trait performance: {e}")
    
    print("\nPerformance by trait:")
    trait_performance_df = pd.DataFrame.from_dict(trait_performance, orient='index')
    print(trait_performance_df)
    
    # Save trait performance to CSV if we have data
    try:
        if trait_performance:
            trait_performance_df = pd.DataFrame.from_dict(trait_performance, orient='index')
            trait_performance_df.to_csv('results/trait_approach_performance.csv')
        else:
            print("No trait performance data to save")
    except Exception as e:
        print(f"Error saving trait performance data: {e}")
    
    # Generate a bar chart comparing performance by trait
    try:
        if trait_performance:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(trait_performance))
            width = 0.35
            
            traits = list(trait_performance.keys())
            liwc_accs = [trait_performance[t]["liwc_accuracy"] for t in traits]
            bert_accs = [trait_performance[t]["bert_accuracy"] for t in traits]
            
            plt.bar(x - width/2, liwc_accs, width, label='LIWC')
            plt.bar(x + width/2, bert_accs, width, label='BERT')
            
            plt.ylabel('Accuracy')
            plt.title('Approach Performance by Big Five Trait')
            plt.xticks(x, traits)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('results/trait_approach_comparison.png')
        else:
            print("Cannot generate comparison chart - no trait performance data available")
    except Exception as e:
        print(f"Error generating comparison chart: {e}")
    
    # Return a summary of findings
    return {
        "liwc_accuracy": liwc_accuracy,
        "bert_accuracy": bert_accuracy,
        "trait_performance": trait_performance,
        "num_discrepancies": len(discrepancies)
    }

if __name__ == "__main__":
    results = main()
    
    # Save the results summary
    with open('results/validation_summary.json', 'w') as f:
        # Convert any numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = {k: convert_numpy(v) if isinstance(v, (np.integer, np.floating, np.ndarray)) 
                         else {sk: convert_numpy(sv) for sk, sv in v.items()} if isinstance(v, dict)
                         else v
                         for k, v in results.items()}
        
        json.dump(json_results, f, indent=2)
    
    print("\nValidation complete. Results saved to results/ directory.")