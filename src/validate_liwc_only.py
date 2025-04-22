"""
Validation script to test the LIWC-style approach for personality analysis
using the IPIP dataset as ground truth, without requiring the transformer model.
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
    
    # Add column for LIWC predictions
    print("\nApplying LIWC-style analysis...")
    big_five_df['liwc_prediction'] = big_five_df['text'].apply(simple_liwc_classifier)
    
    # Calculate accuracy
    liwc_mask = big_five_df['liwc_prediction'].notna()
    
    liwc_accuracy = 0
    if liwc_mask.sum() > 0:
        liwc_accuracy = accuracy_score(
            big_five_df.loc[liwc_mask, 'big_five_trait'], 
            big_five_df.loc[liwc_mask, 'liwc_prediction']
        )
    
    print(f"\nLIWC Accuracy: {liwc_accuracy:.4f} (on {liwc_mask.sum()} items)")
    
    # Generate classification report
    if liwc_mask.sum() > 0:
        print("\nLIWC Classification Report:")
        try:
            print(classification_report(
                big_five_df.loc[liwc_mask, 'big_five_trait'], 
                big_five_df.loc[liwc_mask, 'liwc_prediction']
            ))
        except Exception as e:
            print(f"Could not generate LIWC classification report: {e}")
    
    # Create confusion matrix if we have predictions
    try:
        plt.figure(figsize=(10, 8))
        
        # Determine the unique labels
        unique_traits = sorted(big_five_df['big_five_trait'].unique())
        
        # LIWC confusion matrix
        if liwc_mask.sum() > 0:
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
            plt.text(0.5, 0.5, "No LIWC predictions available", 
                    ha='center', va='center', fontsize=12)
    
        plt.tight_layout()
        plt.savefig('results/liwc_validation.png')
    except Exception as e:
        print(f"Error creating confusion matrix visualization: {e}")
    
    # Save the full results to CSV for further analysis
    try:
        os.makedirs('results', exist_ok=True)
        big_five_df.to_csv('results/liwc_validation_results.csv', index=False)
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
    
    # Analyze trait-specific performance
    trait_performance = {}
    try:
        for trait in sorted(big_five_df['big_five_trait'].unique()):
            trait_df = big_five_df[big_five_df['big_five_trait'] == trait]
            trait_liwc_mask = trait_df['liwc_prediction'].notna()
            
            liwc_trait_acc = 0
            
            # Calculate accuracy only if we have predictions
            if trait_liwc_mask.sum() > 0:
                liwc_trait_acc = accuracy_score(
                    trait_df.loc[trait_liwc_mask, 'big_five_trait'], 
                    trait_df.loc[trait_liwc_mask, 'liwc_prediction']
                )
            
            trait_performance[trait] = {
                "accuracy": liwc_trait_acc,
                "sample_size": trait_liwc_mask.sum(),
                "total_samples": len(trait_df)
            }
    except Exception as e:
        print(f"Error calculating trait performance: {e}")
    
    print("\nPerformance by trait:")
    if trait_performance:
        trait_performance_df = pd.DataFrame.from_dict(trait_performance, orient='index')
        print(trait_performance_df)
        
        # Save trait performance to CSV
        try:
            trait_performance_df.to_csv('results/liwc_trait_performance.csv')
        except Exception as e:
            print(f"Error saving trait performance data: {e}")
        
        # Generate a bar chart of trait performance
        try:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of accuracy by trait
            traits = list(trait_performance.keys())
            accuracies = [trait_performance[t]["accuracy"] for t in traits]
            sample_sizes = [trait_performance[t]["sample_size"] for t in traits]
            
            # Plot bars
            bars = plt.bar(traits, accuracies)
            
            # Add sample size annotations
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"n={sample_sizes[i]}",
                    ha='center'
                )
            
            plt.ylabel('Accuracy')
            plt.title('LIWC-Based Classification Accuracy by Big Five Trait')
            plt.ylim(0, 1.0)
            
            plt.tight_layout()
            plt.savefig('results/liwc_trait_accuracy.png')
        except Exception as e:
            print(f"Error generating trait accuracy chart: {e}")
    else:
        print("No trait performance data available")
    
    # Return a summary of findings
    return {
        "liwc_accuracy": liwc_accuracy,
        "trait_performance": trait_performance if trait_performance else {},
        "sample_size": int(liwc_mask.sum()),
        "total_items": len(big_five_df)
    }

if __name__ == "__main__":
    results = main()
    
    # Save the results summary
    try:
        # Convert results to a JSON-serializable format
        json_results = {}
        
        # Handle basic data types
        for k, v in results.items():
            if k == "trait_performance":
                # Handle nested dictionary
                trait_perf = {}
                for trait, perf in v.items():
                    trait_perf[trait] = {
                        sk: float(sv) if isinstance(sv, (np.integer, np.floating)) else int(sv)
                        for sk, sv in perf.items()
                    }
                json_results[k] = trait_perf
            elif isinstance(v, (np.integer, np.int64)):
                json_results[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                json_results[k] = float(v)
            else:
                json_results[k] = v
        
        # Save to file
        with open('results/liwc_validation_summary.json', 'w') as f:
            json.dump(json_results, f, indent=2)
            
    except Exception as e:
        print(f"Error saving summary to JSON: {e}")
    
    print("\nValidation complete. Results saved to results/ directory.")