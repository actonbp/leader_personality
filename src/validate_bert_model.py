"""
Validation script to evaluate the BERT model's accuracy
on IPIP personality items with known trait classifications.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging
import json
import sys
import time
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our personality analyzer
from src.personality_analyzer import PersonalityAnalyzer

def process_ipip_item(text, analyzer, verbose=False):
    """
    Process a single IPIP item through the BERT model.
    
    Args:
        text (str): The text of the IPIP item
        analyzer: The personality analyzer instance
        verbose (bool): Whether to print detailed output
    
    Returns:
        dict: The analysis results
    """
    if pd.isna(text):
        return None
    
    try:
        # Analyze the text
        results = analyzer.analyze_text(text)
        
        # Find most expressed trait
        most_expressed = max(results.items(), key=lambda x: x[1])
        most_expressed_trait = most_expressed[0]
        confidence = most_expressed[1]
        
        # Add most_expressed key for easier comparison
        results['most_expressed'] = most_expressed_trait
        results['confidence'] = confidence
        
        if verbose:
            print(f"Text: {text}")
            print(f"Most expressed trait: {most_expressed_trait} ({confidence:.4f})")
            print("All traits:", results)
            print()
            
        return results
    except Exception as e:
        if verbose:
            print(f"Error processing item: {e}")
        return None

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the BERT model
    print("Initializing BERT model...")
    analyzer = PersonalityAnalyzer()
    
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
        'Emotional Stability': 'Neuroticism'  # Inverse of Neuroticism - need to invert predicted score
    }
    
    # Create a new column for mapped Big Five traits
    # Handle NaN values
    def map_to_big_five(label):
        if pd.isna(label):
            return 'Other'
        try:
            # Try to find a matching Big Five trait
            for key in big_five_mapping:
                if key in label:
                    return big_five_mapping[key]
            return 'Other'
        except TypeError:
            # Handle any other type errors
            return 'Other'
    
    ipip_df['big_five_trait'] = ipip_df['label'].apply(map_to_big_five)
    
    # Filter to only include items that map to Big Five traits
    big_five_df = ipip_df[ipip_df['big_five_trait'] != 'Other'].copy()
    
    # For 'Emotional Stability' items, we need to invert the Neuroticism prediction
    # since Emotional Stability is the opposite of Neuroticism
    big_five_df['invert_neuroticism'] = ipip_df['label'].apply(
        lambda x: 'Emotional Stability' in x if isinstance(x, str) else False
    )
    
    print(f"Total IPIP items: {len(ipip_df)}")
    print(f"Big Five items: {len(big_five_df)}")
    print(f"Distribution of Big Five traits:")
    print(big_five_df['big_five_trait'].value_counts())
    
    # Process IPIP items with BERT
    print("\nProcessing IPIP items with BERT...")
    
    # Limit sample size for faster testing if needed
    # Adjust sample_size to None to process all items
    sample_size = None
    
    if sample_size is not None:
        # Stratified sampling to maintain distribution
        sampled_df = big_five_df.groupby('big_five_trait', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(big_five_df)))))
        )
        print(f"Using {len(sampled_df)} items (sampled from {len(big_five_df)})")
        big_five_df = sampled_df
    
    # Process items with progress tracking
    start_time = time.time()
    
    bert_results = []
    for idx, row in tqdm(big_five_df.iterrows(), total=len(big_five_df), desc="Analyzing items"):
        result = process_ipip_item(row['text'], analyzer)
        bert_results.append(result)
    
    processing_time = time.time() - start_time
    
    # Add results to dataframe
    big_five_df['bert_results'] = bert_results
    
    # Extract most expressed trait
    big_five_df['bert_prediction'] = big_five_df['bert_results'].apply(
        lambda x: x['most_expressed'] if x is not None else None
    )
    
    # Extract confidence
    big_five_df['bert_confidence'] = big_five_df['bert_results'].apply(
        lambda x: x['confidence'] if x is not None else None
    )
    
    # Extract all trait scores
    for trait in analyzer.label_names:
        big_five_df[f'bert_{trait.lower()}'] = big_five_df['bert_results'].apply(
            lambda x: x[trait] if x is not None else None
        )
    
    # Calculate accuracy
    bert_mask = big_five_df['bert_prediction'].notna()
    
    bert_accuracy = 0
    if bert_mask.sum() > 0:
        bert_accuracy = accuracy_score(
            big_five_df.loc[bert_mask, 'big_five_trait'], 
            big_five_df.loc[bert_mask, 'bert_prediction']
        )
    
    print(f"\nBERT processing time: {processing_time:.2f} seconds for {len(big_five_df)} items")
    print(f"BERT Accuracy: {bert_accuracy:.4f} (on {bert_mask.sum()} items)")
    
    # Generate classification report
    if bert_mask.sum() > 0:
        print("\nBERT Classification Report:")
        try:
            print(classification_report(
                big_five_df.loc[bert_mask, 'big_five_trait'], 
                big_five_df.loc[bert_mask, 'bert_prediction']
            ))
        except Exception as e:
            print(f"Could not generate BERT classification report: {e}")
    
    # Create confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        
        # BERT confusion matrix
        if bert_mask.sum() > 0:
            try:
                bert_cm = confusion_matrix(
                    big_five_df.loc[bert_mask, 'big_five_trait'], 
                    big_five_df.loc[bert_mask, 'bert_prediction'],
                    labels=sorted(big_five_df['big_five_trait'].unique())
                )
                sns.heatmap(bert_cm, annot=True, fmt='d', cmap='Greens',
                            xticklabels=sorted(big_five_df['big_five_trait'].unique()),
                            yticklabels=sorted(big_five_df['big_five_trait'].unique()))
                plt.title(f'BERT Confusion Matrix\nAccuracy: {bert_accuracy:.4f}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
            except Exception as e:
                print(f"Could not generate BERT confusion matrix: {e}")
                plt.text(0.5, 0.5, "Error generating BERT confusion matrix", 
                        ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, "No BERT predictions available", 
                    ha='center', va='center', fontsize=12)
    
        plt.tight_layout()
        plt.savefig('results/bert_validation.png')
    except Exception as e:
        print(f"Error creating confusion matrix visualization: {e}")
    
    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    
    # Create a version of the dataframe without the complex dict column for CSV export
    export_df = big_five_df.drop(columns=['bert_results'])
    export_df.to_csv('results/bert_validation_results.csv', index=False)
    
    # Analyze trait-specific performance
    trait_performance = {}
    try:
        for trait in sorted(big_five_df['big_five_trait'].unique()):
            trait_df = big_five_df[big_five_df['big_five_trait'] == trait]
            trait_bert_mask = trait_df['bert_prediction'].notna()
            
            bert_trait_acc = 0
            if trait_bert_mask.sum() > 0:
                bert_trait_acc = accuracy_score(
                    trait_df.loc[trait_bert_mask, 'big_five_trait'], 
                    trait_df.loc[trait_bert_mask, 'bert_prediction']
                )
            
            # Calculate average confidence for correct and incorrect predictions
            correct_mask = (trait_df['big_five_trait'] == trait_df['bert_prediction']) & trait_bert_mask
            incorrect_mask = (trait_df['big_five_trait'] != trait_df['bert_prediction']) & trait_bert_mask
            
            avg_correct_confidence = trait_df.loc[correct_mask, 'bert_confidence'].mean() if correct_mask.sum() > 0 else 0
            avg_incorrect_confidence = trait_df.loc[incorrect_mask, 'bert_confidence'].mean() if incorrect_mask.sum() > 0 else 0
            
            trait_performance[trait] = {
                "accuracy": bert_trait_acc,
                "sample_size": trait_bert_mask.sum(),
                "total_samples": len(trait_df),
                "avg_correct_confidence": avg_correct_confidence,
                "avg_incorrect_confidence": avg_incorrect_confidence
            }
    except Exception as e:
        print(f"Error calculating trait performance: {e}")
    
    print("\nPerformance by trait:")
    trait_performance_df = pd.DataFrame.from_dict(trait_performance, orient='index')
    print(trait_performance_df)
    
    # Save trait performance to CSV
    trait_performance_df.to_csv('results/bert_trait_performance.csv')
    
    # Generate a bar chart of trait performance
    try:
        plt.figure(figsize=(12, 6))
        
        # Create bar chart of accuracy by trait
        traits = list(trait_performance.keys())
        accuracies = [trait_performance[t]["accuracy"] for t in traits]
        
        # Plot accuracy bars
        bars = plt.bar(traits, accuracies)
        
        # Add sample size and confidence annotations
        for i, bar in enumerate(bars):
            sample_size = trait_performance[traits[i]]["sample_size"]
            avg_conf = trait_performance[traits[i]]["avg_correct_confidence"]
            
            # Add sample size
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"n={sample_size}",
                ha='center'
            )
            
            # Add confidence
            if avg_conf > 0:
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height()/2,
                    f"conf={avg_conf:.2f}",
                    ha='center', color='white', fontweight='bold'
                )
        
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.title('BERT Classification Accuracy by Big Five Trait')
        
        plt.tight_layout()
        plt.savefig('results/bert_trait_accuracy.png')
    except Exception as e:
        print(f"Error generating trait accuracy chart: {e}")
    
    # Return a summary of findings
    return {
        "bert_accuracy": bert_accuracy,
        "trait_performance": trait_performance,
        "sample_size": int(bert_mask.sum()),
        "total_items": len(big_five_df),
        "processing_time": processing_time
    }

if __name__ == "__main__":
    try:
        results = main()
        
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
        with open('results/bert_validation_summary.json', 'w') as f:
            json.dump(json_results, f, indent=2)
            
    except Exception as e:
        print(f"Error during validation: {e}")
    
    print("\nValidation complete. Results saved to results/ directory.")