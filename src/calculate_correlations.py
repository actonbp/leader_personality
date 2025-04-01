#!/usr/bin/env python3
"""
Calculate correlations between BERT and LIWC personality analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_trait_correlations(bert_file, liwc_file, output_dir):
    """
    Calculate correlations between BERT and LIWC personality traits
    and generate visualization
    """
    print(f"Loading BERT results from {bert_file}")
    bert_df = pd.read_csv(bert_file)
    
    print(f"Loading LIWC results from {liwc_file}")
    liwc_df = pd.read_csv(liwc_file)
    
    # Standardize name formats for matching
    bert_df['Name'] = bert_df['Name'].str.strip().str.lower()
    liwc_df['Name'] = liwc_df['Name'].str.strip().str.lower()
    
    # Count CEOs before merging
    print(f"Found {len(bert_df)} CEOs in BERT analysis")
    print(f"Found {len(liwc_df)} CEOs in LIWC analysis")
    
    # Merge datasets on CEO name
    combined_df = pd.merge(bert_df, liwc_df, on='Name', suffixes=('_bert', '_liwc'))
    print(f"Successfully matched {len(combined_df)} CEOs between both datasets")
    
    # Define the traits for correlation
    traits = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    
    # Calculate correlations between corresponding traits
    correlations = {}
    for trait in traits:
        bert_col = f"{trait}"
        liwc_col = f"{trait}"
        
        # In the merged dataframe, the columns have suffixes
        bert_col_merged = f"{trait}_bert"
        liwc_col_merged = f"{trait}_liwc"
        
        # Check if columns exist in the dataframe
        if bert_col_merged in combined_df.columns and liwc_col_merged in combined_df.columns:
            correlation = combined_df[bert_col_merged].corr(combined_df[liwc_col_merged])
            correlations[trait] = correlation
            print(f"Correlation for {trait}: {correlation:.4f}")
        else:
            missing_cols = []
            if bert_col_merged not in combined_df.columns:
                missing_cols.append(bert_col_merged)
            if liwc_col_merged not in combined_df.columns:
                missing_cols.append(liwc_col_merged)
            print(f"Warning: Cannot calculate correlation for {trait} - missing columns: {missing_cols}")
    
    # Save correlations to CSV
    correlation_df = pd.DataFrame({
        'Trait': list(correlations.keys()),
        'Correlation': list(correlations.values())
    })
    correlation_file = os.path.join(output_dir, 'trait_correlations_enhanced_weighted.csv')
    correlation_df.to_csv(correlation_file, index=False)
    print(f"Correlations saved to {correlation_file}")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.zeros((len(traits), len(traits)))
    
    for i, trait1 in enumerate(traits):
        for j, trait2 in enumerate(traits):
            bert_col = f"{trait1}_bert"
            liwc_col = f"{trait2}_liwc"
            
            if bert_col in combined_df.columns and liwc_col in combined_df.columns:
                correlation_matrix[i, j] = combined_df[bert_col].corr(combined_df[liwc_col])
            else:
                correlation_matrix[i, j] = np.nan
    
    # Create a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                xticklabels=traits, yticklabels=traits,
                vmin=-1, vmax=1, center=0, fmt='.4f')
    plt.title('Correlation Between BERT and LIWC Personality Traits')
    plt.xlabel('LIWC Traits')
    plt.ylabel('BERT Traits')
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_file = os.path.join(output_dir, 'bert_liwc_correlations_weighted.png')
    plt.savefig(heatmap_file, dpi=300)
    print(f"Correlation heatmap saved to {heatmap_file}")
    
    # Create scatter plots for each trait
    plt.figure(figsize=(15, 10))
    for i, trait in enumerate(traits):
        bert_col = f"{trait}_bert"
        liwc_col = f"{trait}_liwc"
        
        if bert_col in combined_df.columns and liwc_col in combined_df.columns:
            plt.subplot(2, 3, i+1)
            plt.scatter(combined_df[bert_col], combined_df[liwc_col], alpha=0.7)
            
            # Add regression line
            z = np.polyfit(combined_df[bert_col], combined_df[liwc_col], 1)
            p = np.poly1d(z)
            plt.plot(combined_df[bert_col], p(combined_df[bert_col]), "r--")
            
            plt.xlabel(f'BERT {trait}')
            plt.ylabel(f'LIWC {trait}')
            plt.title(f'{trait} (r={correlations.get(trait, np.nan):.4f})')
    
    plt.tight_layout()
    
    # Save the scatter plots
    scatter_file = os.path.join(output_dir, 'bert_liwc_scatterplots_weighted.png')
    plt.savefig(scatter_file, dpi=300)
    print(f"Scatter plots saved to {scatter_file}")
    
    return correlations, combined_df

if __name__ == "__main__":
    # Set file paths
    bert_file = "results/bert_enhanced_personality_analysis.csv"
    liwc_file = "results/liwc_personality_analysis_enhanced_LIWC.csv"
    output_dir = "results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlations
    correlations, combined_df = calculate_trait_correlations(bert_file, liwc_file, output_dir)
    
    # Print correlation summary
    print("\nCorrelation Summary between BERT and LIWC traits:")
    for trait, corr in correlations.items():
        print(f"{trait}: {corr:.4f}")
        
    # Save combined dataset for further analysis
    combined_file = os.path.join(output_dir, "bert_liwc_combined_data.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"\nCombined dataset saved to {combined_file}") 