#!/usr/bin/env python3
"""
Print simple correlations between BERT and LIWC personality traits
"""

import pandas as pd

# Load the combined dataset
df = pd.read_csv('results/bert_liwc_combined_data.csv')

# Calculate correlations between matching traits
traits = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
print('\nCorrelations between BERT and LIWC traits:')
print('-----------------------------------------')
for trait in traits:
    bert_col = f'{trait}_bert'
    liwc_col = f'{trait}_liwc'
    corr = df[bert_col].corr(df[liwc_col])
    print(f'{trait}: {corr:.4f}')

# Also calculate correlations between all trait combinations
print('\nFull correlation matrix:')
print('----------------------')
corr_cols = [f'{trait}_bert' for trait in traits] + [f'{trait}_liwc' for trait in traits]
corr_matrix = df[corr_cols].corr()

# Print only the cross-correlations (BERT vs LIWC)
print('\nBERT traits (rows) vs LIWC traits (columns):')
print('-------------------------------------------')
bert_traits = [f'{trait}_bert' for trait in traits]
liwc_traits = [f'{trait}_liwc' for trait in traits]
cross_corr = corr_matrix.loc[bert_traits, liwc_traits]

# Print with better formatting
print(' ' * 18, end='')
for liwc in traits:
    print(f'{liwc:>12}', end='')
print()

for bert_trait, bert_full in zip(traits, bert_traits):
    print(f'{bert_trait:15}', end='')
    for liwc_full in liwc_traits:
        print(f'{corr_matrix.loc[bert_full, liwc_full]:12.4f}', end='')
    print() 