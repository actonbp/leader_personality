#!/usr/bin/env python3
"""
Speech Descriptives Analysis for CEO Personality Project

This script analyzes CEO speech transcripts and generates descriptive statistics
to better understand the dataset characteristics. It provides:
1. Basic statistics (word count, sentence count, etc.)
2. Word frequency analysis
3. Simple text visualizations 
4. Correlation with personality scores when available

Authors: Bryan Acton and Nan Liang
Binghamton University
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import glob

# Download required NLTK resources if they don't exist
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SpeechAnalyzer:
    """Analyzes CEO speech transcripts for descriptive statistics."""
    
    def __init__(self, data_dir=None):
        """Initialize the analyzer with the data directory."""
        self.data_dir = data_dir
        self.stop_words = set(stopwords.words('english'))
        self.results = []
        self.word_freqs = {}
        self.combined_text = ""
        
    def process_directory(self, data_dir=None, limit=None):
        """Process all speech files in the specified directory."""
        if data_dir:
            self.data_dir = data_dir
            
        if not self.data_dir:
            raise ValueError("No data directory specified")
            
        # Find all text files in the directory using os.listdir
        try:
            all_files = os.listdir(self.data_dir)
            speech_files = [os.path.join(self.data_dir, f) for f in all_files if f.endswith('.txt')]
        except Exception as e:
            print(f"Error listing directory {self.data_dir}: {e}")
            speech_files = []
        
        # Limit the number of files if specified
        if limit and limit > 0:
            speech_files = speech_files[:limit]
            
        print(f"Processing {len(speech_files)} speech files from {self.data_dir}")
        
        # Process each file
        for file_path in speech_files:
            self.process_file(file_path)
            
        # Create a DataFrame from the results
        self.result_df = pd.DataFrame(self.results)
        
        # Generate aggregate statistics
        self.generate_aggregate_stats()
        
        return self.result_df
    
    def process_file(self, file_path):
        """Process a single speech file and extract descriptive statistics."""
        try:
            # Extract CEO name from the file name
            file_name = os.path.basename(file_path)
            ceo_name = self.extract_ceo_name(file_name)
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Clean the text
            clean_text = self.clean_text(text)
            
            # Add to combined text for aggregate analysis
            self.combined_text += " " + clean_text
            
            # Basic statistics - use simpler methods to avoid NLTK issues
            words = clean_text.split()
            word_count = len(words)
            
            # Simple sentence counting by looking for sentence terminators
            sentences = re.split(r'[.!?]+', clean_text)
            sentences = [s for s in sentences if s.strip()] # Remove empty sentences
            sentence_count = len(sentences)
            
            avg_sentence_length = word_count / max(1, sentence_count)
            
            # Vocabulary richness
            unique_words = set(word.lower() for word in words
                               if word.isalpha() and word.lower() not in self.stop_words)
            vocabulary_size = len(unique_words)
            
            # Word frequency
            filtered_words = [word.lower() for word in words 
                     if word.isalpha() and word.lower() not in self.stop_words]
            word_freq = Counter(filtered_words)
            
            # Store word frequencies for this file
            self.word_freqs[ceo_name] = word_freq
            
            # Top words
            top_words = word_freq.most_common(10)
            
            # Store results
            self.results.append({
                'CEO': ceo_name,
                'File': file_name,
                'Word Count': word_count,
                'Sentence Count': sentence_count,
                'Avg Sentence Length': avg_sentence_length,
                'Vocabulary Size': vocabulary_size,
                'Lexical Diversity': vocabulary_size / max(1, word_count),
                'Top Words': top_words
            })
            
            print(f"Processed {file_name}: {word_count} words, {vocabulary_size} unique words")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    def extract_ceo_name(self, file_name):
        """Extract the CEO name from the file name."""
        # Try different patterns
        patterns = [
            r"^(\d+)?(.+?) - (.+?)\.",  # Pattern like "1Jane Doe - COMPANY.txt"
            r"^(\d+)? ?(.+?) - (.+)",   # Pattern like "1 Jane Doe - COMPANY"
            r"^(.+?)\.",                # Just get everything before the extension
        ]
        
        for pattern in patterns:
            match = re.match(pattern, file_name)
            if match:
                if len(match.groups()) >= 2:
                    # Return the name part (usually group 2, but could be group 1)
                    return match.group(2).strip()
                else:
                    # If only one group, return that
                    return match.group(1).strip()
        
        # If no pattern matches, just return the filename without extension
        return os.path.splitext(file_name)[0]
    
    def clean_text(self, text):
        """Clean and preprocess the text."""
        # Remove special characters and digits
        text = re.sub(r'\[.*?\]', '', text)  # Remove content in brackets
        text = re.sub(r'\(.*?\)', '', text)  # Remove content in parentheses
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.strip()
    
    def generate_aggregate_stats(self):
        """Generate aggregate statistics across all speeches."""
        # Initialize stats with defaults
        self.stats = {
            'Total Files': len(self.results),
            'Avg Word Count': 0,
            'Min Word Count': 0,
            'Max Word Count': 0,
            'Std Dev Word Count': 0,
            'Avg Vocabulary Size': 0,
            'Avg Lexical Diversity': 0,
            'Most Common Words': []
        }
        
        # If no results, return the default stats
        if not self.results or len(self.results) == 0:
            self.overall_word_freq = Counter()
            return self.stats
            
        # Get overall word frequency
        # Process the combined text more carefully
        if not self.combined_text.strip():
            self.overall_word_freq = Counter()
        else:
            # Split into words more safely without using punkt_tab
            all_words = [word.lower() for word in self.combined_text.split() 
                        if word.isalpha() and word.lower() not in self.stop_words]
            self.overall_word_freq = Counter(all_words)
        
        # If we have actual results, calculate real statistics
        if len(self.result_df) > 0:
            # Calculate descriptive statistics
            self.stats.update({
                'Avg Word Count': self.result_df['Word Count'].mean() if 'Word Count' in self.result_df else 0,
                'Min Word Count': self.result_df['Word Count'].min() if 'Word Count' in self.result_df else 0,
                'Max Word Count': self.result_df['Word Count'].max() if 'Word Count' in self.result_df else 0,
                'Std Dev Word Count': self.result_df['Word Count'].std() if 'Word Count' in self.result_df else 0,
                'Avg Vocabulary Size': self.result_df['Vocabulary Size'].mean() if 'Vocabulary Size' in self.result_df else 0,
                'Avg Lexical Diversity': self.result_df['Lexical Diversity'].mean() if 'Lexical Diversity' in self.result_df else 0,
                'Most Common Words': self.overall_word_freq.most_common(20)
            })
        
        return self.stats
    
    def plot_word_count_distribution(self, save_path=None):
        """Plot the distribution of word counts."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.result_df['Word Count'], kde=True)
        plt.title('Distribution of Speech Word Counts')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def plot_top_words(self, n=20, save_path=None):
        """Plot the most common words across all speeches."""
        top_words = self.overall_word_freq.most_common(n)
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(words)), counts, align='center')
        plt.yticks(range(len(words)), words)
        plt.title(f'Top {n} Words Across All CEO Speeches')
        plt.xlabel('Frequency')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def plot_lexical_diversity(self, save_path=None):
        """Plot the lexical diversity by CEO."""
        plt.figure(figsize=(12, 8))
        sorted_df = self.result_df.sort_values('Lexical Diversity', ascending=False)
        sns.barplot(x='Lexical Diversity', y='CEO', data=sorted_df.head(20))
        plt.title('Lexical Diversity by CEO (Top 20)')
        plt.xlabel('Lexical Diversity (Unique Words / Total Words)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def correlate_with_personality(self, personality_file, output_dir=None):
        """Correlate linguistic features with personality traits if available."""
        try:
            # Load personality data
            personality_df = pd.read_csv(personality_file)
            
            # Standardize CEO names for matching
            personality_df['CEO_std'] = personality_df['Name'].str.lower().str.strip()
            self.result_df['CEO_std'] = self.result_df['CEO'].str.lower().str.strip()
            
            # Merge datasets
            merged_df = pd.merge(self.result_df, personality_df, 
                                left_on='CEO_std', right_on='CEO_std', how='inner')
            
            # Check if we have matches
            if len(merged_df) == 0:
                print("No matches found between speech data and personality data")
                return None
            
            print(f"Found {len(merged_df)} CEOs with both speech and personality data")
            
            # Define features and traits for correlation
            features = ['Word Count', 'Vocabulary Size', 'Lexical Diversity', 'Avg Sentence Length']
            traits = [col for col in personality_df.columns if col in 
                     ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']]
            
            # Calculate correlations
            correlation_df = pd.DataFrame()
            for feature in features:
                for trait in traits:
                    if trait in merged_df.columns:
                        corr = merged_df[feature].corr(merged_df[trait])
                        correlation_df = correlation_df.append({
                            'Feature': feature,
                            'Trait': trait,
                            'Correlation': corr
                        }, ignore_index=True)
            
            # Save to CSV if output directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                correlation_df.to_csv(os.path.join(output_dir, 'linguistic_personality_correlations.csv'), 
                                     index=False)
            
            # Plot correlations
            plt.figure(figsize=(12, 8))
            correlation_pivot = correlation_df.pivot(index='Feature', columns='Trait', values='Correlation')
            sns.heatmap(correlation_pivot, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlations Between Linguistic Features and Personality Traits')
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'linguistic_personality_correlations.png'), dpi=300)
            
            plt.show()
            
            return correlation_df
        
        except Exception as e:
            print(f"Error correlating with personality data: {str(e)}")
            return None

def generate_quarto_report(analyzer, output_path):
    """Generate a Quarto report with the analysis results."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the Quarto document
    quarto_content = f"""---
title: "CEO Speech Descriptive Analysis"
subtitle: "A Linguistic Analysis of CEO Communications"
author: "Bryan Acton and Nan Liang"
date: today
format:
  html:
    theme: cosmo
    toc: true
    code-fold: true
---

```{{python}}
#| echo: false
#| include: false
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json

# Load pre-computed results
with open("{os.path.join(os.path.dirname(output_path), 'speech_stats.json')}", 'r') as f:
    stats = json.load(f)
    
results_df = pd.read_csv("{os.path.join(os.path.dirname(output_path), 'speech_results.csv')}")
```

## Executive Summary

This report presents a detailed linguistic analysis of {analyzer.stats['Total Files']} CEO speech transcripts. We examine speech characteristics including word counts, vocabulary richness, and linguistic patterns to better understand how CEOs communicate.

## Dataset Overview

```{{python}}
#| echo: false
print(f"Total speeches analyzed: {stats['Total Files']}")
print(f"Average word count per speech: {stats['Avg Word Count']:.1f} words")
print(f"Range of speech lengths: {stats['Min Word Count']} to {stats['Max Word Count']} words")
print(f"Average vocabulary size: {stats['Avg Vocabulary Size']:.1f} unique words")
print(f"Average lexical diversity: {stats['Avg Lexical Diversity']:.3f}")
```

## Word Count Distribution

The histogram below shows the distribution of speech lengths across all CEOs:

```{{python}}
#| echo: false
plt.figure(figsize=(10, 6))
sns.histplot(results_df['Word Count'], kde=True)
plt.title('Distribution of Speech Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()
```

## Vocabulary Richness

Lexical diversity measures the ratio of unique words to total words in a speech. Higher values indicate a more diverse vocabulary.

```{{python}}
#| echo: false
plt.figure(figsize=(12, 8))
sorted_df = results_df.sort_values('Lexical Diversity', ascending=False)
sns.barplot(x='Lexical Diversity', y='CEO', data=sorted_df.head(20))
plt.title('Lexical Diversity by CEO (Top 20)')
plt.xlabel('Lexical Diversity (Unique Words / Total Words)')
plt.tight_layout()
plt.show()
```

## Common Words Across CEO Speeches

The most frequently used words across all CEO speeches:

```{{python}}
#| echo: false
common_words = stats['Most Common Words']
words, counts = zip(*common_words)

plt.figure(figsize=(12, 8))
plt.barh(range(len(words)), counts, align='center')
plt.yticks(range(len(words)), words)
plt.title(f'Top {len(words)} Words Across All CEO Speeches')
plt.xlabel('Frequency')
plt.show()
```

## CEO Speech Characteristics

The table below shows key speech metrics for each CEO:

```{{python}}
#| echo: false
display_cols = ['CEO', 'Word Count', 'Vocabulary Size', 'Lexical Diversity', 'Avg Sentence Length']
results_df[display_cols].sort_values('Word Count', ascending=False).head(20)
```

## Analysis Implications

Based on the linguistic analysis of CEO speeches, we can draw several insights:

1. **Speech Length Variation**: There is considerable variation in how much CEOs speak, with some being significantly more verbose than others.

2. **Vocabulary Usage**: The average CEO uses approximately {analyzer.stats['Avg Vocabulary Size']:.0f} unique words in their communications.

3. **Linguistic Complexity**: The average lexical diversity score of {analyzer.stats['Avg Lexical Diversity']:.3f} suggests that CEOs tend to use relatively diverse vocabulary in their communications.

4. **Common Themes**: The most frequent words (excluding common stopwords) provide insight into the topics and concepts that CEOs emphasize.

## Next Steps

Future analysis could explore:

1. Sentiment analysis of CEO communications
2. Topic modeling to identify key themes
3. Comparison of linguistic features with company performance metrics
4. More detailed correlation with personality traits from different assessment methods

## Appendix: Data Sources

The analysis was performed on {analyzer.stats['Total Files']} CEO speech transcripts from earnings calls and public presentations.
"""
    
    # Write the Quarto document
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(quarto_content)
    
    print(f"Quarto report generated at {output_path}")

def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CEO speech transcripts for descriptive statistics')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing speech files')
    parser.add_argument('--output_dir', type=str, default='results/speech_analysis', help='Directory for output files')
    parser.add_argument('--limit', type=int, help='Limit the number of files to process')
    parser.add_argument('--personality_file', type=str, help='CSV file with personality data for correlation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the analyzer
    analyzer = SpeechAnalyzer()
    
    # Process the directory
    results = analyzer.process_directory(args.data_dir, args.limit)
    
    # Save results to CSV
    results_csv = os.path.join(args.output_dir, 'speech_results.csv')
    results_df = results.copy()
    
    # Check if we have any results before saving
    if len(results_df) > 0 and 'Top Words' in results_df.columns:
        # Convert list of tuples to string for CSV output
        results_df['Top Words'] = results_df['Top Words'].apply(lambda x: ', '.join([f"{word}:{count}" for word, count in x]))
    
    results_df.to_csv(results_csv, index=False)
    
    # Save stats to JSON
    stats_json = os.path.join(args.output_dir, 'speech_stats.json')
    stats_copy = analyzer.stats.copy()
    
    # Convert numpy types to native Python types for JSON serialization
    for key, value in stats_copy.items():
        if isinstance(value, (np.int64, np.float64)):
            stats_copy[key] = float(value)
    
    # Convert tuple to list for the most common words
    stats_copy['Most Common Words'] = [(str(word), int(count)) for word, count in stats_copy['Most Common Words']]
    
    with open(stats_json, 'w') as f:
        json.dump(stats_copy, f, indent=2)
    
    # Generate plots
    analyzer.plot_word_count_distribution(save_path=os.path.join(args.output_dir, 'word_count_distribution.png'))
    analyzer.plot_top_words(save_path=os.path.join(args.output_dir, 'top_words.png'))
    analyzer.plot_lexical_diversity(save_path=os.path.join(args.output_dir, 'lexical_diversity.png'))
    
    # Correlate with personality if file provided
    if args.personality_file:
        analyzer.correlate_with_personality(args.personality_file, args.output_dir)
    
    # Generate Quarto report
    quarto_path = os.path.join(args.output_dir, 'speech_analysis.qmd')
    generate_quarto_report(analyzer, quarto_path)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    print(f"To render the Quarto report: quarto render {quarto_path}")

if __name__ == "__main__":
    main()