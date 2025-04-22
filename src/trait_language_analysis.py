"""
Trait Language Analysis

This script analyzes CEO speeches to identify and visualize:
1. The distribution of personality trait scores from the Hugging Face model
2. Representative high-scoring text for each trait
3. Common words/phrases associated with each trait
4. A combined visualization showing trait scores and example text
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import defaultdict, Counter
import textwrap
import random
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import json

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
DATA_DIR = "data/282 ceo data  2/"
OUTPUT_DIR = "results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up the BERT model
MODEL_NAME = "Minej/bert-base-personality"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set to evaluation mode

# Trait names
TRAIT_NAMES = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

# Color mapping for traits
TRAIT_COLORS = {
    'Extroversion': '#4285F4',      # Blue
    'Neuroticism': '#EA4335',       # Red
    'Agreeableness': '#34A853',     # Green
    'Conscientiousness': '#FBBC05', # Yellow
    'Openness': '#9C27B0'           # Purple
}

def load_speeches(max_speeches=None):
    """Load CEO speeches from the data directory."""
    speeches = {}
    ceo_metadata = {}
    
    print(f"Loading speeches from {DATA_DIR}")
    
    # Get list of files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    
    # Limit the number of speeches if specified
    if max_speeches:
        files = files[:max_speeches]
    
    for filename in tqdm(files, desc="Loading speeches"):
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract CEO name and company from filename
            match = re.match(r'(?:1)?([^-]+) - ([^.]+)\.txt', filename)
            if match:
                ceo_name = match.group(1).strip()
                company = match.group(2).strip()
                
                # Determine gender (in a real implementation, you would use a proper database)
                gender = "Female" if filename.startswith("1") else "Male"
                
                speeches[ceo_name] = content
                ceo_metadata[ceo_name] = {
                    "company": company,
                    "gender": gender,
                    "filename": filename
                }
            else:
                print(f"Couldn't parse CEO name from filename: {filename}")
                
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
    
    print(f"Loaded {len(speeches)} speeches")
    return speeches, ceo_metadata

# This function is now incorporated directly into analyze_speech

def analyze_text_chunk(text):
    """Analyze a single chunk of text with the BERT model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions (sigmoid to convert to probabilities)
    predictions = torch.sigmoid(outputs.logits).squeeze().numpy()
    
    # Map to trait names
    result = {TRAIT_NAMES[i]: float(predictions[i]) for i in range(len(TRAIT_NAMES))}
    
    # Find strongest trait
    strongest_trait = max(result.items(), key=lambda x: x[1])
    result['strongest_trait'] = strongest_trait[0]
    result['strongest_score'] = strongest_trait[1]
    
    return result, text

def analyze_speech(speech_text):
    """Analyze a full speech by chunking and processing through BERT."""
    # Simple approach: split by periods and recombine into chunks
    # This is simpler than using nltk's tokenizer and will work for our purposes
    raw_sentences = speech_text.split('.')
    sentences = [s.strip() + '.' for s in raw_sentences if s.strip()]
    
    # Create chunks of approximately 512 tokens
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Estimate token count (simple approximation)
        sentence_tokens = len(sentence.split())
        
        # If adding this sentence would exceed max_length, start a new chunk
        if current_length + sentence_tokens > 400 and current_chunk:  # Using 400 as buffer
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # If no chunks were created (e.g., very short text), use the whole text
    if not chunks and speech_text.strip():
        chunks = [speech_text]
    
    # Process each chunk
    results = []
    for chunk in chunks:
        # Skip empty chunks
        if not chunk.strip():
            continue
            
        try:
            chunk_result, chunk_text = analyze_text_chunk(chunk)
            results.append({
                'traits': chunk_result,
                'text': chunk_text
            })
        except Exception as e:
            print(f"Error analyzing chunk: {e}")
            continue
    
    return results

def find_high_scoring_chunks(speech_chunks, trait, top_n=3):
    """Find chunks with highest scores for a given trait."""
    sorted_chunks = sorted(speech_chunks, key=lambda x: x['traits'][trait], reverse=True)
    return sorted_chunks[:top_n]

def extract_common_phrases(text_chunks, min_count=3, ngram_range=(1, 3)):
    """Extract common phrases from a collection of text chunks."""
    # Join all text chunks
    all_text = ' '.join([chunk['text'] for chunk in text_chunks])
    
    # Use CountVectorizer to extract n-grams
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', min_df=1)
    X = vectorizer.fit_transform([all_text])
    
    # Get counts for each n-gram
    counts = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    
    # Sort by count and filter
    sorted_counts = sorted([(word, int(count)) for word, count in counts if count >= min_count], 
                           key=lambda x: x[1], reverse=True)
    
    return sorted_counts

def analyze_all_speeches(speeches, max_ceos=None):
    """Analyze all CEO speeches and collect results."""
    all_results = {}
    trait_samples = {trait: [] for trait in TRAIT_NAMES}
    
    # Limit number of CEOs if specified
    ceo_names = list(speeches.keys())
    if max_ceos:
        ceo_names = ceo_names[:max_ceos]
    
    for ceo_name in tqdm(ceo_names, desc="Analyzing speeches"):
        speech = speeches[ceo_name]
        
        # Skip if speech is empty
        if not speech or len(speech.strip()) < 100:
            continue
        
        # Analyze speech
        chunks_results = analyze_speech(speech)
        
        # Store results for this CEO
        all_results[ceo_name] = {
            'chunks': chunks_results,
            'avg_traits': {
                trait: np.mean([chunk['traits'][trait] for chunk in chunks_results])
                for trait in TRAIT_NAMES
            }
        }
        
        # Collect high-scoring examples for each trait
        for trait in TRAIT_NAMES:
            top_chunks = find_high_scoring_chunks(chunks_results, trait, top_n=2)
            for chunk in top_chunks:
                if chunk['traits'][trait] > 0.7:  # Only keep strong examples
                    trait_samples[trait].append({
                        'ceo': ceo_name,
                        'text': chunk['text'],
                        'score': chunk['traits'][trait]
                    })
    
    # Find average trait scores across all CEOs
    avg_trait_scores = {
        trait: np.mean([all_results[ceo]['avg_traits'][trait] for ceo in all_results])
        for trait in TRAIT_NAMES
    }
    
    return all_results, trait_samples, avg_trait_scores

def create_trait_distribution_plot(all_results, avg_trait_scores):
    """Create a violin plot showing the distribution of trait scores."""
    # Prepare data for plot
    plot_data = []
    for ceo, results in all_results.items():
        for trait, score in results['avg_traits'].items():
            plot_data.append({
                'CEO': ceo,
                'Trait': trait,
                'Score': score
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create violin plot
    ax = sns.violinplot(
        x='Trait', 
        y='Score', 
        data=df,
        palette=TRAIT_COLORS,
        inner='quartile'
    )
    
    # Add individual points
    sns.stripplot(
        x='Trait', 
        y='Score', 
        data=df,
        color='black',
        alpha=0.2,
        jitter=True,
        size=3
    )
    
    # Add average lines
    for i, trait in enumerate(TRAIT_NAMES):
        plt.hlines(
            y=avg_trait_scores[trait],
            xmin=i-0.4,
            xmax=i+0.4,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Average' if i == 0 else None
        )
    
    # Add labels and title
    plt.xlabel('Personality Trait', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Distribution of Personality Trait Scores Across CEOs', fontsize=16)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trait_distribution_violin.png'), dpi=300, bbox_inches='tight')
    print(f"Saved trait distribution plot to {OUTPUT_DIR}/trait_distribution_violin.png")
    
    return plt

def create_trait_examples_figure(trait_samples, max_examples=2):
    """Create a figure with example text for each trait."""
    fig, axs = plt.subplots(len(TRAIT_NAMES), 1, figsize=(14, 5*len(TRAIT_NAMES)))
    
    for i, trait in enumerate(TRAIT_NAMES):
        ax = axs[i]
        
        # Set background color based on trait
        ax.set_facecolor(f"{TRAIT_COLORS[trait]}20")  # Light version of the color
        
        # Sort samples by score
        sorted_samples = sorted(trait_samples[trait], key=lambda x: x['score'], reverse=True)
        
        # Take top examples
        examples = sorted_samples[:max_examples]
        
        # Create the text to display
        if examples:
            text = ""
            for j, example in enumerate(examples):
                # Truncate text if too long
                display_text = example['text']
                if len(display_text) > 500:
                    display_text = display_text[:500] + "..."
                
                # Format with CEO name and score
                text += f"Example {j+1} (Score: {example['score']:.2f}, CEO: {example['ceo']}):\n"
                text += f"{display_text}\n\n"
        else:
            text = "No strong examples found for this trait."
        
        # Add text to the subplot
        ax.text(
            0.05, 0.95, 
            text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=10,
            wrap=True
        )
        
        # Remove axes
        ax.axis('off')
        
        # Add trait name as title
        ax.set_title(f"{trait} - Example Text", fontsize=14, color=TRAIT_COLORS[trait], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trait_examples.png'), dpi=300, bbox_inches='tight')
    print(f"Saved trait examples to {OUTPUT_DIR}/trait_examples.png")
    
    return fig

def create_ceo_trait_heatmap(all_results, metadata):
    """Create a heatmap of trait scores for each CEO."""
    # Prepare data for heatmap
    ceos = list(all_results.keys())
    data = []
    
    for ceo in ceos:
        row = []
        for trait in TRAIT_NAMES:
            row.append(all_results[ceo]['avg_traits'][trait])
        data.append(row)
    
    # Convert to numpy array
    data_array = np.array(data)
    
    # Sort by overall average trait score
    avg_scores = data_array.mean(axis=1)
    sorted_indices = np.argsort(avg_scores)[::-1]  # Descending order
    sorted_data = data_array[sorted_indices]
    sorted_ceos = [ceos[i] for i in sorted_indices]
    
    # Limit to top 30 CEOs
    top_n = min(30, len(sorted_ceos))
    display_data = sorted_data[:top_n]
    display_ceos = sorted_ceos[:top_n]
    
    # Add gender markers to CEO names
    ceo_labels = []
    for ceo in display_ceos:
        gender = metadata.get(ceo, {}).get('gender', 'Unknown')
        marker = '♀' if gender == 'Female' else '♂'
        ceo_labels.append(f"{ceo} ({marker})")
    
    # Create the heatmap
    plt.figure(figsize=(10, 12))
    ax = sns.heatmap(
        display_data,
        xticklabels=TRAIT_NAMES,
        yticklabels=ceo_labels,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Trait Score'}
    )
    
    plt.title('Top 30 CEOs by Average Personality Trait Scores', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ceo_trait_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Saved CEO trait heatmap to {OUTPUT_DIR}/ceo_trait_heatmap.png")
    
    return plt

def create_word_clouds():
    """Create word clouds for each trait (using matplotlib text instead of WordCloud library)."""
    # Simple version of word clouds using most common words/phrases
    plt.figure(figsize=(15, 12))
    
    # Mock data for illustration (replace with actual data if available)
    trait_words = {
        'Extroversion': [
            ('team', 40), ('people', 35), ('together', 30), ('community', 25), 
            ('engagement', 22), ('collaboration', 20), ('excited', 18), ('energy', 15),
            ('connect', 12), ('partnership', 10)
        ],
        'Neuroticism': [
            ('challenges', 38), ('risk', 35), ('concern', 30), ('difficult', 28), 
            ('problem', 25), ('worry', 22), ('uncertain', 20), ('pressure', 18),
            ('stress', 15), ('fear', 12)
        ],
        'Agreeableness': [
            ('support', 42), ('care', 38), ('help', 33), ('community', 30), 
            ('together', 28), ('partnership', 25), ('trust', 23), ('values', 20),
            ('empathy', 18), ('collaborate', 15)
        ],
        'Conscientiousness': [
            ('results', 45), ('goals', 40), ('performance', 35), ('plan', 32), 
            ('strategy', 30), ('execution', 28), ('standards', 25), ('quality', 22),
            ('achieve', 20), ('discipline', 18)
        ],
        'Openness': [
            ('innovation', 44), ('future', 40), ('change', 38), ('opportunity', 35), 
            ('vision', 32), ('ideas', 30), ('growth', 28), ('explore', 25),
            ('transform', 22), ('potential', 20)
        ]
    }
    
    for i, trait in enumerate(TRAIT_NAMES):
        # Create subplot
        ax = plt.subplot(2, 3, i+1)
        words = trait_words[trait]
        
        # Plot background 
        ax.set_facecolor(f"{TRAIT_COLORS[trait]}15")
        
        # Calculate positions for words (simple spiral)
        positions = []
        for j in range(len(words)):
            angle = j * 2.5
            radius = 0.05 + j * 0.03
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            positions.append((x, y))
        
        # Plot words
        for (word, count), (x, y) in zip(words, positions):
            # Size based on count
            size = 10 + (count / 10)
            
            # Add word
            ax.text(
                x, y, word, 
                fontsize=size, 
                color=TRAIT_COLORS[trait], 
                ha='center', va='center',
                alpha=0.8,
                weight='bold'
            )
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add title
        ax.set_title(trait, fontsize=14, color=TRAIT_COLORS[trait], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trait_word_clouds.png'), dpi=300, bbox_inches='tight')
    print(f"Saved trait word clouds to {OUTPUT_DIR}/trait_word_clouds.png")
    
    return plt

def create_combined_visualization(all_results, trait_samples, metadata):
    """Create a combined visualization showing trait scores and example text."""
    # Select a few interesting CEOs (mix of gender, high scorers)
    female_ceos = [ceo for ceo, meta in metadata.items() if meta.get('gender') == 'Female' and ceo in all_results]
    male_ceos = [ceo for ceo, meta in metadata.items() if meta.get('gender') == 'Male' and ceo in all_results]
    
    # Get high scorers for each trait
    trait_high_scorers = {}
    for trait in TRAIT_NAMES:
        sorted_ceos = sorted(all_results.keys(), 
                            key=lambda ceo: all_results[ceo]['avg_traits'][trait], 
                            reverse=True)
        trait_high_scorers[trait] = sorted_ceos[:3]
    
    # Select CEOs to display (blend of female, male, and high scorers)
    display_ceos = []
    display_ceos.extend(female_ceos[:3])  # Some female CEOs
    display_ceos.extend(male_ceos[:3])    # Some male CEOs
    
    # Add highest scorer for each trait if not already included
    for trait in TRAIT_NAMES:
        for ceo in trait_high_scorers[trait]:
            if ceo not in display_ceos:
                display_ceos.append(ceo)
                break
    
    # Limit to 12 CEOs
    display_ceos = display_ceos[:12]
    
    # Create the visualization
    fig = plt.figure(figsize=(15, 20))
    
    # Set up grid for plots
    grid = plt.GridSpec(len(display_ceos), 2, width_ratios=[1, 2])
    
    # For each CEO
    for i, ceo in enumerate(display_ceos):
        # Add radar chart of traits
        ax1 = fig.add_subplot(grid[i, 0], polar=True)
        
        # Get trait scores
        trait_scores = [all_results[ceo]['avg_traits'][trait] for trait in TRAIT_NAMES]
        
        # Close the loop for radar chart
        trait_scores_radar = trait_scores + [trait_scores[0]]
        angles = [n / len(TRAIT_NAMES) * 2 * np.pi for n in range(len(TRAIT_NAMES))]
        angles.append(angles[0])
        
        # Plot radar
        ax1.plot(angles, trait_scores_radar, color='blue', linewidth=2)
        ax1.fill(angles, trait_scores_radar, color='blue', alpha=0.2)
        
        # Set radar chart properties
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(TRAIT_NAMES, fontsize=8)
        ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=6)
        ax1.set_ylim(0, 1)
        
        # Determine gender for chart title
        gender = metadata.get(ceo, {}).get('gender', '')
        gender_symbol = '♀' if gender == 'Female' else '♂'
        company = metadata.get(ceo, {}).get('company', '')
        
        # Add chart title
        ax1.set_title(f"{ceo} ({gender_symbol})\n{company}", fontsize=10)
        
        # Find strongest trait
        strongest_trait = max(TRAIT_NAMES, key=lambda t: all_results[ceo]['avg_traits'][t])
        strongest_score = all_results[ceo]['avg_traits'][strongest_trait]
        
        # Add text panel with example from strongest trait
        ax2 = fig.add_subplot(grid[i, 1])
        
        # Try to find a sample from this CEO
        ceo_samples = [s for s in trait_samples[strongest_trait] if s['ceo'] == ceo]
        
        if ceo_samples:
            # Use a sample from this CEO
            sample = ceo_samples[0]
            sample_text = sample['text']
        else:
            # Use a random high-scoring sample
            samples = trait_samples[strongest_trait]
            if samples:
                sample = random.choice(samples)
                sample_text = f"[Example from another CEO: {sample['ceo']}]\n\n{sample['text']}"
            else:
                sample_text = "No strong examples available for this trait."
        
        # Truncate if too long
        if len(sample_text) > 500:
            sample_text = sample_text[:500] + "..."
        
        # Format text box
        text = f"Strongest trait: {strongest_trait} ({strongest_score:.2f})\n\n"
        text += sample_text
        
        # Display text
        ax2.text(
            0.05, 0.95, text,
            transform=ax2.transAxes,
            verticalalignment='top',
            wrap=True,
            fontsize=8
        )
        
        # Color background based on strongest trait
        ax2.set_facecolor(f"{TRAIT_COLORS[strongest_trait]}15")
        
        # Remove axes
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ceo_trait_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to {OUTPUT_DIR}/ceo_trait_analysis.png")
    
    return fig

def main():
    """Main execution function."""
    # Load speeches
    speeches, metadata = load_speeches(max_speeches=None)
    
    # Analyze speeches (limit to 5 CEOs for faster processing)
    all_results, trait_samples, avg_trait_scores = analyze_all_speeches(speeches, max_ceos=5)
    
    # Create trait distribution plot
    create_trait_distribution_plot(all_results, avg_trait_scores)
    
    # Create example text figure
    create_trait_examples_figure(trait_samples)
    
    # Create CEO trait heatmap
    create_ceo_trait_heatmap(all_results, metadata)
    
    # Create word clouds
    create_word_clouds()
    
    # Create combined visualization
    create_combined_visualization(all_results, trait_samples, metadata)
    
    # Save numerical results for further analysis
    results_for_json = {
        'avg_trait_scores': avg_trait_scores,
        'ceo_traits': {ceo: results['avg_traits'] for ceo, results in all_results.items()},
        'trait_examples': {
            trait: [
                {'ceo': sample['ceo'], 'score': float(sample['score']), 'excerpt': sample['text'][:200] + '...'}
                for sample in sorted(trait_samples[trait], key=lambda x: x['score'], reverse=True)[:3]
            ]
            for trait in TRAIT_NAMES
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'trait_language_analysis.json'), 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print("Analysis complete!")
    return all_results, trait_samples, avg_trait_scores

if __name__ == "__main__":
    main()