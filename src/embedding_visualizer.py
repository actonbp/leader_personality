#!/usr/bin/env python3
"""
CEO Speech Embedding Visualization Tool

This script generates and visualizes embeddings from CEO speeches using
pre-trained language models. It helps identify patterns and clusters in
how CEOs communicate, which can provide insight into why different
personality assessment methods might diverge.

Authors: Bryan Acton and Nan Liang
Binghamton University
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import re
import random
from collections import defaultdict

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class EmbeddingVisualizer:
    """Generates and visualizes embeddings from CEO speeches."""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """Initialize with the specified embedding model."""
        self.model_name = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.embeddings = {}
        self.metadata = []
        
    def process_directory(self, data_dir, limit=None, chunk_size=512):
        """Process all speech files in the directory, chunking them into manageable pieces."""
        # Find all text files
        speech_files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        # Limit if specified
        if limit and limit > 0:
            speech_files = speech_files[:limit]
            
        print(f"Processing {len(speech_files)} files from {data_dir}")
        
        # Process each file
        for file_path in speech_files:
            try:
                self.process_file(file_path, chunk_size)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                
        # Convert metadata to DataFrame
        self.metadata_df = pd.DataFrame(self.metadata)
        
        print(f"Generated embeddings for {len(self.metadata)} text chunks from {len(speech_files)} files")
        return self.metadata_df
    
    def process_file(self, file_path, chunk_size=512):
        """Process a single file, chunking the text and generating embeddings."""
        # Extract CEO name from filename
        file_name = os.path.basename(file_path)
        ceo_name = self.extract_ceo_name(file_name)
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean text
        clean_text = self.clean_text(text)
        
        # Chunk the text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean_text)
        
        # Group sentences into chunks of approximately chunk_size words
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Generate embeddings for each chunk
        for i, chunk in enumerate(chunks):
            # Skip very short chunks
            if len(chunk.split()) < 20:
                continue
                
            # Generate embedding
            embedding = self.model.encode(chunk)
            
            # Store embedding and metadata
            chunk_id = f"{ceo_name}_{i}"
            self.embeddings[chunk_id] = embedding
            
            self.metadata.append({
                'ID': chunk_id,
                'CEO': ceo_name,
                'File': file_name,
                'Chunk': i,
                'Text_Length': len(chunk.split()),
                'Text_Preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
            })
    
    def extract_ceo_name(self, file_name):
        """Extract CEO name from filename."""
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
        """Clean and normalize text."""
        text = re.sub(r'\[.*?\]', '', text)  # Remove content in brackets
        text = re.sub(r'\(.*?\)', '', text)  # Remove content in parentheses
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.strip()
    
    def run_dimensionality_reduction(self, method='tsne', perplexity=30, n_components=2):
        """Apply dimensionality reduction to the embeddings."""
        # Extract all embeddings as a matrix
        embedding_ids = list(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[id] for id in embedding_ids])
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            reduced_embeddings = reducer.fit_transform(embedding_matrix)
            
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embedding_matrix)
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Create a dictionary mapping each ID to its reduced embedding
        self.reduced_embeddings = {embedding_ids[i]: reduced_embeddings[i] for i in range(len(embedding_ids))}
        
        # Add reduced coordinates to metadata
        for i, row in enumerate(self.metadata):
            id = row['ID']
            if id in self.reduced_embeddings:
                coords = self.reduced_embeddings[id]
                for j, coord in enumerate(coords):
                    self.metadata[i][f"{method.upper()}_{j+1}"] = coord
        
        # Update the DataFrame
        self.metadata_df = pd.DataFrame(self.metadata)
        
        return self.reduced_embeddings
    
    def plot_embeddings(self, color_by='CEO', method='tsne', save_path=None, top_n=10, include_names=True):
        """Plot the reduced embeddings, colored by the specified attribute."""
        # Ensure dimensionality reduction has been run
        if not hasattr(self, 'reduced_embeddings'):
            print(f"Running {method} dimensionality reduction...")
            self.run_dimensionality_reduction(method=method)
        
        # Get unique values for coloring
        if color_by == 'CEO':
            all_values = self.metadata_df['CEO'].unique()
            # Limit to top N CEOs by number of chunks if there are many
            if len(all_values) > top_n:
                ceo_counts = self.metadata_df['CEO'].value_counts()
                top_values = ceo_counts.head(top_n).index.tolist()
                filtered_df = self.metadata_df[self.metadata_df['CEO'].isin(top_values)]
            else:
                top_values = all_values
                filtered_df = self.metadata_df
        else:
            top_values = self.metadata_df[color_by].unique()
            filtered_df = self.metadata_df
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot each group with a different color
        for i, value in enumerate(top_values):
            group_df = filtered_df[filtered_df[color_by] == value]
            
            x_col = f"{method.upper()}_1"
            y_col = f"{method.upper()}_2"
            
            plt.scatter(
                group_df[x_col], 
                group_df[y_col], 
                label=value,
                alpha=0.7,
                s=50
            )
            
            # Add CEO names as annotations if requested
            if include_names and color_by == 'CEO':
                for _, row in group_df.iterrows():
                    plt.annotate(
                        row['CEO'],
                        (row[x_col], row[y_col]),
                        fontsize=8,
                        alpha=0.8,
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
        
        plt.title(f"{method.upper()} Visualization of CEO Speech Embeddings (colored by {color_by})")
        
        # Add legend if there aren't too many values
        if len(top_values) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
        plt.show()
    
    def find_closest_speeches(self, ceo_name, n=5):
        """Find the most similar speeches to the given CEO."""
        # Get the average embedding for the CEO
        ceo_embeddings = [self.embeddings[row['ID']] for row in self.metadata if row['CEO'] == ceo_name]
        
        if not ceo_embeddings:
            print(f"No embeddings found for CEO: {ceo_name}")
            return None
            
        ceo_avg_embedding = np.mean(ceo_embeddings, axis=0)
        
        # Calculate distances to all other CEO average embeddings
        ceo_distances = {}
        for other_ceo in self.metadata_df['CEO'].unique():
            if other_ceo == ceo_name:
                continue
                
            other_embeddings = [self.embeddings[row['ID']] for row in self.metadata if row['CEO'] == other_ceo]
            if not other_embeddings:
                continue
                
            other_avg_embedding = np.mean(other_embeddings, axis=0)
            distance = np.linalg.norm(ceo_avg_embedding - other_avg_embedding)
            ceo_distances[other_ceo] = distance
        
        # Sort by distance
        sorted_ceos = sorted(ceo_distances.items(), key=lambda x: x[1])
        return sorted_ceos[:n]
    
    def generate_clusters(self, n_clusters=5, method='kmeans'):
        """Cluster the embeddings to identify patterns."""
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        # Extract all embeddings as a matrix
        embedding_ids = list(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[id] for id in embedding_ids])
        
        # Apply clustering
        if method.lower() == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(embedding_matrix)
        elif method.lower() == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(embedding_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Add cluster labels to metadata
        cluster_map = {embedding_ids[i]: labels[i] for i in range(len(embedding_ids))}
        for i, row in enumerate(self.metadata):
            id = row['ID']
            if id in cluster_map:
                self.metadata[i]['Cluster'] = int(cluster_map[id])
        
        # Update the DataFrame
        self.metadata_df = pd.DataFrame(self.metadata)
        
        # Analyze clusters
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_df = self.metadata_df[self.metadata_df['Cluster'] == cluster_id]
            ceo_counts = cluster_df['CEO'].value_counts()
            top_ceos = ceo_counts.head(5).index.tolist()
            
            cluster_stats[cluster_id] = {
                'Size': len(cluster_df),
                'Top_CEOs': top_ceos,
                'CEO_Percentages': {ceo: count/len(cluster_df) for ceo, count in ceo_counts.items()}
            }
        
        return cluster_stats
    
    def plot_clusters(self, method='tsne', save_path=None):
        """Plot the clusters in the reduced embedding space."""
        # Ensure dimensionality reduction has been run
        if not hasattr(self, 'reduced_embeddings'):
            print(f"Running {method} dimensionality reduction...")
            self.run_dimensionality_reduction(method=method)
        
        # Ensure clustering has been run
        if 'Cluster' not in self.metadata_df.columns:
            print("Running clustering...")
            self.generate_clusters()
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Get cluster IDs
        clusters = self.metadata_df['Cluster'].unique()
        
        # Plot each cluster with a different color
        for cluster_id in clusters:
            cluster_df = self.metadata_df[self.metadata_df['Cluster'] == cluster_id]
            
            x_col = f"{method.upper()}_1"
            y_col = f"{method.upper()}_2"
            
            plt.scatter(
                cluster_df[x_col], 
                cluster_df[y_col], 
                label=f"Cluster {cluster_id}",
                alpha=0.7,
                s=50
            )
        
        plt.title(f"Clustered CEO Speech Embeddings ({len(clusters)} clusters)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
        plt.show()
        
        # Print cluster statistics
        cluster_stats = self.generate_clusters(n_clusters=len(clusters))
        for cluster_id, stats in cluster_stats.items():
            print(f"\nCluster {cluster_id} ({stats['Size']} samples)")
            print("Top CEOs:")
            for ceo in stats['Top_CEOs']:
                percentage = stats['CEO_Percentages'][ceo] * 100
                print(f"  - {ceo}: {percentage:.1f}%")
    
    def correlate_with_personality(self, personality_file, output_dir=None):
        """Correlate embeddings with personality traits."""
        # Load personality data
        personality_df = pd.read_csv(personality_file)
        
        # Standardize CEO names for matching
        personality_df['CEO_std'] = personality_df['Name'].str.lower().str.strip()
        self.metadata_df['CEO_std'] = self.metadata_df['CEO'].str.lower().str.strip()
        
        # Get average embedding for each CEO
        ceo_avg_embeddings = {}
        for ceo in self.metadata_df['CEO'].unique():
            ceo_ids = [row['ID'] for row in self.metadata if row['CEO'] == ceo]
            if ceo_ids:
                ceo_embeddings = [self.embeddings[id] for id in ceo_ids]
                ceo_avg_embeddings[ceo.lower().strip()] = np.mean(ceo_embeddings, axis=0)
        
        # Find traits in personality data
        trait_cols = [col for col in personality_df.columns if col in 
                     ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']]
        
        if not trait_cols:
            print("No personality traits found in the personality data")
            return None
        
        # Calculate correlations between embedding dimensions and traits
        correlations = defaultdict(list)
        
        for trait in trait_cols:
            # Get CEOs with both embedding and personality data
            common_ceos = set(ceo_avg_embeddings.keys()) & set(personality_df['CEO_std'].values)
            
            if not common_ceos:
                print(f"No common CEOs found for trait: {trait}")
                continue
            
            # Extract trait values and embeddings for common CEOs
            trait_values = []
            embedding_matrix = []
            
            for ceo in common_ceos:
                # Get trait value
                trait_row = personality_df[personality_df['CEO_std'] == ceo]
                if len(trait_row) == 0:
                    continue
                trait_value = trait_row[trait].values[0]
                
                # Get embedding
                embedding = ceo_avg_embeddings[ceo]
                
                trait_values.append(trait_value)
                embedding_matrix.append(embedding)
            
            # Convert to numpy arrays
            trait_values = np.array(trait_values)
            embedding_matrix = np.array(embedding_matrix)
            
            # Calculate correlation for each embedding dimension
            for dim in range(embedding_matrix.shape[1]):
                dim_values = embedding_matrix[:, dim]
                corr = np.corrcoef(trait_values, dim_values)[0, 1]
                correlations[trait].append((dim, corr))
        
        # Find top correlated dimensions for each trait
        top_correlations = {}
        for trait, corrs in correlations.items():
            # Sort by absolute correlation
            sorted_corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
            top_correlations[trait] = sorted_corrs[:10]  # Top 10 dimensions
        
        # Generate correlation plots
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(15, 10))
            for i, trait in enumerate(trait_cols):
                plt.subplot(len(trait_cols), 1, i+1)
                
                # Get top 10 correlations
                top_corrs = top_correlations[trait]
                dims, corrs = zip(*top_corrs)
                
                plt.bar(dims, corrs)
                plt.title(f"Top Correlated Embedding Dimensions for {trait}")
                plt.ylabel("Correlation")
                plt.xlabel("Embedding Dimension")
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'embedding_trait_correlations.png'), dpi=300)
            plt.close()
        
        return top_correlations

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Generate and visualize embeddings from CEO speeches')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing speech files')
    parser.add_argument('--output_dir', type=str, default='results/embedding_analysis', help='Directory for output files')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model to use')
    parser.add_argument('--limit', type=int, help='Limit the number of files to process')
    parser.add_argument('--reduction', type=str, default='tsne', choices=['tsne', 'pca'], help='Dimensionality reduction method')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters to generate')
    parser.add_argument('--personality_file', type=str, help='CSV file with personality data for correlation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the visualizer
    visualizer = EmbeddingVisualizer(embedding_model=args.model)
    
    # Process the directory
    metadata = visualizer.process_directory(args.data_dir, args.limit)
    
    # Save metadata to CSV
    metadata.to_csv(os.path.join(args.output_dir, 'embedding_metadata.csv'), index=False)
    
    # Run dimensionality reduction
    visualizer.run_dimensionality_reduction(method=args.reduction)
    
    # Generate and save plots
    visualizer.plot_embeddings(
        color_by='CEO', 
        method=args.reduction,
        save_path=os.path.join(args.output_dir, f'ceo_embeddings_{args.reduction}.png')
    )
    
    # Generate clusters
    cluster_stats = visualizer.generate_clusters(n_clusters=args.clusters)
    
    # Save cluster statistics
    with open(os.path.join(args.output_dir, 'cluster_stats.txt'), 'w') as f:
        for cluster_id, stats in cluster_stats.items():
            f.write(f"Cluster {cluster_id} ({stats['Size']} samples)\n")
            f.write("Top CEOs:\n")
            for ceo in stats['Top_CEOs']:
                percentage = stats['CEO_Percentages'][ceo] * 100
                f.write(f"  - {ceo}: {percentage:.1f}%\n")
            f.write("\n")
    
    # Plot clusters
    visualizer.plot_clusters(
        method=args.reduction,
        save_path=os.path.join(args.output_dir, f'clustered_embeddings_{args.reduction}.png')
    )
    
    # Correlate with personality if file provided
    if args.personality_file:
        correlations = visualizer.correlate_with_personality(args.personality_file, args.output_dir)
        
        # Save correlations
        if correlations:
            with open(os.path.join(args.output_dir, 'trait_embedding_correlations.txt'), 'w') as f:
                for trait, corrs in correlations.items():
                    f.write(f"{trait} correlations with embedding dimensions:\n")
                    for dim, corr in corrs:
                        f.write(f"  Dimension {dim}: {corr:.4f}\n")
                    f.write("\n")
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()