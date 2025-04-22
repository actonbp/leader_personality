"""
CEO Speech Embedding Analysis

This script processes CEO speeches using the Hugging Face embedding model,
applies UMAP dimensionality reduction, and visualizes the results to explore
natural language patterns among CEOs.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap
import re
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Import HuggingFace transformers
from transformers import AutoTokenizer, AutoModel
import torch

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
DATA_DIR = "data/282 ceo data  2/"
OUTPUT_DIR = "results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f"{OUTPUT_DIR}/embedding_analysis.log"),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_speeches():
    """
    Load all CEO speech files from the data directory.
    
    Returns:
        dict: A dictionary mapping CEO names to their speech text
    """
    speeches = {}
    ceo_metadata = {}
    
    logger.info(f"Loading speeches from {DATA_DIR}")
    
    # Get list of files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    
    for filename in tqdm(files, desc="Loading speeches"):
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract CEO name and company from filename (format: "1Jane Doe - COMP.txt" or "John Doe - COMP.txt")
            match = re.match(r'(?:1)?([^-]+) - ([^.]+)\.txt', filename)
            if match:
                ceo_name = match.group(1).strip()
                company = match.group(2).strip()
                
                # Determine gender (in a real implementation, you would use a proper database)
                # For this example, we'll use a simplified approach with the "1" prefix indicating female CEOs
                gender = "Female" if filename.startswith("1") else "Male"
                
                speeches[ceo_name] = content
                ceo_metadata[ceo_name] = {
                    "company": company,
                    "gender": gender,
                    "filename": filename
                }
            else:
                logger.warning(f"Couldn't parse CEO name from filename: {filename}")
                
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
    
    logger.info(f"Loaded {len(speeches)} speeches")
    return speeches, ceo_metadata

def preprocess_speech(text):
    """
    Clean and preprocess speech text.
    
    Args:
        text (str): Raw speech text
        
    Returns:
        str: Preprocessed text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common headers/footers
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;?!-]', '', text)
    
    return text.strip()

def get_embeddings(speeches, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generate embeddings for CEO speeches using the specified Hugging Face model.
    
    Args:
        speeches (dict): Dictionary of CEO speeches
        model_name (str): Name of the HuggingFace model to use
        
    Returns:
        dict: Dictionary mapping CEO names to their speech embeddings
    """
    logger.info(f"Generating embeddings using {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Set to evaluation mode
    model.eval()
    
    # Function to get embeddings for a single text
    def get_embedding(text):
        # Tokenize and truncate if needed
        inputs = tokenizer(text, padding=True, truncation=True, 
                          max_length=512, return_tensors="pt")
        
        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use mean pooling to get sentence embedding
        # Take the mean of the last hidden states across the token dimension
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    
    # Process each speech
    embeddings = {}
    for ceo_name, speech in tqdm(speeches.items(), desc="Generating embeddings"):
        # Preprocess speech
        processed_speech = preprocess_speech(speech)
        
        # Get embedding
        try:
            embedding = get_embedding(processed_speech)
            embeddings[ceo_name] = embedding
        except Exception as e:
            logger.error(f"Error generating embedding for {ceo_name}: {str(e)}")
    
    logger.info(f"Generated embeddings for {len(embeddings)} CEOs")
    return embeddings

def reduce_dimensions(embeddings, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Reduce the dimensionality of the embeddings using UMAP.
    
    Args:
        embeddings (dict): Dictionary of CEO embeddings
        n_components (int): Number of dimensions to reduce to
        n_neighbors (int): UMAP parameter for local neighborhood size
        min_dist (float): UMAP parameter for minimum distance between points
        
    Returns:
        tuple: (reduced_embeddings, ceo_names)
    """
    logger.info(f"Reducing dimensions with UMAP (n_components={n_components})")
    
    # Prepare data for UMAP
    ceo_names = list(embeddings.keys())
    embedding_array = np.array([embeddings[name] for name in ceo_names])
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embedding_array)
    
    # Initialize and fit UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings_scaled)
    
    logger.info(f"Reduced dimensions from {embedding_array.shape[1]} to {n_components}")
    return reduced_embeddings, ceo_names

def visualize_embeddings(reduced_embeddings, ceo_names, metadata, output_path):
    """
    Create a visualization of the reduced embeddings.
    
    Args:
        reduced_embeddings (np.array): The reduced embeddings
        ceo_names (list): List of CEO names
        metadata (dict): Dictionary of CEO metadata
        output_path (str): Path to save the visualization
    """
    logger.info("Creating embedding visualization")
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'CEO': ceo_names,
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'Gender': [metadata[name]['gender'] for name in ceo_names],
        'Company': [metadata[name]['company'] for name in ceo_names]
    })
    
    # Set up the plot
    plt.figure(figsize=(14, 12))
    
    # Create the scatter plot
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='Gender',
        style='Gender',
        s=150,
        alpha=0.8
    )
    
    # Add CEO name labels
    for i, row in df.iterrows():
        plt.text(
            row['x'] + 0.05, 
            row['y'] + 0.02, 
            row['CEO'] + f" ({row['Company']})",
            fontsize=9,
            alpha=0.8
        )
    
    # Add grid
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add title and labels
    plt.title('CEO Speech Patterns in Embedding Space', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    
    # Improve legend
    plt.legend(title='Gender', fontsize=10, title_fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Visualization saved to {output_path}")
    
    # Also save the data for further analysis
    df.to_csv(os.path.join(OUTPUT_DIR, 'ceo_embedding_coordinates.csv'), index=False)

def analyze_clusters(reduced_embeddings, ceo_names, metadata):
    """
    Analyze the clusters formed in the embedding space.
    
    Args:
        reduced_embeddings (np.array): The reduced embeddings
        ceo_names (list): List of CEO names
        metadata (dict): Dictionary of CEO metadata
        
    Returns:
        dict: Analysis results
    """
    logger.info("Analyzing embedding patterns")
    
    # Create DataFrame with embedding coordinates
    df = pd.DataFrame({
        'CEO': ceo_names,
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'Gender': [metadata[name]['gender'] for name in ceo_names],
        'Company': [metadata[name]['company'] for name in ceo_names]
    })
    
    # Find nearest neighbors for each CEO
    def find_nearest_neighbors(df, ceo_name, n=5):
        ceo_coords = df[df['CEO'] == ceo_name][['x', 'y']].values[0]
        
        # Calculate distances
        df['distance'] = df.apply(
            lambda row: np.sqrt((row['x'] - ceo_coords[0])**2 + (row['y'] - ceo_coords[1])**2), 
            axis=1
        )
        
        # Get nearest neighbors (excluding the CEO themselves)
        neighbors = df[df['CEO'] != ceo_name].sort_values('distance').head(n)
        return neighbors[['CEO', 'Company', 'distance']]
    
    # Calculate nearest neighbors for each CEO
    nearest_neighbors = {}
    for ceo in ceo_names:
        nearest_neighbors[ceo] = find_nearest_neighbors(df, ceo)
    
    # Find outliers (CEOs far from others)
    avg_distance = df.apply(
        lambda row: df[df['CEO'] != row['CEO']]['distance'].mean(),
        axis=1
    )
    
    df['avg_distance'] = avg_distance
    outliers = df.sort_values('avg_distance', ascending=False).head(5)
    
    # Analyze gender distribution
    gender_distribution = df.groupby('Gender').agg({
        'x': ['mean', 'std'],
        'y': ['mean', 'std'],
        'CEO': 'count'
    })
    
    # Save analysis results
    analysis_results = {
        "nearest_neighbors": nearest_neighbors,
        "outliers": outliers[['CEO', 'Company', 'avg_distance']].to_dict('records'),
        "gender_distribution": gender_distribution.to_dict()
    }
    
    # Save as JSON
    import json
    with open(os.path.join(OUTPUT_DIR, 'embedding_analysis_results.json'), 'w') as f:
        # Convert any NumPy values to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Ensure nested dictionaries are also converted
        def clean_for_json(d):
            if isinstance(d, dict):
                return {k: clean_for_json(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_for_json(i) for i in d]
            else:
                return convert_numpy(d)
        
        json.dump(clean_for_json(analysis_results), f, indent=2)
    
    logger.info("Analysis complete and results saved")
    return analysis_results

def main():
    """Main execution function"""
    logger.info("Starting CEO embedding analysis")
    
    # Load speeches
    speeches, metadata = load_speeches()
    
    # Generate embeddings
    embeddings = get_embeddings(speeches)
    
    # Reduce dimensions
    reduced_embeddings, ceo_names = reduce_dimensions(embeddings)
    
    # Visualize embeddings
    output_path = os.path.join(OUTPUT_DIR, 'ceo_embedding_visualization.png')
    visualize_embeddings(reduced_embeddings, ceo_names, metadata, output_path)
    
    # Analyze clusters
    analysis_results = analyze_clusters(reduced_embeddings, ceo_names, metadata)
    
    logger.info("CEO embedding analysis complete")
    
    # Return results for inspection
    return {
        'embeddings': embeddings,
        'reduced_embeddings': reduced_embeddings,
        'ceo_names': ceo_names,
        'metadata': metadata,
        'analysis_results': analysis_results
    }

if __name__ == "__main__":
    main()