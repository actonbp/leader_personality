#!/bin/bash

# Setup script to run CEO embedding analysis
echo "Setting up environment and running CEO embedding analysis..."

# Make sure required packages are installed
pip install -q transformers torch numpy pandas matplotlib seaborn scikit-learn umap-learn tqdm

# Navigate to the source directory
cd "$(dirname "$0")"

# Run the analysis script
echo "Running CEO embedding analysis..."
python3 src/ceo_embedding_analysis.py

echo "Analysis complete! Results saved to the 'results' directory."
echo "Main visualization: results/ceo_embedding_visualization.png"
echo "Embedding coordinates: results/ceo_embedding_coordinates.csv"
echo "Analysis results: results/embedding_analysis_results.json"