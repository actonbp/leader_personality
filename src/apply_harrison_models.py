#!/usr/bin/env python3
"""
Apply pre-trained personality models (from Harrison et al.) to text data using
various embedding techniques (OpenAI or Sentence Transformers).

Outputs a single CSV with all CEOs and all traits found in the models directory.
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file, if it exists
# This will not override existing environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Embedding Functions ---

# Placeholder for OpenAI pricing per 1K tokens
OPENAI_PRICING = {
    "text-embedding-ada-002": 0.0001,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
}

def get_openai_embedding(text: str, model_name: str = "text-embedding-ada-002", api_key: str = None) -> Tuple[Optional[np.ndarray], int]:
    """Generates text embeddings using OpenAI API and returns embedding and token count."""
    tokens = 0
    try:
        from openai import OpenAI
        import tiktoken
    except ImportError as e:
        logging.error(f"{e}. Please install required libraries: pip install openai tiktoken")
        return None, tokens

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OPENAI_API_KEY environment variable not set.")
            return None, tokens

    client = OpenAI(api_key=api_key)

    # Calculate token count
    try:
        # Note: Encoding name might differ for older models if needed, but ada-002 uses cl100k_base
        encoding = tiktoken.encoding_for_model(model_name) 
        tokens = len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Could not count tokens for model {model_name}: {e}. Cost estimation may be inaccurate.")

    try:
        if model_name == "text-embedding-3-large":
            response = client.embeddings.create(input=text, model=model_name, dimensions=1536)
        else:
            response = client.embeddings.create(input=text, model=model_name)

        embedding = response.data[0].embedding
        if len(embedding) != 1536:
            logging.warning(f"Warning: Embedding dimension for {model_name} is {len(embedding)}, expected 1536.")
        return np.array(embedding), tokens
    except Exception as e:
        logging.error(f"Error getting OpenAI embedding for model {model_name}: {e}")
        return None, tokens

def get_sentence_transformer_embedding(text: str, model_name: str) -> Tuple[Optional[np.ndarray], int]:
    """Generates text embeddings using a Sentence Transformer model. Returns embedding and 0 tokens."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logging.error("Sentence Transformers library not found. Please install with: pip install sentence-transformers")
        return None, 0

    try:
        # Consider caching the model loading if running many times
        model = SentenceTransformer(model_name) 
        embedding = model.encode(text)
        if len(embedding) != 1536:
             logging.warning(f"Warning: Embedding dimension for {model_name} is {len(embedding)}, expected 1536.")
        return np.array(embedding), 0 # Return 0 tokens for open source models
    except Exception as e:
        logging.error(f"Error getting Sentence Transformer embedding for model {model_name}: {e}")
        return None, 0

# --- Main Analysis Function ---

def analyze_text_with_harrison_model(text: str, pipeline_model, embedding_model_name: str, openai_api_key: str = None) -> Tuple[float, int]:
    """
    Analyzes a text using a loaded Harrison et al. pipeline model object and a specified embedding technique.

    Args:
        text (str): The input text to analyze.
        pipeline_model: The loaded pycaret Pipeline object.
        embedding_model_name (str): Name of the embedding model to use.
        openai_api_key (str, optional): OpenAI API key.

    Returns:
        Tuple[float, int]: The predicted score and the number of tokens used (0 if not applicable).
    """
    tokens_used = 0

    # 1. Generate the embedding based on the chosen model
    embedding = None
    if embedding_model_name.startswith("text-embedding-"):
        embedding, tokens_used = get_openai_embedding(text, embedding_model_name, api_key=openai_api_key)
    elif "/" in embedding_model_name:
        embedding, tokens_used = get_sentence_transformer_embedding(text, embedding_model_name)
    else:
        logging.error(f"Unsupported embedding model name format: {embedding_model_name}")
        return np.nan, 0

    if embedding is None:
        logging.warning(f"Failed to generate embedding for text using {embedding_model_name}. Skipping.")
        return np.nan, tokens_used

    # 2. Apply the pipeline model (scaler + estimator)
    try:
        # Create a DataFrame matching the expected input format for pycaret pipelines
        # It expects a DataFrame, even if only using the embedding
        # We assume the embedding corresponds to features expected by the scaler in the pipeline
        num_features = len(embedding)
        # Create generic column names F0, F1, ... FN
        feature_names = list(range(num_features))
        input_df = pd.DataFrame([embedding], columns=feature_names)

        # Use the pipeline to predict (handles scaling and estimation internally)
        # PyCaret's predict_model returns a DataFrame, the prediction is often in a 'prediction_label' column
        prediction_df = pipeline_model.predict(input_df)
        
        # Extract the prediction value
        if isinstance(prediction_df, pd.DataFrame):
            # Original logic if it returns a DataFrame
            if 'prediction_label' in prediction_df.columns:
                 prediction_value = prediction_df['prediction_label'].iloc[0]
            elif 'Label' in prediction_df.columns:
                 prediction_value = prediction_df['Label'].iloc[0]
            elif 'prediction_score' in prediction_df.columns:
                 prediction_value = prediction_df['prediction_score'].iloc[0]
            else:
                logging.warning(f"Could not find standard prediction column names in DataFrame. Using first column: {prediction_df.columns[0]}")
                prediction_value = prediction_df.iloc[0, 0]
        elif isinstance(prediction_df, np.ndarray) and prediction_df.ndim >= 1:
            # If it returns a NumPy array, assume the first element is the prediction
            prediction_value = prediction_df[0]
            if prediction_df.size > 1:
                logging.warning(f"Prediction result is a NumPy array with multiple values. Using the first one: {prediction_value}")
        else:
            # Handle unexpected return type
            logging.error(f"Unexpected prediction result type: {type(prediction_df)}. Cannot extract score.")
            return np.nan, tokens_used

        return float(prediction_value), tokens_used
        
    except ValueError as e:
        logging.error(f"ValueError applying pipeline model to embedding (often shape mismatch): {e}")
        logging.error(f"Input DataFrame shape: {input_df.shape if 'input_df' in locals() else 'N/A'}")
        return np.nan, tokens_used
    except Exception as e:
        logging.error(f"Error applying pipeline model to embedding: {e}")
        logging.error(f"Full traceback:\\n{traceback.format_exc()}")
        logging.error(f"Embedding shape used to create input DF: {embedding.shape}")
        return np.nan, tokens_used

# --- Main Execution Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze text data using Harrison et al. personality models. Outputs a single CSV with all traits."
    )
    parser.add_argument("--data-dir", type=str, required=True, 
                        help="Directory containing the text files (.txt) to analyze.")
    parser.add_argument("--models-dir", type=str, default="models", 
                        help="Directory containing the pickled model files (.pkl) for traits.")
    parser.add_argument("--embedding-model", type=str, default="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja", 
                        help="Name of the embedding model (default: 'sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja'). See script header for compatible options.")
    parser.add_argument("--output-file", type=str, required=True, 
                        help="Path to save the resulting aggregated CSV file.")
    parser.add_argument("--openai-key", type=str, default=None, 
                        help="OpenAI API key (optional, defaults to OPENAI_API_KEY env var).")

    args = parser.parse_args()

    # --- Validation ---
    models_path = Path(args.models_dir)
    data_path = Path(args.data_dir)
    output_path = Path(args.output_file)

    if not models_path.is_dir():
        logging.error(f"Models directory not found: {models_path}")
        exit(1)
    if not data_path.is_dir():
        logging.error(f"Data directory not found: {data_path}")
        exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load Models ---
    loaded_models = {}
    pkl_files = list(models_path.glob("*.pkl"))
    if not pkl_files:
        logging.error(f"No .pkl model files found in {models_path}")
        exit(1)

    for pkl_file in pkl_files:
        try:
            trait_name = pkl_file.stem.split('_')[1] # e.g., 'agree' from 'openai_agree_model.pkl'
            # Load the PyCaret pipeline object directly
            model_obj = joblib.load(pkl_file) 
            # Basic check if it looks like a pipeline (more robust checks might be needed)
            if hasattr(model_obj, 'predict') and hasattr(model_obj, 'steps'): 
                # Attempt to disable joblib memory caching to avoid permission errors in /var/folders
                if hasattr(model_obj, 'memory') and model_obj.memory is not None:
                    logging.info(f"Disabling memory caching for model: {pkl_file.name}")
                    model_obj.memory = None # Disable caching
                    
                loaded_models[trait_name] = model_obj
                logging.info(f"Successfully loaded pipeline model for trait '{trait_name}' from {pkl_file.name}")
            else:
                logging.warning(f"Skipping {pkl_file.name}: loaded object doesn't look like a scikit-learn/pycaret pipeline.")
        except AttributeError as ae:
             # Handle specific AttributeErrors during loading if versions are mismatched
             logging.error(f"AttributeError loading model from {pkl_file.name}: {ae}. This often indicates a version mismatch.")
             logging.error(f"Ensure scikit-learn version is compatible (e.g., 1.1.1 based on original requirements).")
        except Exception as e:
            logging.error(f"Error loading model from {pkl_file.name}: {e}")
            
    if not loaded_models:
        logging.error("No valid models could be loaded. Exiting.")
        exit(1)

    # --- Process Files ---
    logging.info(f"Starting analysis using embedding model: {args.embedding_model}")
    logging.info(f"Processing data from: {data_path}")
    
    all_results = []
    total_tokens = 0
    files_to_process = list(data_path.glob("*.txt"))
    logging.info(f"Found {len(files_to_process)} files to process.")

    for file_path in files_to_process:
        logging.debug(f"Processing {file_path.name}...")
        ceo_result = {'Name': None, 'File': file_path.name}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract CEO name
            ceo_name = file_path.stem.split('-')[0].strip()
            # Handle cases where name might not have digits or needs cleaning
            if ceo_name:
                 ceo_name = ''.join([c for c in ceo_name if not c.isdigit()]).strip()
            else:
                 ceo_name = file_path.stem # Fallback if parsing fails
            ceo_result['Name'] = ceo_name

            if not content.strip():
                logging.warning(f"File {file_path.name} is empty. Skipping content analysis.")
                for trait_name in loaded_models.keys():
                    ceo_result[trait_name] = np.nan
            else:
                file_tokens = 0
                # Analyze with each loaded trait model
                for trait_name, pipeline_obj in loaded_models.items():
                    score, tokens = analyze_text_with_harrison_model(
                        content,
                        pipeline_obj, # Pass the loaded pipeline object
                        args.embedding_model,
                        args.openai_key
                    )
                    ceo_result[trait_name] = score
                    # Accumulate tokens only once per file
                    if file_tokens == 0 and tokens > 0:
                        file_tokens = tokens 
                total_tokens += file_tokens 
                
            all_results.append(ceo_result)

        except Exception as e:
            logging.error(f"Critical error processing file {file_path.name}: {e}")
            # Add partial result if possible
            if ceo_result['Name'] is None:
                 ceo_result['Name'] = file_path.stem 
            for trait_name in loaded_models.keys():
                if trait_name not in ceo_result:
                    ceo_result[trait_name] = np.nan
            all_results.append(ceo_result)

    # --- Save Results & Report Cost ---
    final_df = pd.DataFrame(all_results)
    # Reorder columns: Name, File, then traits alphabetically
    trait_cols = sorted([col for col in final_df.columns if col not in ['Name', 'File']])
    final_df = final_df[['Name', 'File'] + trait_cols]
    
    final_df.to_csv(output_path, index=False)
    logging.info(f"Analysis complete. Aggregated results saved to {output_path}")

    # Report OpenAI cost if applicable
    if args.embedding_model.startswith("text-embedding-"):
        model_price = OPENAI_PRICING.get(args.embedding_model, 0)
        if model_price > 0 and total_tokens > 0:
            estimated_cost = (total_tokens / 1000) * model_price
            logging.info(f"OpenAI API Usage: {total_tokens} tokens processed.")
            logging.info(f"Estimated Cost for {args.embedding_model}: ${estimated_cost:.6f}")
        elif total_tokens > 0:
            logging.warning(f"Could not estimate cost for {args.embedding_model}. Price unknown.")
        else:
            logging.info("No tokens processed using OpenAI.") 