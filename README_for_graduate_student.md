# CEO Personality Analysis: BERT vs. LIWC Comparison

## Overview

This project investigates the discrepancy between BERT-based personality trait detection and LIWC (Linguistic Inquiry and Word Count) approaches when analyzing CEO speeches. It specifically addresses the negative correlation between neuroticism scores from these two methods that you observed in your research.

## Key Findings

1. **Methodological differences explain correlations**:
   - The negative correlation between BERT and LIWC neuroticism scores is expected due to fundamentally different approaches (contextual embeddings vs. word counting)
   - BERT analyzes text contextually while LIWC counts explicit word occurrences
   - Business language requires domain-specific adjustments to capture traits accurately

2. **Enhanced LIWC dictionary**:
   - We created an expanded LIWC-like dictionary with more business-relevant terms
   - Added categories like "business risk terms" and "hedging language" for neuroticism
   - While this helped with some traits, negative correlations persisted

3. **Available personality models**:
   - Several pre-trained models for personality assessment are available
   - `Minej/bert-base-personality` is specifically designed for Big Five personality detection
   - Other options include `KevSun/Personality_LM` and `gmenchetti/setfit-personality-mpnet`

4. **Recommendations**:
   - Use specialized personality models rather than general language models
   - Consider fine-tuning a model specifically on CEO/business language
   - Validate findings against external criteria (e.g., established personality assessments)
   - Consider a hybrid approach combining strengths of both methods

## Code and Analysis

This repository contains:

1. **Python script** (`src/ceo_personality_comparison.py`):
   - Processes CEO speech transcripts
   - Implements both BERT and LIWC-like analyses
   - Compares results and generates visualizations
   - Includes simple and enhanced LIWC dictionaries

2. **Quarto report** (`results/ceo_bert_liwc_comparison.qmd`):
   - Detailed analysis of the two approaches
   - Visualization of correlations
   - Discussion of findings and recommendations

3. **Result files**:
   - Processed data in CSV format
   - Correlation plots
   - Comparison visualizations

## How to Use This Code

### Requirements

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

You'll also need Quarto for rendering the report:
```bash
# Mac
brew install quarto

# Windows
# Download from https://quarto.org/docs/get-started/
```

### Easy Analysis with the Wrapper Script

For convenience, I've created a wrapper script that makes it easy to run the personality analysis with different options. Here's how to use it:

1. Make the script executable (one-time setup):
   ```
   chmod +x run_analysis.sh
   ```

2. Run the script with your desired options:
   ```
   ./run_analysis.sh [OPTIONS]
   ```

### Available Command-Line Arguments

- `--help`: Show help message with all available options
- `--demo`: Run in demo mode without downloading large models (good for testing)
- `--full`: Run analysis on all available files (by default it processes 30 files)
- `--limit N`: Process only N files (e.g., `--limit 50` for 50 files)
- `--simple-liwc`: Use the simple LIWC dictionary
- `--enhanced-liwc`: Use the enhanced LIWC dictionary (default)
- `--personality`: Use the dedicated personality model (Minej/bert-base-personality)
- `--distilbert`: Use DistilBERT model (faster but less accurate)
- `--weighted`: Use confidence-weighted analysis for improved accuracy (NEW!)
- `--list-models`: List available personality models
- `--render`: Render the Quarto document to create an HTML report

### NEW: Confidence-Weighted Analysis 

The script now includes a more sophisticated analysis method that gives more weight to text chunks where the model is more confident in its predictions. This typically produces more accurate personality assessments by:

1. Breaking the speech into manageable chunks
2. Analyzing each chunk separately
3. Calculating a confidence score for each prediction
4. Weighting the results based on confidence (more confident predictions have more influence)

To use this enhanced method, add the `--weighted` flag to your command:
```
./run_analysis.sh --weighted --personality
```

For best results, combine it with the dedicated personality model.

### Examples

Quick demo:
```
./run_analysis.sh --demo --render
```

Analyze 50 files with personality model:
```
./run_analysis.sh --limit 50 --personality
```

Full analysis with simple LIWC:
```
./run_analysis.sh --full --simple-liwc
```

Run with enhanced confidence weighting:
```
./run_analysis.sh --weighted --personality --render
```

### Running the Analysis Manually

If you prefer, you can also run the Python script directly:

```bash
python src/ceo_personality_comparison.py --limit 30 --demo
```

Options:
- `--limit N`: Process only N files (for faster testing)
- `--demo`: Use random values instead of downloading and running the actual BERT model
- `--simple-liwc`: Use the simplified LIWC dictionary instead of the enhanced version
- `--model MODEL_NAME`: Specify which HuggingFace model to use
- `--render`: Automatically render the Quarto document after analysis
- `--list-personality-models`: Show available personality-related models on HuggingFace

For the full analysis with a specific BERT model:

```bash
python src/ceo_personality_comparison.py --model Minej/bert-base-personality --render
```

### Viewing Results

After running the analysis:

1. Open `results/ceo_bert_liwc_comparison.html` to view the rendered report
2. Check CSV files in the `results` folder for raw data:
   - `bert_personality_analysis.csv`
   - `liwc_personality_analysis_enhanced_LIWC.csv` or `liwc_personality_analysis_simple_LIWC.csv`
   - `trait_correlations_enhanced_LIWC.csv` or `trait_correlations_simple_LIWC.csv`

## Next Steps for Your Research

1. **Model validation**:
   - Check if the pre-trained BERT model was trained on appropriate data
   - Consider creating a labeled dataset of CEO language with known personality traits

2. **Domain adaptation**:
   - Further expand the LIWC dictionaries for business language
   - Consider fine-tuning a BERT model on business communications

3. **Multi-method approach**:
   - Incorporate human coding of a sample of speeches
   - Compare with external measures (e.g., media personality assessments of CEOs)
   - Consider other NLP approaches beyond BERT and LIWC

4. **Business outcomes**:
   - Correlate personality traits with business performance metrics
   - Investigate how different traits manifest in business leadership contexts

## Contact

If you have any questions or need further assistance with this analysis, please contact Bryan Acton.

Good luck with your research! 