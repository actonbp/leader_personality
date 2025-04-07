# CEO Leadership & Personality Analysis

A collaborative research project by Bryan Acton and Nan Liang at Binghamton University, investigating personality traits in CEO communications using NLP and embedding models.

## Project Overview

This project analyzes the linguistic patterns and personality traits of Fortune 500 CEOs through their public communications. We use both traditional linguistic analysis (LIWC) and modern embedding-based approaches to understand leadership communication patterns.

### Key Research Questions

1. How do different NLP methods (LIWC vs. embedding models) capture personality traits in formal business communications?
2. What linguistic patterns are common across CEO speeches?
3. How do these patterns correlate with the Big Five personality traits?
4. Can we develop better methods for personality assessment from text?

## Data Sources

- **282 CEO Speeches**: Transcripts from S&P 500 CEOs, primarily from earnings calls
- **844 Earnings Calls**: Expanded dataset of quarterly earnings call transcripts

## Technical Approach

The project employs multiple analytical methods:

1. **BERT-based Personality Analysis**: Using pre-trained personality detection models
2. **LIWC Analysis**: Traditional word-counting approach with psychology-informed dictionaries
3. **Linguistic Feature Analysis**: Statistical analysis of vocabulary, sentence structure, etc.
4. **Embedding-based Analysis**: Exploring vector representations of CEO language

## Core Components

- **Data Preprocessing**: Extracting and cleaning CEO speech segments
- **Descriptive Analysis**: Statistical measures of CEO communication patterns
- **Personality Assessment**: Applying multiple methods for trait detection
- **Comparison Framework**: Evaluating concordance between different approaches
- **Visualization**: Making complex relationships interpretable

## Getting Started

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Basic Speech Analysis**:
   ```bash
   python src/speech_descriptives.py --data_dir "data/282 ceo data  2" --output_dir "results/speech_analysis"
   ```

3. **Personality Analysis**:
   ```bash
   ./run_analysis.sh --personality --weighted
   ```

4. **Generate Reports**:
   ```bash
   quarto render results/speech_analysis/speech_analysis.qmd
   ```

## Project Structure

```
leader_personality/
├── data/                      # Raw and preprocessed data
│   ├── 282 ceo data  2/       # Primary CEO speech dataset 
│   ├── 844earnings call/      # Extended earnings call dataset
│   ├── preprocessed_ceo_speeches/  # Cleaned speech files
│   └── speeches/              # Additional curated speeches
├── src/                       # Source code
│   ├── speech_descriptives.py # NEW: Linguistic analysis
│   ├── personality_analyzer.py # BERT-based personality detection
│   ├── enhanced_personality_analyzer.py # Weighted confidence model
│   ├── preprocess_ceo_transcripts.py # Text cleaning pipeline
│   └── visualization.py       # Visualization utilities
├── results/                   # Analysis outputs
│   ├── speech_analysis/       # NEW: Linguistic feature analysis
│   ├── ceo_analysis.qmd       # Quarto report for personality analysis
│   └── personality_analysis.csv # Raw personality scores
└── docs/                      # Documentation
```

## Next Steps

1. **Descriptive Analysis**: Generate comprehensive linguistic statistics on the full dataset
2. **Embedding Exploration**: Visualize CEO speech embeddings to identify patterns
3. **Method Comparison**: Further investigate the divergence between LIWC and BERT approaches
4. **Domain Adaptation**: Explore fine-tuning language models on business communications

## Contributors

- **Bryan Acton** - Binghamton University
- **Nan Liang** - Binghamton University