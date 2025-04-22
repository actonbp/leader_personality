#!/bin/bash

# CEO Speech Descriptive Analysis Script
# --------------------------------------
# A simplified script for running the descriptive analysis tools
# Authors: Bryan Acton and Nan Liang, Binghamton University

# Default parameters
DATA_SET="282"  # Use the 282 CEO dataset by default
LIMIT=50        # Process only 50 files by default for quick analysis
EMBEDDING=false # Don't run embedding analysis by default
PERSONALITY=""  # No personality file by default

# Help function
show_help() {
    echo "Usage: $(basename $0) [OPTIONS]"
    echo
    echo "CEO Speech Descriptive Analysis Script"
    echo
    echo "Options:"
    echo "  --help         Show this help message and exit"
    echo "  --full         Process all files in the dataset"
    echo "  --limit N      Process only N files (default: 50)"
    echo "  --282          Use the 282 CEO dataset (default)"
    echo "  --844          Use the 844 earnings call dataset"
    echo "  --embedding    Also run embedding analysis"
    echo "  --personality FILE  Correlate with personality data in specified CSV file"
    echo
    echo "Examples:"
    echo "  $(basename $0) --limit 10              # Quick analysis of 10 files"
    echo "  $(basename $0) --full --embedding      # Full analysis with embeddings"
    echo "  $(basename $0) --844 --limit 20        # Analyze 20 files from 844 dataset"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --full)
            LIMIT=0
            shift
            ;;
        --limit)
            LIMIT=$2
            shift 2
            ;;
        --282)
            DATA_SET="282"
            shift
            ;;
        --844)
            DATA_SET="844"
            shift
            ;;
        --embedding)
            EMBEDDING=true
            shift
            ;;
        --personality)
            PERSONALITY=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set data directory based on selected dataset
if [ "$DATA_SET" == "282" ]; then
    DATA_DIR="data/282 ceo data  2"
    echo "Using 282 CEO dataset"
else
    DATA_DIR="data/844earnings call"
    echo "Using 844 earnings call dataset"
fi

# Set output directories
SPEECH_OUTPUT="results/speech_analysis_${DATA_SET}"
EMBEDDING_OUTPUT="results/embedding_analysis_${DATA_SET}"

# Create output directories
mkdir -p "$SPEECH_OUTPUT"
mkdir -p "$EMBEDDING_OUTPUT"

# Display configuration
echo "CEO Speech Descriptive Analysis Configuration:"
echo "Dataset: $DATA_SET ($DATA_DIR)"
echo "Files to process: $([ $LIMIT -eq 0 ] && echo 'All' || echo $LIMIT)"
echo "Run embedding analysis: $EMBEDDING"
if [ -n "$PERSONALITY" ]; then
    echo "Personality file: $PERSONALITY"
fi
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required but not installed."
    exit 1
fi

# Check for required packages
REQUIRED_PACKAGES="pandas numpy matplotlib seaborn nltk sklearn sentence_transformers"
MISSING_PACKAGES=""

for package in $REQUIRED_PACKAGES; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $package"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "Missing packages:$MISSING_PACKAGES"
    echo "Installing required packages..."
    pip3 install $MISSING_PACKAGES
fi

# Construct command arguments for speech analysis
SPEECH_ARGS="--data_dir \"$DATA_DIR\" --output_dir \"$SPEECH_OUTPUT\""

if [ $LIMIT -gt 0 ]; then
    SPEECH_ARGS="$SPEECH_ARGS --limit $LIMIT"
fi

if [ -n "$PERSONALITY" ]; then
    SPEECH_ARGS="$SPEECH_ARGS --personality_file \"$PERSONALITY\""
fi

# Run the speech descriptive analysis
echo "Running speech descriptive analysis..."
python3 src/speech_descriptives.py --data_dir "$DATA_DIR" --output_dir "$SPEECH_OUTPUT" ${LIMIT:+--limit $LIMIT} ${PERSONALITY:+--personality_file "$PERSONALITY"}

# If embedding analysis is requested, run it
if [ "$EMBEDDING" = true ]; then
    # Construct command arguments for embedding analysis
    EMBEDDING_ARGS="--data_dir \"$DATA_DIR\" --output_dir \"$EMBEDDING_OUTPUT\""
    
    if [ $LIMIT -gt 0 ]; then
        EMBEDDING_ARGS="$EMBEDDING_ARGS --limit $LIMIT"
    fi
    
    if [ -n "$PERSONALITY" ]; then
        EMBEDDING_ARGS="$EMBEDDING_ARGS --personality_file \"$PERSONALITY\""
    fi
    
    echo "Running embedding analysis with command: python3 src/embedding_visualizer.py $EMBEDDING_ARGS"
    eval python3 src/embedding_visualizer.py $EMBEDDING_ARGS
fi

# Check if quarto is installed
if command -v quarto &> /dev/null; then
    echo "Rendering Quarto report..."
    quarto render "$SPEECH_OUTPUT/speech_analysis.qmd"
    echo "Analysis complete. Report is available at $SPEECH_OUTPUT/speech_analysis.html"
else
    echo "Analysis complete. Quarto is not installed, so the report was not rendered."
    echo "To generate the HTML report, install Quarto (https://quarto.org) and run:"
    echo "quarto render \"$SPEECH_OUTPUT/speech_analysis.qmd\""
fi