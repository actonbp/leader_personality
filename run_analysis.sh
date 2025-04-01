#!/bin/bash

# CEO Personality Analysis Wrapper Script
# ---------------------------------------

# Default parameters
LIMIT=30
MODEL="distilbert"
DEMO=false
LIWC="enhanced"
RENDER=false
WEIGHTED=false
DATA_DIR=""

# Help function
show_help() {
    echo "Usage: $(basename $0) [OPTIONS]"
    echo
    echo "CEO Personality Analysis Script"
    echo
    echo "Options:"
    echo "  --help            Show this help message and exit"
    echo "  --demo            Run in demo mode without downloading models"
    echo "  --full            Run full analysis on all files"
    echo "  --limit N         Process only N files (default: 30)"
    echo "  --simple-liwc     Use simple LIWC dictionary"
    echo "  --enhanced-liwc   Use enhanced LIWC dictionary (default)"
    echo "  --personality     Use dedicated personality model (Minej/bert-base-personality)"
    echo "  --distilbert      Use DistilBERT model (faster but less accurate)"
    echo "  --weighted        Use confidence-weighted analysis for improved accuracy"
    echo "  --list-models     List available personality models"
    echo "  --render          Render the Quarto document"
    echo "  --data-dir PATH   Use a specific data directory (default: data/282 ceo data  2)"
    echo
    echo "Examples:"
    echo "  $(basename $0) --demo --render         # Quick demo with rendering"
    echo "  $(basename $0) --limit 50 --personality # Run on 50 files using personality model"
    echo "  $(basename $0) --full --simple-liwc    # Full run with simple LIWC"
    echo "  $(basename $0) --weighted --personality # Use improved confidence-weighted analysis"
    echo "  $(basename $0) --full --data-dir \"data/282 ceo data  2\" # Analyze all 282 CEOs"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --demo)
            DEMO=true
            shift
            ;;
        --full)
            LIMIT=0
            shift
            ;;
        --limit)
            LIMIT=$2
            shift 2
            ;;
        --simple-liwc)
            LIWC="simple"
            shift
            ;;
        --enhanced-liwc)
            LIWC="enhanced"
            shift
            ;;
        --personality)
            MODEL="personality"
            shift
            ;;
        --distilbert)
            MODEL="distilbert"
            shift
            ;;
        --weighted)
            WEIGHTED=true
            shift
            ;;
        --data-dir)
            DATA_DIR=$2
            shift 2
            ;;
        --list-models)
            python3 src/ceo_personality_comparison.py --list-models
            exit 0
            ;;
        --render)
            RENDER=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Display configuration
echo "CEO Personality Analysis Configuration:"
echo "Files to process: $([ $LIMIT -eq 0 ] && echo 'All' || echo $LIMIT)"
echo "Model: $MODEL"
echo "LIWC dictionary: $LIWC"
echo "Demo mode: $DEMO"
echo "Weighted confidence: $WEIGHTED"
if [ -n "$DATA_DIR" ]; then
    echo "Data directory: $DATA_DIR"
else
    echo "Data directory: default (data/282 ceo data  2)"
fi
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required but not installed."
    exit 1
fi

# Check for required packages
REQUIRED_PACKAGES="transformers pandas numpy matplotlib seaborn torch"
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

# Construct command arguments
CMD_ARGS=""

# Add limit
if [ $LIMIT -gt 0 ]; then
    CMD_ARGS="$CMD_ARGS --limit $LIMIT"
else
    CMD_ARGS="$CMD_ARGS --full"
fi

# Add model
if [ "$MODEL" == "personality" ]; then
    CMD_ARGS="$CMD_ARGS --personality"
elif [ "$MODEL" == "distilbert" ]; then
    CMD_ARGS="$CMD_ARGS --distilbert"
fi

# Add LIWC option
if [ "$LIWC" == "simple" ]; then
    CMD_ARGS="$CMD_ARGS --simple-liwc"
fi

# Add demo mode
if [ "$DEMO" == "true" ]; then
    CMD_ARGS="$CMD_ARGS --demo"
fi

# Add confidence weighting
if [ "$WEIGHTED" == "true" ]; then
    CMD_ARGS="$CMD_ARGS --weighted"
fi

# Add data directory if specified
if [ -n "$DATA_DIR" ]; then
    CMD_ARGS="$CMD_ARGS --data-dir \"$DATA_DIR\""
fi

# Run the Python script
echo "Running analysis with command: python3 src/ceo_personality_comparison.py $CMD_ARGS"
eval python3 src/ceo_personality_comparison.py $CMD_ARGS

# Render if requested
if [ "$RENDER" == "true" ]; then
    echo "Rendering Quarto document..."
    quarto render results/ceo_bert_liwc_comparison.qmd
    echo "Analysis complete. Report is available at results/ceo_bert_liwc_comparison.html"
else
    echo "Analysis complete. To generate the HTML report, run:"
    echo "quarto render results/ceo_bert_liwc_comparison.qmd"
fi 