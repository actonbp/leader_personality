# Speech Personality Analyzer 🎯

A user-friendly tool to analyze personality traits in speeches from leaders (CEOs, coaches, politicians). The tool focuses on the Big Five personality traits, with special attention to neuroticism.

## What Does This Tool Do? 🤔

This tool helps you:
- Analyze speeches to identify personality traits
- Create beautiful visualizations of the results
- Compare different leaders' personality profiles
- Track changes in personality traits over time

## Quick Start Guide 🚀

### 1. First-Time Setup

```bash
# Clone this repository
git clone https://github.com/yourusername/leader_personality.git
cd leader_personality

# Create a virtual environment (only needed once)
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Adding Speeches to Analyze 📝

1. Place your speech text files in the appropriate folder:
   - CEO speeches go in: `data/speeches/ceos/`
   - Coach speeches go in: `data/speeches/coaches/`
   - Political speeches go in: `data/speeches/politicians/`

2. Text files should be plain text (.txt) format
   - Remove any special formatting
   - One speech per file
   - Use clear filenames (e.g., `jane_fraser_investor_day_2024.txt`)

### 3. Running the Analysis 📊

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)

# Run the personality analysis
python src/personality_analyzer.py

# Create visualizations
python src/visualization.py
```

### 4. Finding Your Results 📈

After running the scripts, you'll find:
- Analysis results: `data/outputs/analysis/analysis_results.json`
- Visualizations: `data/outputs/visualizations/`
  - Individual trait comparisons (e.g., `neuroticism_comparison.png`)
  - Personality radar charts for each leader
  - Overall trait distributions

## Repository Structure 📁

```
leader_personality/
├── data/
│   ├── speeches/           # Put your speech files here
│   │   ├── ceos/          # CEO speeches
│   │   ├── coaches/       # Coach speeches
│   │   └── politicians/   # Political speeches
│   └── outputs/           # Results are saved here
│       ├── analysis/      # JSON analysis results
│       └── visualizations/# Generated plots and charts
├── src/
│   ├── personality_analyzer.py  # Main analysis script
│   └── visualization.py   # Visualization script
├── venv/                  # Virtual environment (created during setup)
├── requirements.txt       # Required Python packages
└── README.md             # This file
```

## Visualization Examples 🎨

The tool creates several types of visualizations:
1. Bar charts comparing specific traits across leaders
2. Radar charts showing all traits for individual leaders
3. Distribution plots showing overall patterns

All visualizations are saved as PNG files in `data/outputs/visualizations/`

## Troubleshooting 🔧

Common issues and solutions:

1. **"Command not found: python"**
   - Make sure Python is installed on your system
   - Try using `python3` instead of `python`

2. **"No such file or directory: venv"**
   - Make sure you're in the correct directory
   - Try creating the virtual environment again

3. **Import errors after installation**
   - Make sure your virtual environment is activated
   - Try reinstalling requirements: `pip install -r requirements.txt`

## Need Help? 🆘

If you run into any issues:
1. Check the Troubleshooting section above
2. Make sure all files are in the correct directories
3. Ensure your virtual environment is activated
4. Check that your speech files are plain text (.txt)

## Future Features 🔮

- Support for more personality models
- Interactive visualizations
- Batch processing of multiple speeches
- Time-series analysis of personality changes
- Industry-specific benchmarking