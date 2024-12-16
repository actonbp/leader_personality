# CEO Personality Analysis 🎯

A research project analyzing personality traits in CEO speeches using BERT-based models, with a particular focus on neuroticism and emotional stability in leadership communication.

## Project Overview 🔍

This project analyzes personality traits of Fortune 500 CEOs through their public communications, using:
- BERT-based personality detection models
- Natural Language Processing techniques
- Statistical analysis and visualization
- Focus on Big Five personality traits, especially neuroticism

## Key Features 🌟

- Automated speech processing pipeline
- Personality trait analysis using BERT
- Interactive visualizations
- Statistical analysis of trait distributions
- Comparative analysis across CEOs
- Focus on emotional stability patterns

## Project Structure 📁

```
leader_personality/
├── data/
│   ├── speeches/           # Raw speech transcripts
│   │   └── ceos/          # CEO-specific speeches
│   └── cleaned/           # Processed speech files
├── src/
│   ├── personality_analyzer.py  # Core analysis module
│   ├── visualization.py         # Data visualization
│   └── standardize_files.py     # Text preprocessing
├── results/
│   ├── analysis_report.md       # Detailed findings
│   ├── ceo_analysis.qmd        # Quarto analysis document
│   └── personality_analysis.csv # Raw analysis data
├── docs/
│   └── data_preprocessing_guidelines.md
├── requirements.txt
└── README.md
```

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/leader_personality.git
cd leader_personality
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 📊

1. **Prepare Speech Data**:
   - Place CEO speech transcripts in `data/speeches/ceos/`
   - Run standardization:
   ```bash
   python src/standardize_files.py
   ```

2. **Run Analysis**:
   ```bash
   python src/personality_analyzer.py
   ```

3. **Generate Visualizations**:
   ```bash
   python src/visualization.py
   ```

4. **View Results**:
   - Open `results/ceo_analysis.html` for interactive visualizations
   - Check `results/analysis_report.md` for detailed findings

## Key Findings 📈

- CEOs show remarkably consistent neuroticism scores (range: 0.476-0.508)
- Professional communication patterns suggest strong emotional control
- Subtle variations in handling uncertainty and challenges
- Need for more sophisticated analysis methods identified

## Future Directions 🔮

1. **Ensemble Model Approach**
   - Multiple BERT architectures
   - Traditional NLP techniques
   - Domain-specific models

2. **Data Expansion**
   - More speeches per CEO
   - Diverse communication contexts
   - Historical data analysis

3. **Enhanced Analysis**
   - Multimodal analysis (text + audio)
   - Context-aware sentiment analysis
   - Cross-validation with expert assessments

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Dependencies 📦

- Python 3.8+
- transformers==4.35.2
- torch==2.2.1
- pandas==2.1.3
- seaborn==0.13.0
- matplotlib==3.8.2
- Quarto (for report generation)

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Author ✍️

Bryan Acton

## Acknowledgments 🙏

- BERT personality model developers
- Fortune 500 CEO communications teams
- Open-source NLP community