import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from typing import List, Dict
import os

class PersonalityVisualizer:
    def __init__(self, results_file: str = 'data/outputs/analysis/analysis_results.json'):
        """Initialize visualizer with analysis results."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Convert results to DataFrame for easier plotting
        self.df = self._create_dataframe()
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for result in self.results:
            row = {
                'Name': os.path.splitext(result['file_name'])[0],
                'Date': result['analysis_date']
            }
            row.update(result['results'])
            data.append(row)
        return pd.DataFrame(data)

    def plot_trait_comparison(self, trait: str = 'Neuroticism', 
                            figsize: tuple = (10, 6)):
        """Create a bar plot comparing a specific trait across leaders."""
        plt.figure(figsize=figsize)
        
        # Sort by trait value
        sorted_df = self.df.sort_values(trait, ascending=True)
        
        # Create bar plot
        ax = sns.barplot(data=sorted_df, x='Name', y=trait)
        
        # Customize plot
        plt.title(f'{trait} Comparison Across Leaders', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(sorted_df[trait]):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'data/outputs/visualizations/{trait.lower()}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_personality_radar(self, leader_name: str = None, 
                             figsize: tuple = (10, 10)):
        """Create a radar plot of personality traits for a specific leader."""
        # If no leader specified, use first one
        if leader_name is None:
            leader_name = self.df['Name'].iloc[0]
        
        # Get data for the leader
        leader_data = self.df[self.df['Name'] == leader_name].iloc[0]
        
        # Get traits and values
        traits = self.df.columns[2:].tolist()  # Skip Name and Date
        values = [leader_data[trait] for trait in traits]
        
        # Close the plot (connect last point to first)
        values += values[:1]
        traits += traits[:1]
        
        # Calculate angle for each trait
        angles = [n / float(len(traits)-1) * 2 * 3.14159 for n in range(len(traits))]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(3.14159 / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each trait and label them
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(traits[:-1])
        
        # Add title
        plt.title(f"Personality Profile: {leader_name}", y=1.05)
        
        # Save plot
        plt.savefig(f'data/outputs/visualizations/{leader_name.lower()}_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_trait_distribution(self, figsize: tuple = (15, 10)):
        """Create violin plots showing the distribution of all traits."""
        plt.figure(figsize=figsize)
        
        # Melt DataFrame to get traits in one column
        df_melted = pd.melt(self.df, 
                           id_vars=['Name', 'Date'], 
                           var_name='Trait', 
                           value_name='Score')
        
        # Create violin plot
        sns.violinplot(data=df_melted, x='Trait', y='Score')
        
        # Customize plot
        plt.title('Distribution of Personality Traits Across All Leaders', pad=20)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('data/outputs/visualizations/trait_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_all_visualizations(self):
        """Create all available visualizations."""
        # Create visualizations directory if it doesn't exist
        os.makedirs('data/outputs/visualizations', exist_ok=True)
        
        # Create individual trait comparisons
        for trait in self.df.columns[2:]:  # Skip Name and Date
            self.plot_trait_comparison(trait)
        
        # Create radar plots for each leader
        for leader in self.df['Name'].unique():
            self.plot_personality_radar(leader)
        
        # Create trait distribution plot
        self.plot_trait_distribution()

def main():
    # Create visualizer and generate all plots
    visualizer = PersonalityVisualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main() 