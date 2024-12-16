import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_personality_traits(results_file: str, output_dir: str = "results"):
    """Create various visualizations of personality analysis results."""
    # Read results
    df = pd.read_csv(results_file)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Heatmap of all traits
    plt.figure(figsize=(12, 8))
    trait_columns = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    heatmap_data = df.pivot(columns=trait_columns, values=trait_columns)
    sns.heatmap(df[trait_columns], annot=True, cmap='RdYlBu', center=0.5,
                yticklabels=df['Name'], fmt='.3f')
    plt.title('Personality Traits Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/traits_heatmap.png")
    plt.close()
    
    # 2. Bar plot comparing neuroticism
    plt.figure(figsize=(12, 6))
    neuroticism_data = df.sort_values('Neuroticism', ascending=False)
    sns.barplot(data=neuroticism_data, x='Name', y='Neuroticism')
    plt.xticks(rotation=45, ha='right')
    plt.title('Neuroticism Scores by CEO')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/neuroticism_comparison.png")
    plt.close()
    
    # 3. Radar chart of traits for each CEO
    trait_cols = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    angles = np.linspace(0, 2*np.pi, len(trait_cols), endpoint=False)
    
    # Close the plot by appending the first value
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
    
    for idx, row in df.iterrows():
        values = row[trait_cols].values
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Name'])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(trait_cols)
    ax.set_ylim(0.4, 0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Personality Traits Radar Chart')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/traits_radar.png")
    plt.close()
    
    # 4. Box plot of all traits
    plt.figure(figsize=(12, 6))
    df_melted = df.melt(id_vars=['Name'], value_vars=trait_cols, 
                        var_name='Trait', value_name='Score')
    sns.boxplot(data=df_melted, x='Trait', y='Score')
    plt.title('Distribution of Personality Traits Across CEOs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/traits_distribution.png")
    plt.close()
    
    # 5. Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[trait_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Correlation Between Personality Traits')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trait_correlations.png")
    plt.close()
    
    # Generate summary statistics
    summary_stats = df[trait_cols].describe()
    summary_stats.to_csv(f"{output_dir}/trait_statistics.csv")
    
    # Return summary for display
    return summary_stats

if __name__ == "__main__":
    # Create visualizations
    results_file = "results/personality_analysis.csv"
    summary = plot_personality_traits(results_file)
    
    print("\nPersonality Traits Summary Statistics:")
    print("====================================")
    print(summary) 