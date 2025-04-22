"""
Improved CEO Embedding Visualization

Creates a cleaner visualization of the CEO embedding space by:
1. Identifying natural clusters 
2. Highlighting only representative CEOs
3. Using better visual encoding for clarity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Set random seed for reproducibility
np.random.seed(42)

# Load the embedding coordinates
df = pd.read_csv("results/ceo_embedding_coordinates.csv")

# Function to create confidence ellipse
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's size.
    facecolor : str
        The color for the ellipse.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse), (mean_x, mean_y)

# Identify clusters using DBSCAN
X = df[['x', 'y']].values
clustering = DBSCAN(eps=0.8, min_samples=5).fit(X)
df['cluster'] = clustering.labels_

# Create industry mapping
# This is a simplified approach - in a real implementation, 
# you would use a proper database or API to get industry information
industries = {
    'IBM': 'Technology', 'ADBE': 'Technology', 'QCOM': 'Technology', 'AAPL': 'Technology',
    'MSFT': 'Technology', 'NVDA': 'Technology', 'HPE': 'Technology', 'CSCO': 'Technology',
    'AMD': 'Technology', 'INTC': 'Technology', 'META': 'Technology', 'ORCL': 'Technology',
    
    'CVS': 'Healthcare', 'HUM': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'ABT': 'Healthcare', 'AMGN': 'Healthcare', 'JNJ': 'Healthcare', 'MRK': 'Healthcare',
    'CNC': 'Healthcare', 'ELV': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare',
    
    'XOM': 'Energy', 'CVX': 'Energy', 'PCG': 'Energy', 'DVN': 'Energy', 
    'COP': 'Energy', 'OXY': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'OKE': 'Energy', 'SRE': 'Energy', 'DUK': 'Energy', 'NEE': 'Energy',
    
    'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance',
    'MS': 'Finance', 'C': 'Finance', 'AXP': 'Finance', 'BLK': 'Finance',
    'COF': 'Finance', 'STT': 'Finance', 'USB': 'Finance', 'TFC': 'Finance',
    
    'WMT': 'Retail', 'TGT': 'Retail', 'HD': 'Retail', 'LOW': 'Retail',
    'COST': 'Retail', 'AMZN': 'Retail', 'DG': 'Retail', 'KR': 'Retail',
    'TJX': 'Retail', 'BBY': 'Retail', 'M': 'Retail', 'JWN': 'Retail',
    
    # Add more mappings as needed
}

# Add industry column to dataframe
df['Industry'] = df['Company'].map(lambda x: industries.get(x, 'Other'))

# Count clusters
cluster_counts = df['cluster'].value_counts()
print(f"Found {len(cluster_counts)} clusters, including {sum(cluster_counts == -1)} noise points")

# Function to select notable representatives from each cluster
def select_representatives(df, n_per_cluster=2, include_female=True):
    """Select representative CEOs from each cluster."""
    representatives = []
    
    # For each cluster
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        
        # If this is a valid cluster (not noise points)
        if cluster >= 0:
            # Always include at least one female CEO if available
            female_ceos = cluster_df[cluster_df['Gender'] == 'Female']
            if not female_ceos.empty and include_female:
                # Choose a random female CEO
                female_rep = female_ceos.sample(min(1, len(female_ceos)))
                representatives.append(female_rep)
                
                # Remove selected female CEO from consideration
                remaining = cluster_df[~cluster_df.index.isin(female_rep.index)]
                
                # Choose additional representatives if needed
                if len(remaining) > 0 and n_per_cluster > 1:
                    male_reps = remaining.sample(min(n_per_cluster-1, len(remaining)))
                    representatives.append(male_reps)
            else:
                # No female CEOs, just choose n_per_cluster representatives
                reps = cluster_df.sample(min(n_per_cluster, len(cluster_df)))
                representatives.append(reps)
        else:
            # For noise points, select a few notable ones
            if len(cluster_df) > 0:
                notable_noise = cluster_df.sample(min(3, len(cluster_df)))
                representatives.append(notable_noise)
    
    # Combine all representatives
    if representatives:
        return pd.concat(representatives)
    else:
        return pd.DataFrame()

# Select representative CEOs
representatives = select_representatives(df, n_per_cluster=3)

# Create the visualization
plt.figure(figsize=(14, 10))

# Set up colors by industry
industry_colors = {
    'Technology': '#3498db',  # Blue
    'Healthcare': '#2ecc71',  # Green
    'Energy': '#e74c3c',      # Red
    'Finance': '#f39c12',     # Orange
    'Retail': '#9b59b6',      # Purple
    'Other': '#7f8c8d'        # Gray
}

# Add some jitter to prevent direct overlap
jitter_strength = 0.05
df['x_jitter'] = df['x'] + np.random.normal(0, jitter_strength, size=len(df))
df['y_jitter'] = df['y'] + np.random.normal(0, jitter_strength, size=len(df))

# Create the main scatter plot (all CEOs as small dots)
for industry, color in industry_colors.items():
    ind_df = df[df['Industry'] == industry]
    plt.scatter(
        ind_df['x_jitter'], 
        ind_df['y_jitter'], 
        s=40,  # Smaller size for background dots
        alpha=0.3,  # Transparency
        color=color,
        edgecolor='none',
        label=industry
    )

# Identify and draw cluster ellipses
for cluster in sorted(df['cluster'].unique()):
    if cluster >= 0:  # Skip noise points (cluster = -1)
        cluster_df = df[df['cluster'] == cluster]
        
        # Determine dominant industry in cluster
        industry_counts = cluster_df['Industry'].value_counts()
        dominant_industry = industry_counts.index[0] if not industry_counts.empty else 'Other'
        
        # Get color for this cluster
        color = industry_colors.get(dominant_industry, '#7f8c8d')
        
        # Create confidence ellipse if enough points
        if len(cluster_df) >= 5:
            ellipse, (center_x, center_y) = confidence_ellipse(
                cluster_df['x'].values, 
                cluster_df['y'].values, 
                plt.gca(), 
                n_std=1.5, 
                alpha=0.2, 
                facecolor=color, 
                edgecolor=color, 
                linewidth=2
            )
            
            # Add cluster label
            plt.text(
                center_x, center_y,
                f"Cluster {cluster}\n({dominant_industry})",
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10,
                fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
            )

# Highlight representative CEOs
for _, row in representatives.iterrows():
    gender_marker = 'X' if row['Gender'] == 'Female' else 'o'
    color = industry_colors.get(row['Industry'], '#7f8c8d')
    
    # Plot the CEO marker
    plt.scatter(
        row['x'], row['y'],
        s=120,  # Larger size for representatives
        marker=gender_marker,
        color=color,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.9,
        zorder=10  # Ensure representatives are on top
    )
    
    # Add CEO name with a white outline for readability
    text = plt.text(
        row['x'] + 0.1, row['y'] + 0.1,
        f"{row['CEO']} ({row['Company']})",
        fontsize=9,
        fontweight='bold',
        color='black',
        zorder=11
    )
    text.set_path_effects([
        pe.withStroke(linewidth=3, foreground='white')
    ])

# Add title and labels
plt.title('CEO Speech Patterns in Embedding Space', fontsize=16, pad=20)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(), by_label.keys(),
    title='Industry',
    loc='upper right',
    fontsize=10,
    title_fontsize=12
)

# Add annotations explaining the dimensions
plt.annotate(
    'Technical/Operational Focus',
    xy=(-4, -2),
    xytext=(-5, -2),
    fontsize=10,
    arrowprops=dict(arrowstyle='->'),
    horizontalalignment='right'
)

plt.annotate(
    'Market/Customer Focus',
    xy=(4, -2),
    xytext=(5, -2),
    fontsize=10,
    arrowprops=dict(arrowstyle='->'),
    horizontalalignment='left'
)

plt.annotate(
    'Strategic/Visionary Language',
    xy=(-2, 7),
    xytext=(-2, 8),
    fontsize=10,
    arrowprops=dict(arrowstyle='->'),
    verticalalignment='top',
    horizontalalignment='center'
)

plt.annotate(
    'Practical/Operational Language',
    xy=(-2, 3),
    xytext=(-2, 2),
    fontsize=10,
    arrowprops=dict(arrowstyle='->'),
    verticalalignment='bottom',
    horizontalalignment='center'
)

# Add grid
plt.grid(alpha=0.2, linestyle='--')

# Add a note about gender markers
plt.figtext(
    0.5, 0.01,
    "Note: 'X' markers represent female CEOs, 'o' markers represent male CEOs. Ellipses indicate identified clusters.",
    ha='center',
    fontsize=10
)

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig('results/ceo_embedding_clusters.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'results/ceo_embedding_clusters.png'")

# Also create a force-directed graph version for alternative visualization
try:
    import networkx as nx
    
    # Create a graph based on proximity
    G = nx.Graph()
    
    # Add nodes
    for i, row in representatives.iterrows():
        G.add_node(
            row['CEO'], 
            x=row['x'], 
            y=row['y'], 
            company=row['Company'],
            gender=row['Gender'],
            industry=row['Industry'],
            cluster=row['cluster']
        )
    
    # Add edges based on proximity (distance in embedding space)
    representatives_list = representatives.to_dict('records')
    for i, ceo1 in enumerate(representatives_list):
        for ceo2 in representatives_list[i+1:]:
            # Calculate Euclidean distance
            dist = np.sqrt((ceo1['x'] - ceo2['x'])**2 + (ceo1['y'] - ceo2['y'])**2)
            
            # Only connect if relatively close (threshold can be adjusted)
            if dist < 1.5:
                G.add_edge(ceo1['CEO'], ceo2['CEO'], weight=1.0/dist)
    
    # Create network visualization
    plt.figure(figsize=(16, 12))
    
    # Set node colors by industry
    node_colors = [industry_colors.get(G.nodes[n]['industry'], '#7f8c8d') for n in G.nodes]
    
    # Set node shapes by gender
    node_shapes = []
    for n in G.nodes:
        if G.nodes[n]['gender'] == 'Female':
            node_shapes.append('X')
        else:
            node_shapes.append('o')
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Draw the network
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=300, 
        alpha=0.8,
        linewidths=1,
        edgecolors='black'
    )
    
    nx.draw_networkx_edges(
        G, pos, 
        alpha=0.3, 
        width=1.0
    )
    
    nx.draw_networkx_labels(
        G, pos, 
        font_size=8, 
        font_weight='bold'
    )
    
    # Add a title
    plt.title('CEO Similarity Network Based on Speech Patterns', fontsize=16, pad=20)
    
    # Add legend for industries
    for industry, color in industry_colors.items():
        plt.plot([], [], 'o', color=color, label=industry)
    
    plt.legend(title='Industry', loc='upper right')
    
    # Remove axes
    plt.axis('off')
    
    # Add annotation explaining the network
    plt.figtext(
        0.5, 0.01,
        "Note: CEOs are connected if their speech patterns are similar. Proximity indicates similarity in language use.",
        ha='center',
        fontsize=10
    )
    
    # Save the network visualization
    plt.savefig('results/ceo_speech_network.png', dpi=300, bbox_inches='tight')
    print("Network visualization saved to 'results/ceo_speech_network.png'")
    
except ImportError:
    print("NetworkX not installed. Skipping network visualization.")

print("Analysis complete!")