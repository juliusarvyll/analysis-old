import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def test_baseline_pdf():
    """Test function to debug baseline comparison PDF export"""
    # Create sample data mimicking the comparison_data structure
    comparison_data = {
        'feature1': {
            'current_mean': 2.5,
            'baseline_mean': 2.0,
            'pct_change': 25.0
        },
        'feature2': {
            'current_mean': 1.8,
            'baseline_mean': 2.2,
            'pct_change': -18.2
        }
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract data for plotting
    metrics = list(comparison_data.keys())
    current_values = [comparison_data[m]['current_mean'] for m in metrics]
    baseline_values = [comparison_data[m]['baseline_mean'] for m in metrics]
    percentage_changes = [comparison_data[m]['pct_change'] for m in metrics]

    # Set up bar positions
    bar_width = 0.35
    index = np.arange(len(metrics))

    # Create bars
    current_bars = ax.bar(index, current_values, bar_width,
                         label='Current', color='#3498db', alpha=0.8)
    baseline_bars = ax.bar(index + bar_width, baseline_values, bar_width,
                         label='Baseline', color='#2ecc71', alpha=0.8)

    # Add percentage changes as text above bars
    for i, (curr, base, perc) in enumerate(zip(current_values, baseline_values, percentage_changes)):
        if curr > base:
            color = 'green' if perc > 0 else 'red'
            y_pos = max(curr, base) + 0.05
            text = f"+{perc:.1f}%" if perc > 0 else f"{perc:.1f}%"
        else:
            color = 'red' if perc < 0 else 'green'
            y_pos = max(curr, base) + 0.05
            text = f"{perc:.1f}%"

        ax.text(i + bar_width/2, y_pos, text,
               ha='center', va='bottom', color=color, fontweight='bold')

    # Customize plot appearance
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Test Comparison with Baseline Metrics', fontsize=16, pad=20)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save to PDF
    with PdfPages('test_baseline.pdf') as pdf:
        pdf.savefig(fig)

    print("Test baseline PDF created successfully!")

def test_cluster_pdf():
    """Test function to debug cluster plot PDF export"""
    # Create sample clustered data
    np.random.seed(42)
    n_samples = 200
    n_features = 2
    n_clusters = 3

    # Generate sample data
    X = np.random.randn(n_samples, n_features)

    # Create clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Scatter plot
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        s=50,
        alpha=0.7,
        cmap='viridis',
        edgecolors='w',
        linewidth=0.5
    )

    # Add cluster centers
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c='red',
        alpha=0.8,
        marker='X',
        edgecolors='black',
        linewidth=1.5,
        label='Cluster Centers'
    )

    # Add labels and title
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.title(f"Test Cluster Plot\n{n_clusters} Clusters Identified", fontsize=16, pad=20)

    # Add legend
    plt.legend(loc='upper right', title="Clusters",
              fancybox=True, framealpha=0.7,
              title_fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Cluster Label', fontsize=12)

    # Set background color
    ax.set_facecolor('#f8f8f8')

    # Save to PDF
    with PdfPages('test_cluster.pdf') as pdf:
        pdf.savefig(fig)

    print("Test cluster PDF created successfully!")

if __name__ == "__main__":
    test_baseline_pdf()
    test_cluster_pdf()
