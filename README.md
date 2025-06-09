# Clustering & Association Rule Mining Analysis App

## Overview
This application is a modern PyQt5-based GUI for interactive data analysis, clustering, association rule mining, and AI-powered recommendations. It is designed for CSV datasets with both numeric and categorical data, supporting multi-file loading, department/dataset filtering, and export features.

## Features
- **Load and combine multiple CSV files**
- **Department and dataset selection**
- **Tabs for Clustering, Association Rules (network plot), ARM Results (table), Descriptive Analysis, Histograms, Recommendations, and Trends**
- **Dynamic feature rating system**
- **Groq AI integration for recommendations and trend analysis**
- **Export to PDF, Save/Load project**
- **Modern, responsive, fullscreen UI**

## Technical Details & Math Used

### Clustering (KMeans)
- **Algorithm:** KMeans clustering (`sklearn.cluster.KMeans`)
- **Objective:** Partition data into k clusters by minimizing within-cluster sum of squares:
  \[
  \text{argmin}_C \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2
  \]
  where $\mu_i$ is the centroid of cluster $C_i$.
- **k Selection:**
  - **Elbow Method:** Plot inertia (sum of squared distances to cluster centers) for different k and look for the "elbow" point.
  - **Silhouette Score:** Measures how similar a point is to its own cluster vs. other clusters:
    \[
    s = \frac{b - a}{\max(a, b)}
    \]
    where $a$ is the mean intra-cluster distance, $b$ is the mean nearest-cluster distance.
  - The app automatically selects k using the highest silhouette score.
- **Preprocessing:**
  - Standard scaling of all numeric columns (`StandardScaler`)
  - PCA (Principal Component Analysis) to 2D for visualization (`sklearn.decomposition.PCA`)
- **Plot:**
  - Each point is a row, colored by cluster assignment
  - Axes: PCA 1, PCA 2

### Principal Component Analysis (PCA)
- **Purpose:** Reduce dimensionality for visualization.
- **Math:** Projects data onto directions of maximum variance (eigenvectors of the covariance matrix).

### Association Rule Mining (Apriori)
- **Preprocessing:**
  - Categorical columns: fill NaNs, lowercase, strip, rare values replaced with 'other'
  - Numeric columns: binned into 3 quantiles (low, medium, high)
  - One-hot encoding (`pd.get_dummies`)
- **Algorithm:**
  - **Apriori:** Finds frequent itemsets with minimum support.
    - **Support:**
      \[
      \text{support}(A) = \frac{\text{count}(A)}{N}
      \]
  - **Association Rules:**
    - **Confidence:**
      \[
      \text{confidence}(A \Rightarrow B) = \frac{\text{support}(A \cup B)}{\text{support}(A)}
      \]
    - **Lift:**
      \[
      \text{lift}(A \Rightarrow B) = \frac{\text{confidence}(A \Rightarrow B)}{\text{support}(B)}
      \]
- **Plot:**
  - Nodes: items (feature=value)
  - Edges: rules (A→B), width by lift

### Descriptive Analysis
- **Numeric summary:** min, max, mean, median, std, skew/shape for each numeric column
- **Categorical summary:** value counts for each categorical column
- **AI Analysis:** Groq AI interprets the stats and provides recommendations

### Histograms
- **All numeric columns shown**
- **20 bins per histogram**

### Recommendations & Trends
- **Groq AI integration:**
  - Sends feature means, ratings, and department to Groq LLM
  - Returns actionable recommendations and insights
- **Trends:**
  - Plots feature means across years (from dataset filenames)
  - Groq AI trend analysis interprets and provides insights

## Data Flow
- User loads CSV(s) → DataFrames stored in `self.datasets`
- User selects dataset/department → Filters data for analysis
- Each tab updates based on current selection
- All analysis is performed in-memory using pandas, sklearn, mlxtend, and matplotlib

## Requirements
- Python 3.7+
- PyQt5
- pandas
- scikit-learn
- matplotlib
- networkx
- mlxtend
- reportlab (for PDF export)
- groq (for AI recommendations)

## How to Run
1. Install requirements:
   ```bash
   pip install pyqt5 pandas scikit-learn matplotlib networkx mlxtend reportlab groq
   ```
2. Run the script:
   ```bash
   python analysis-script.py
   ```
3. Use the UI to load data, analyze, and export results.

## Notes
- The app is designed for fullscreen use, but adapts to your screen size.
- All data is processed locally except for Groq AI calls (which require an API key).
- Project save/load uses Python pickle format.

---

Feel free to modify the code for your own analysis needs!
