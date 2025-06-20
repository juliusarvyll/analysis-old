# Clustering & Association Rule Mining Analysis App

## Overview
This application is a modern PyQt5-based GUI for interactive data analysis, clustering, association rule mining, and AI-powered recommendations. It is designed for CSV datasets with both numeric and categorical data, supporting multi-file loading, department/dataset filtering, export features, and project save/load.

## Screenshots

Below are example screenshots of the application's main analysis and visualization features. Each plot or table is explained for clarity.

### Clustering Plot
![Clustering Plot](screenshots/cluster%20plot.png)
*Shows the result of KMeans clustering. Each line represents a cluster's feature profile, or a PCA scatter plot if there are few features. Cluster sizes are annotated. Useful for visualizing groupings in your data.*

### Elbow Method & Silhouette Score
![Elbow Method & Silhouette Score](screenshots/elbow%20method%20silhoutte%20score.png)
*Displays the Elbow Method (inertia vs. k) and Silhouette Score plots. These help determine the optimal number of clusters for KMeans. The app automatically selects the best k using the highest silhouette score.*

### Association Rule Mining (ARM) Plot
![ARM Plot](screenshots/arm%20plot.png)
*Visualizes discovered association rules as a network. Nodes represent items (feature=value), and edges represent rules (A→B), with edge width indicating rule strength (lift). Useful for exploring relationships between features.*

### ARM Results Table
![ARM Results](screenshots/arm%20results.png)
*Tabular view of discovered association rules, showing antecedents, consequents, support, confidence, and lift. Includes AI-powered analysis of the rules.*

### Descriptive Histogram
![Descriptive Histogram](screenshots/descriptive%20histogram.png)
*Shows histograms for all numeric features, with 20 bins each. Feature means are rated using customizable ranges. Useful for understanding the distribution and central tendency of each feature.*

### Recommendations
![Recommendations](screenshots/recom.png)
*Displays AI-generated recommendations based on feature means, ratings, and department. Helps users interpret the data and take action.*

### Trends Plot
![Trends Plot](screenshots/trends%20plot.png)
*Plots feature means across years (extracted from dataset filenames), with AI-powered trend analysis and insights. Useful for identifying changes and patterns over time.*

## Features
- **Load and combine multiple CSV files**
- **Department and dataset selection**
- **Tabs for Clustering, Association Rules (network plot), ARM Results (table), Descriptive Analysis, Histograms, Recommendations, and Trends**
- **Dynamic feature rating system with customizable rating ranges**
- **Groq AI integration for recommendations, descriptive analysis, and trend analysis**
- **Export to PDF (all plots, tables, and AI analyses)**
- **Save/Load project state (.pkl file)**
- **Remove individual datasets from analysis**
- **Modern, responsive, fullscreen UI with branding/logo support**
- **Configurable app title, logo, and university name via `.env` file**

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
  - The app automatically selects k using the highest silhouette score and shows both plots in a popup.
- **Preprocessing:**
  - Standard scaling of all numeric columns (`StandardScaler`)
  - PCA (Principal Component Analysis) to 2D for visualization (`sklearn.decomposition.PCA`)
- **Plot:**
  - If enough numeric features: line plot of cluster centers (feature profiles)
  - Otherwise: PCA scatter plot
  - Cluster sizes are annotated

### Principal Component Analysis (PCA)
- **Purpose:** Reduce dimensionality for visualization.
- **Math:** Projects data onto directions of maximum variance (eigenvectors of the covariance matrix).

### Association Rule Mining (Apriori)
- **Preprocessing:**
  - Categorical columns: fill NaNs, lowercase, strip, rare values replaced with 'other'
  - Numeric columns: binned into 3 quantiles (low, medium, high) and added as categorical features
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
- **ARM Results Tab:**
  - Table of discovered rules (antecedents, consequents, support, confidence, lift)
  - AI-powered analysis of rules

### Descriptive Analysis
- **Numeric summary:** min, max, mean, median, std, skew/shape for each numeric column
- **Categorical summary:** value counts for each categorical column
- **AI Analysis:** Groq AI interprets the stats and provides recommendations
- **Histograms:**
  - All numeric columns shown
  - 20 bins per histogram
  - Scrollable area with dynamic height
  - Each feature's mean is rated using customizable rating ranges

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
- Project state (datasets, AI analyses, rating ranges) can be saved/loaded as a `.pkl` file

## Configuration
- App title, logo, and university name can be customized via a `.env` file in the app directory:
  ```env
  LOGO_PATH=spup-logo.png
  UNIVERSITY_NAME=St. Paul University Philippines
  APP_TITLE=Event Feedback Analysis
  ```
- If `.env` does not exist, it will be created with defaults on first run.

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
- python-dotenv (for configuration)

## How to Run
1. Install requirements:
   ```bash
   pip install pyqt5 pandas scikit-learn matplotlib networkx mlxtend reportlab groq python-dotenv
   ```
2. (Optional) Place your logo and set up `.env` for branding.
3. Run the script:
   ```bash
   python analysis-script.py
   ```
4. Use the UI to load data, analyze, and export results. Use the **More Actions** menu for project save/load, PDF export, rating range customization, and dataset removal.

## Notes
- The app is designed for fullscreen use, but adapts to your screen size.
- All data is processed locally except for Groq AI calls (which require an API key).
- Project save/load uses Python pickle format (`.pkl`).
- Error messages are shown in the UI if analysis fails.
- AI analyses are cached per dataset/department for efficiency.

---

Feel free to modify the code for your own analysis needs!
