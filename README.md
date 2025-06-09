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

## Tab Descriptions & Technical Details

### 1. Clustering
- **Algorithm:** KMeans clustering (`sklearn.cluster.KMeans`)
- **Preprocessing:**
  - Standard scaling of all numeric columns (`StandardScaler`)
  - PCA (Principal Component Analysis) to 2D for visualization (`sklearn.decomposition.PCA`)
- **Plot:**
  - Each point is a row, colored by cluster assignment
  - Axes: PCA 1, PCA 2
- **Formulas:**
  - **KMeans:**
    - Assigns each point to the nearest centroid (minimizing within-cluster sum of squares)
    - Objective: \( \sum_{i=1}^n \min_{j} \|x_i - \mu_j\|^2 \)
  - **PCA:**
    - Projects data onto the directions of maximum variance
    - Finds eigenvectors of the covariance matrix

### 2. Association Rules (Network Plot)
- **Preprocessing:**
  - Categorical columns: fill NaNs, lowercase, strip, rare values replaced with 'other'
  - Numeric columns: binned into 3 quantiles (low, medium, high)
  - One-hot encoding (`pd.get_dummies`)
- **Algorithm:**
  - Frequent itemset mining: Apriori (`mlxtend.frequent_patterns.apriori`)
    - **Support:** Fraction of rows containing an itemset
    - **Formula:** \( \text{support}(A) = \frac{\text{count}(A)}{N} \)
  - Association rules (`mlxtend.frequent_patterns.association_rules`)
    - **Confidence:** \( \text{confidence}(A \Rightarrow B) = \frac{\text{support}(A \cup B)}{\text{support}(A)} \)
    - **Lift:** \( \text{lift}(A \Rightarrow B) = \frac{\text{confidence}(A \Rightarrow B)}{\text{support}(B)} \)
- **Plot:**
  - Nodes: items (feature=value)
  - Edges: rules (A→B), width by lift

### 3. ARM Results (Table)
- **Table columns:** Antecedents, Consequents, Support, Confidence, Lift
- **Fully scrollable and scalable**

### 4. Descriptive Analysis
- **Feature means:** Simple arithmetic mean for each numeric column
- **Dynamic rating:** User-defined ranges (e.g., needs improvement, satisfactory, etc.)
- **Categorical summary:** Value counts for each categorical column

### 5. Histograms
- **Up to 6 numeric columns per department/dataset**
- **20 bins per histogram**

### 6. Recommendations
- **Groq AI integration:**
  - Sends feature means, ratings, and department to Groq LLM
  - Returns actionable recommendations and insights

### 7. Trends
- **Plots feature means across years (from dataset filenames)**
- **Baseline:** Oldest year
- **Groq AI trend analysis:** Interprets trends and provides insights

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
