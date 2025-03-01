import gc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import networkx as nx
from kneed import KneeLocator
from sklearn.decomposition import PCA
import logging
from sklearn.metrics import silhouette_score
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add to global variables section
simulation_tab = None

def analyze_event_ratings(df):
    """
    Analyze event ratings using fixed 0-3 rating scale with input validation
    Returns:
        - avg_scores: Mean scores for each feature
        - needs_improvement: Features scoring <= 0.74
        - moderately_satisfactory: Features scoring 0.75-1.49
        - satisfactory: Features scoring 1.50-2.24
        - very_satisfactory: Features scoring >= 2.25
    """
    try:
        # Validate input data
        if df.empty:
            raise ValueError("Empty dataframe provided")

        # Get only numeric columns (exclude department_name)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if 'department_name' in numeric_columns:
            numeric_columns = numeric_columns.drop('department_name')

        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in the data")

        # Convert to numeric, coerce errors to NaN
        numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Calculate mean scores, handling NaN values
        avg_scores = numeric_df.mean().round(2)

        # Fixed rating scale thresholds
        needs_improvement_threshold = 0.74    # 0.00-0.74
        moderately_satisfactory_threshold = 1.49  # 0.75-1.49
        satisfactory_threshold = 2.24        # 1.50-2.24
        very_satisfactory_threshold = 3.00    # 2.25-3.00

        # Validate scores are within expected range
        if (avg_scores < 0).any() or (avg_scores > 3).any():
            logging.warning("Some scores are outside the expected 0-3 range")

        # Categorize scores based on fixed ranges
        needs_improvement = avg_scores[avg_scores <= needs_improvement_threshold]
        moderately_satisfactory = avg_scores[(avg_scores > needs_improvement_threshold) &
                                          (avg_scores <= moderately_satisfactory_threshold)]
        satisfactory = avg_scores[(avg_scores > moderately_satisfactory_threshold) &
                               (avg_scores <= satisfactory_threshold)]
        very_satisfactory = avg_scores[avg_scores > satisfactory_threshold]

        # Log categorization results
        logging.info(f"Total features analyzed: {len(avg_scores)}")
        logging.info(f"Features needing improvement: {len(needs_improvement)}")
        logging.info(f"Features moderately satisfactory: {len(moderately_satisfactory)}")
        logging.info(f"Features satisfactory: {len(satisfactory)}")
        logging.info(f"Features very satisfactory: {len(very_satisfactory)}")

        return avg_scores, needs_improvement, moderately_satisfactory, satisfactory, very_satisfactory

    except Exception as e:
        logging.error(f"Error in analyze_event_ratings: {str(e)}")
        raise

def prepare_for_association_rules(df, selected_features, year=None, sample_size=None, random_state=42):
    """
    Convert selected features to transactions with ratings using fixed rating scale.
    Optimized for memory efficiency and data quality with sampling for large datasets.

    Args:
        df: DataFrame containing the data
        selected_features: List of features to include in the analysis
        year: Optional year filter
        sample_size: Optional sample size for large datasets (int or float percentage)
        random_state: Random seed for reproducible sampling
    """
    try:
        logging.info("Starting preparation of association rules data")
        logging.debug(f"Selected features: {selected_features}")

        if df.empty or not selected_features:
            logging.error("Empty dataframe or no features selected")
            return pd.DataFrame()

        # Sample data if needed for large datasets
        original_size = len(df)
        if sample_size is not None:
            if isinstance(sample_size, float) and 0 < sample_size < 1:
                # Sample by percentage
                df = df.sample(frac=sample_size, random_state=random_state)
            elif isinstance(sample_size, int) and sample_size > 0 and sample_size < len(df):
                # Sample by absolute count
                df = df.sample(n=sample_size, random_state=random_state)

            logging.info(f"Sampled data from {original_size} to {len(df)} rows ({len(df)/original_size:.1%})")

        # Use a more memory-efficient approach for large datasets
        if len(df) > 10000:
            return _prepare_for_association_rules_chunked(df, selected_features)
        else:
            return _prepare_for_association_rules_standard(df, selected_features)

    except Exception as e:
        logging.error(f"Error in prepare_for_association_rules: {e}")
        return pd.DataFrame()

def _prepare_for_association_rules_standard(df, selected_features):
    """Standard implementation for smaller datasets"""
    binary_df = pd.DataFrame(index=df.index)

    # Define rating categories based on fixed scale (0-3)
    rating_categories = {
        'Needs_Improvement': (0, 0.74),
        'Moderately_Satisfactory': (0.75, 1.49),
        'Satisfactory': (1.50, 2.24),
        'Very_Satisfactory': (2.25, 3.00)
    }

    def categorize_rating(x, categories=rating_categories):
        if pd.isna(x):
            return 'Missing'
        for category, (lower, upper) in categories.items():
            if lower <= x <= upper:
                return category
        return 'Out_of_Range'

    # Process selected features efficiently
    for feature in selected_features:
        if feature in df.columns:
            # Convert to numeric and handle errors
            numeric_values = pd.to_numeric(df[feature], errors='coerce')

            # Skip features with too many missing values
            missing_pct = numeric_values.isna().mean() * 100
            if missing_pct > 50:
                logging.warning(f"Skipping feature {feature} - {missing_pct:.1f}% missing values")
                continue

            categories = numeric_values.apply(categorize_rating)

            # Check distribution of categories
            category_counts = categories.value_counts(normalize=True) * 100

            # Log category distribution
            for category, pct in category_counts.items():
                logging.debug(f"{feature} - {category}: {pct:.1f}%")

            # Create binary columns for each rating category
            for category in rating_categories.keys():
                # Only create columns for categories that appear with sufficient frequency
                category_count = (categories == category).sum()
                if category_count > 0:
                    col_name = f"{feature}_{category}"
                    binary_df[col_name] = (categories == category).astype(int)

                    # Log column creation
                    pct = (category_count / len(df)) * 100
                    logging.debug(f"Created column {col_name}: {category_count} occurrences ({pct:.1f}%)")

            # Log feature processing
            valid_count = numeric_values.notna().sum()
            logging.debug(f"Processed {feature}: {valid_count} valid values")

    # Validate final binary dataframe
    if binary_df.empty:
        logging.warning("No valid binary columns created")
        return pd.DataFrame()

    # Remove columns with too few occurrences (less than 1%)
    min_count = len(df) * 0.01
    column_counts = binary_df.sum()
    low_count_columns = column_counts[column_counts < min_count].index.tolist()

    if low_count_columns:
        logging.info(f"Removing {len(low_count_columns)} columns with less than 1% occurrence")
        binary_df = binary_df.drop(columns=low_count_columns)

    if binary_df.empty:
        logging.warning("No columns remain after filtering low-occurrence columns")
        return pd.DataFrame()

    logging.info(f"Created binary dataframe with {len(binary_df.columns)} columns")
    return binary_df

def _prepare_for_association_rules_chunked(df, selected_features, chunk_size=5000):
    """Chunked implementation for larger datasets to reduce memory usage"""
    # Initialize an empty DataFrame to store the results
    binary_df = pd.DataFrame(index=df.index)

    # Define rating categories based on fixed scale (0-3)
    rating_categories = {
        'Needs_Improvement': (0, 0.74),
        'Moderately_Satisfactory': (0.75, 1.49),
        'Satisfactory': (1.50, 2.24),
        'Very_Satisfactory': (2.25, 3.00)
    }

    # Process features one by one to minimize memory usage
    for feature in selected_features:
        if feature not in df.columns:
            continue

        # Convert to numeric and handle errors
        numeric_values = pd.to_numeric(df[feature], errors='coerce')

        # Skip features with too many missing values
        missing_pct = numeric_values.isna().mean() * 100
        if missing_pct > 50:
            logging.warning(f"Skipping feature {feature} - {missing_pct:.1f}% missing values")
            continue

        # Process in chunks to reduce memory usage
        for category, (lower, upper) in rating_categories.items():
            col_name = f"{feature}_{category}"
            # Use vectorized operations instead of apply for better performance
            column_data = ((numeric_values >= lower) & (numeric_values <= upper)).astype(int)

            # Only add columns with sufficient occurrences (at least 1%)
            count = column_data.sum()
            if count >= len(df) * 0.01:
                binary_df[col_name] = column_data
                # Log column creation
                pct = (count / len(df)) * 100
                logging.debug(f"Created column {col_name}: {count} occurrences ({pct:.1f}%)")

        # Log feature processing
        valid_count = numeric_values.notna().sum()
        logging.debug(f"Processed {feature}: {valid_count} valid values")

    # Validate final binary dataframe
    if binary_df.empty:
        logging.warning("No valid binary columns created")
        return pd.DataFrame()

    logging.info(f"Created binary dataframe with {len(binary_df.columns)} columns and {len(binary_df)} rows")
    return binary_df

def generate_association_rules(binary_df, min_support=0.05, min_confidence=0.5, min_lift=1.0,
                              max_len=3, use_parallel=True):
    """
    Generate association rules from binary data with optimized parameters and validation.
    Includes parallel processing support for large datasets.

    Args:
        binary_df: Binary DataFrame prepared for association rule mining
        min_support: Minimum support threshold (adjusted automatically for large datasets)
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold for filtering rules
        max_len: Maximum length of itemsets
        use_parallel: Whether to use parallel processing
    """
    try:
        logging.info("Starting association rules generation")

        if binary_df.empty:
            logging.warning("Empty DataFrame provided for association rules")
            return pd.DataFrame()

        # For very large datasets, adjust parameters automatically
        row_count = len(binary_df)
        col_count = len(binary_df.columns)

        logging.info(f"Association rule mining on {row_count} rows and {col_count} columns")

        # Calculate data density (percentage of non-zero values)
        density = binary_df.sum().sum() / (row_count * col_count) * 100
        logging.info(f"Data density: {density:.2f}%")

        # Adjust parameters based on data characteristics
        if density < 5:
            # For very sparse data, lower the support threshold
            logging.info("Data is very sparse, lowering support threshold")
            min_support = min(min_support, 0.02)

        if row_count > 100000:
            logging.info(f"Large dataset detected ({row_count} rows). Adjusting parameters.")
            # Increase min_support for very large datasets to reduce computation
            min_support = max(min_support, 0.1)
            max_len = min(max_len, 2)  # Reduce max_len for very large datasets
        elif row_count > 50000:
            min_support = max(min_support, 0.075)
        elif row_count > 10000:
            min_support = max(min_support, 0.05)
        elif row_count < 1000:
            # For small datasets, use a lower minimum support
            min_support = min(min_support, 0.03)

        # Ensure min_support is at least 2 transactions
        adjusted_min_support = max(min_support, 2 / row_count)
        logging.info(f"Using adjusted min_support: {adjusted_min_support}")

        # Generate frequent itemsets with optimized parameters
        try:
            # Set up progress tracking
            start_time = time.time()
            logging.info(f"Starting apriori algorithm with {row_count} rows and {col_count} columns")

            # Configure parallel processing if enabled
            if use_parallel and row_count > 5000:
                logging.info("Using parallel processing")
                frequent_itemsets = apriori(
                    binary_df,
                    min_support=adjusted_min_support,
                    use_colnames=True,
                    max_len=max_len,
                    verbose=1,
                    low_memory=True  # Use low memory mode for large datasets
                    # n_jobs parameter removed as it's not supported
                )
            else:
                frequent_itemsets = apriori(
                    binary_df,
                    min_support=adjusted_min_support,
                    use_colnames=True,
                    max_len=max_len,
                    verbose=1,
                    low_memory=True   # Use low memory mode for large datasets
                )

            apriori_time = time.time() - start_time
            logging.info(f"Apriori completed in {apriori_time:.2f} seconds")

            if frequent_itemsets is None or frequent_itemsets.empty:
                logging.warning(f"No frequent itemsets found with min_support={adjusted_min_support}")

                # Try with a lower support threshold as a fallback
                if adjusted_min_support > 0.01:
                    fallback_support = max(0.01, adjusted_min_support / 2)
                    logging.info(f"Retrying with lower support threshold: {fallback_support}")

                    frequent_itemsets = apriori(
                        binary_df,
                        min_support=fallback_support,
                        use_colnames=True,
                        max_len=max_len,
                        verbose=1,
                        low_memory=True
                    )

                    if frequent_itemsets is None or frequent_itemsets.empty:
                        logging.warning(f"Still no frequent itemsets found with min_support={fallback_support}")
                        return pd.DataFrame()
                else:
                    return pd.DataFrame()

            logging.info(f"Found {len(frequent_itemsets)} frequent itemsets")

            # For very large results, sample the frequent itemsets to reduce memory usage
            if len(frequent_itemsets) > 10000:
                logging.info(f"Large number of frequent itemsets ({len(frequent_itemsets)}). Sampling top 10000 by support.")
                frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(10000)

            # Generate rules with optimized confidence threshold
            rules_start_time = time.time()

            # Try with progressively lower confidence thresholds if needed
            confidence_thresholds = [min_confidence, 0.3, 0.2]
            rules = pd.DataFrame()

            for conf_threshold in confidence_thresholds:
                logging.info(f"Trying with confidence threshold: {conf_threshold}")

                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=conf_threshold,
                    support_only=False
                )

                if not rules.empty:
                    logging.info(f"Found {len(rules)} rules with confidence threshold {conf_threshold}")
                    break
                else:
                    logging.warning(f"No rules found with confidence threshold {conf_threshold}")

            rules_time = time.time() - rules_start_time
            logging.info(f"Rules generation completed in {rules_time:.2f} seconds")

            if rules.empty:
                logging.warning("No rules generated with any confidence threshold")
                return pd.DataFrame()

            # Filter rules by lift for significance
            min_lift_thresholds = [min_lift, 0.8, 0.5]
            significant_rules = pd.DataFrame()

            for lift_threshold in min_lift_thresholds:
                significant_rules = rules[rules['lift'] > lift_threshold]

                if not significant_rules.empty:
                    logging.info(f"Found {len(significant_rules)} rules with lift threshold {lift_threshold}")
                    break
                else:
                    logging.warning(f"No rules found with lift threshold {lift_threshold}")

            if significant_rules.empty:
                logging.warning("No significant rules found with any lift threshold")
                return pd.DataFrame()

            # For very large results, limit the number of rules
            if len(significant_rules) > 5000:
                logging.info(f"Large number of rules ({len(significant_rules)}). Limiting to top 5000 by lift.")
                significant_rules = significant_rules.sort_values(
                    ['lift', 'confidence'],
                    ascending=[False, False]
                ).head(5000)
            else:
                # Sort rules by lift and confidence
                significant_rules = significant_rules.sort_values(
                    ['lift', 'confidence'],
                    ascending=[False, False]
                )

            # Add support percentage for better interpretation
            significant_rules['support_pct'] = significant_rules['support'] * 100
            significant_rules['confidence_pct'] = significant_rules['confidence'] * 100

            total_time = time.time() - start_time
            logging.info(f"Generated {len(significant_rules)} significant rules in {total_time:.2f} seconds")
            return significant_rules

        except MemoryError:
            logging.error("Memory error during rule generation. Try reducing the dataset size or increasing min_support.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error in rule generation: {str(e)}", exc_info=True)
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"Unexpected error in generate_association_rules: {str(e)}", exc_info=True)
        return pd.DataFrame()

def generate_recommendations_from_rules(self, rules, df, min_lift=1.5, max_recommendations=10):
    """
    Generate recommendations based on association rules analysis using fixed rating scale.
    Only includes top recommendations based on lift and confidence for features with low ratings.
    Excludes Overall_Rating from recommendations.
    """
    recommendations = {}

    if rules.empty:
        return recommendations

    # Use fixed rating scale thresholds
    min_rating = self.RATING_SCALE['min']
    max_rating = self.RATING_SCALE['max']
    needs_improvement = self.RATING_SCALE['thresholds']['needs_improvement']

    # Calculate average ratings for each feature
    avg_ratings = df.mean()

    # Sort rules by lift and confidence for strongest associations first
    sorted_rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

    recommendation_count = 0

    for _, rule in sorted_rules.iterrows():
        if rule['lift'] >= min_lift and recommendation_count < max_recommendations:
            for antecedent in rule['antecedents']:
                # Handle features with underscores in their names
                parts = antecedent.split('_')
                # Find the rating category by checking known suffixes
                for i in range(len(parts)-1, 0, -1):
                    suffix = '_'.join(parts[i:])
                    if suffix in ['Needs_Improvement', 'Moderately_Satisfactory',
                                 'Satisfactory', 'Very_Satisfactory']:
                        feature = '_'.join(parts[:i])
                        rating = suffix
                        break
                else:  # If no rating category found
                    continue  # skip invalid format

                # Skip if the feature is Overall_Rating
                if feature == 'Overall_Rating':
                    continue

                # Skip if feature's average rating is not low
                if feature in avg_ratings and avg_ratings[feature] >= needs_improvement:
                    continue

                if feature not in recommendations:
                    recommendations[feature] = []

                for consequent in rule['consequents']:
                    # Apply the same splitting logic to consequents
                    parts = consequent.split('_')
                    for i in range(len(parts)-1, 0, -1):
                        suffix = '_'.join(parts[i:])
                        if suffix in ['Needs_Improvement', 'Moderately_Satisfactory',
                                     'Satisfactory', 'Very_Satisfactory']:
                            cons_feature = '_'.join(parts[:i])
                            cons_rating = suffix
                            break
                        else:
                            continue

                    if cons_feature == 'Overall_Rating':
                        continue

                    if rating in ['Needs_Improvement', 'Moderately_Satisfactory']:
                        recommendation = {
                            'text': f"Improve {feature} (current avg: {avg_ratings[feature]:.2f}/{max_rating:.1f}) to enhance {cons_feature}",
                            'action': f"• Implement targeted improvements in {feature} to achieve {cons_rating} {cons_feature}",
                            'support': rule['support'],
                            'confidence': rule['confidence'],
                            'lift': rule['lift']
                        }

                        if rating == 'Needs_Improvement':
                            recommendation['action'] += f"\n• Conduct immediate review of {feature.replace('_', ' ').lower()} processes"
                            recommendation['action'] += f"\n• Set up weekly monitoring of {feature} metrics"
                            recommendation['priority'] = 'High'
                        elif rating == 'Moderately_Satisfactory':
                            recommendation['action'] += f"\n• Develop enhancement plan for {feature}"
                            recommendation['action'] += f"\n• Implement monthly progress tracking"
                            recommendation['priority'] = 'Medium'

                        if recommendation not in recommendations[feature]:
                            recommendations[feature].append(recommendation)
                            recommendation_count += 1

                            if recommendation_count >= max_recommendations:
                                return recommendations

    return recommendations

def generate_event_recommendations(low_scores):
    """Generate standard recommendations based on low scores."""
    base_recommendations = {
        'Overall_Rating': [
            {
                'text': "Conduct comprehensive program review",
                'action': "Implement systematic review process",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            },
            {
                'text': "Implement regular feedback sessions",
                'action': "Schedule monthly feedback meetings",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Objectives_Met': [
            {
                'text': "Clearly communicate objectives",
                'action': "Create detailed objective documentation",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Venue_Rating': [
            {
                'text': "Consider alternative venues",
                'action': "Research and evaluate new venues",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Schedule_Rating': [
            {
                'text': "Offer more flexible scheduling options",
                'action': "Implement scheduling survey",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Speaker_Rating': [
            {
                'text': "Invite more engaging speakers",
                'action': "Create speaker evaluation process",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ]
    }
    return {k: base_recommendations.get(k, []) for k in low_scores.index if k in base_recommendations}

def generate_event_maintenance_recommendations(high_scores):
    """Generate maintenance recommendations for high-performing areas."""
    maintenance_recommendations = {
        'Overall_Rating': [
            {
                'text': "Document successful practices",
                'action': "Create best practices documentation",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Objectives_Met': [
            {
                'text': "Maintain clear documentation",
                'action': "Regular documentation review",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Venue_Rating': [
            {
                'text': "Maintain venue relationships",
                'action': "Schedule regular venue reviews",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Schedule_Rating': [
            {
                'text': "Keep consistent scheduling",
                'action': "Document scheduling process",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Speaker_Rating': [
            {
                'text': "Build speaker database",
                'action': "Create speaker tracking system",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ]
    }
    return {k: maintenance_recommendations.get(k, []) for k in high_scores.index if k in maintenance_recommendations}

def interpret_event_association_rules(rules):
    """Generate human-readable interpretations for the association rules with fixed rating scale"""
    if rules.empty:
        return "No significant associations found between event features."

    interpretations = ["Association Rules Analysis:\n"]

    for _, rule in rules.head(15).iterrows():
        # Extract feature and rating from antecedents and consequents
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])

        # Format antecedents and consequents
        ant_str = []
        for ant in antecedents:
            feature, rating = ant.rsplit('_', 1)
            rating = rating.replace('_', ' ')  # Convert Needs_Improvement to "Needs Improvement"
            ant_str.append(f"'{feature.replace('_', ' ')}' is {rating}")

        cons_str = []
        for cons in consequents:
            feature, rating = cons.rsplit('_', 1)
            rating = rating.replace('_', ' ')  # Convert Needs_Improvement to "Needs Improvement"
            cons_str.append(f"'{feature.replace('_', ' ')}' is {rating}")

        # Create rule interpretation
        rule_str = f"Rule: If {' AND '.join(ant_str)}, then {' AND '.join(cons_str)}\n"
        rule_str += f"(Support: {rule['support']:.2f}, Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f})\n"

        interpretations.append(rule_str)

    return "\n".join(interpretations)

from functools import lru_cache

@lru_cache(maxsize=32)
def find_optimal_clusters(data_hash, max_clusters=10):
    """
    Find the optimal number of clusters using the elbow method.

    Parameters:
    - data_hash (bytes): The data to cluster in hashable format.
    - max_clusters (int): The maximum number of clusters to test.

    Returns:
    - elbow (int): The optimal number of clusters.
    """
    # Convert data to hashable format for caching
    data = np.frombuffer(data_hash)

    wcss = []
    K = range(1, max_clusters + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_curve.png')
    plt.close()

    # Use KneeLocator to automatically find the elbow point
    kn = KneeLocator(
        K, wcss, curve='convex', direction='decreasing'
    )

    return kn.elbow

class Visualizer:
    """Separate visualization logic"""
    def __init__(self, root):
        self.root = root

    def create_plot(self, plot_type, data, **kwargs):
        plot_methods = {
            'cluster': self._plot_clusters,
            'histogram': self._plot_histograms,
            'rules': self._plot_association_rules
        }
        return plot_methods[plot_type](data, **kwargs)

class AnalysisGUI:
    def __init__(self, root):
        """Initialize the GUI"""
        try:
            # Initialize data storage first
            self.datasets = {}  # Dictionary to store datasets by year
            self.df = None  # Combined dataframe
            self.selected_features = []  # Selected features for analysis
            self.departments = []  # Store available departments
            self.current_department = tk.StringVar(value="All Departments")  # Store current department selection

            # Add fixed rating scale constants
            self.RATING_SCALE = {
                'min': 0.0,
                'max': 3.0,
                'thresholds': {
                    'needs_improvement': 0.74,
                    'moderately_satisfactory': 1.49,
                    'satisfactory': 2.24,
                    'very_satisfactory': 3.00
                }
            }

            self.root = root
            self.root.title("Event Analysis Tool")

            # Make the window fullscreen
            self.root.state('zoomed')  # For Windows
            # For Linux/Mac, use:
            # self.root.attributes('-zoomed', True)

            # Set minimum window size
            self.root.minsize(1024, 768)

            # Create navigation bar
            self.create_navigation()

            # Create department filter frame
            self.create_department_filter()

            # Create notebook for tabs
            self.tab_control = ttk.Notebook(self.root)
            self.tab_control.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

            # Initialize tabs
            self.output_tab = ttk.Frame(self.tab_control)
            self.cluster_tab = ttk.Frame(self.tab_control)
            self.rules_tab = ttk.Frame(self.tab_control)
            self.descriptive_tab = ttk.Frame(self.tab_control)
            self.histogram_tab = ttk.Frame(self.tab_control)
            # self.distribution_tab = ttk.Frame(self.tab_control)  # Removed distribution tab
            self.recommendations_tab = ttk.Frame(self.tab_control)
            self.baseline_tab = ttk.Frame(self.tab_control)
            self.cluster_trends_tab = ttk.Frame(self.tab_control)  # Add new tab for cluster trends per year

            # Add tabs to notebook
            self.tab_control.add(self.output_tab, text='Analysis Results')
            self.tab_control.add(self.cluster_tab, text='Clustering')
            self.tab_control.add(self.rules_tab, text='Association Rules')
            self.tab_control.add(self.descriptive_tab, text='Descriptive Stats')
            self.tab_control.add(self.histogram_tab, text='Histograms')
            # self.tab_control.add(self.distribution_tab, text='Distribution')  # Removed distribution tab
            self.tab_control.add(self.recommendations_tab, text='Recommendations')
            self.tab_control.add(self.baseline_tab, text='Baseline Comparisons')
            self.tab_control.add(self.cluster_trends_tab, text='Cluster Trends Per Year')  # Add new tab

            # Create scrolled text widgets for output and recommendations tabs
            self.output_text = scrolledtext.ScrolledText(self.output_tab, height=30)
            self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            self.recommendations_text = scrolledtext.ScrolledText(self.recommendations_tab, height=30)
            self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            print("AnalysisGUI initialized successfully.")

        except Exception as e:
            print(f"Error in AnalysisGUI.__init__: {e}")
            raise  # Re-raise the exception to see the full traceback

    def create_department_filter(self):
        """Create the department filter dropdown"""
        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(fill=tk.X, padx=5, pady=2)

        # Create label
        ttk.Label(filter_frame, text="Filter by Department:").pack(side=tk.LEFT, padx=5)

        # Create department dropdown
        self.department_dropdown = ttk.Combobox(filter_frame, textvariable=self.current_department)
        self.department_dropdown.pack(side=tk.LEFT, padx=5)

        # Bind department change event
        self.department_dropdown.bind('<<ComboboxSelected>>', self.on_department_change)

    def run_analysis(self):
        """Run the analysis with the selected parameters"""
        try:
            # Clear previous results
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Starting analysis...\n")
            self.output_text.update()

            # Validate that we have data
            if not self.datasets:
                self.output_text.insert(tk.END, "No datasets loaded. Please load data first.\n")
                return

            # Validate that we have selected features
            if not hasattr(self, 'selected_features') or not self.selected_features:
                self.output_text.insert(tk.END, "No features selected. Please select features first.\n")
                return

            # Get filtered data based on department selection
            filtered_df = self.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                self.output_text.insert(tk.END, "No data available after filtering.\n")
                return

            # 1. Generate recommendations
            self.output_text.insert(tk.END, "\nGenerating recommendations...\n")
            self.output_text.update()

            try:
                self.plot_recommendations()
            except Exception as e:
                self.output_text.insert(tk.END, f"Error generating recommendations: {str(e)}\n")
                logging.error(f"Recommendations error: {str(e)}")

            # 2. Create descriptive statistics visualizations
            try:
                self.plot_descriptive()
            except Exception as e:
                self.output_text.insert(tk.END, f"Error during descriptive analysis: {str(e)}\n")
                logging.error(f"Descriptive analysis error: {str(e)}")

            # 3. Create histograms
            try:
                self.plot_histograms()
            except Exception as e:
                self.output_text.insert(tk.END, f"Error creating histograms: {str(e)}\n")
                logging.error(f"Histogram error: {str(e)}")

            # 4. Perform clustering analysis
            self.output_text.insert(tk.END, "Performing clustering analysis...\n")
            self.output_text.update()

            try:
                clustered_df, kmeans, cluster_sizes = self.cluster_events(filtered_df, self.selected_features)
                self.plot_clusters(clustered_df, kmeans, "Cluster Analysis")
            except Exception as e:
                self.output_text.insert(tk.END, f"Error during clustering: {str(e)}\n")
                logging.error(f"Clustering error: {str(e)}")

            # 5. Generate association rules
            self.output_text.insert(tk.END, "Generating association rules...\n")
            self.output_text.update()

            try:
                # Prepare data for association rules
                binary_df = prepare_for_association_rules(filtered_df, self.selected_features)

                # Set parameters for association rule mining
                self.min_support = 0.05
                self.min_confidence = 0.3
                self.min_lift = 1.0
                self.max_len = 3

                # Generate association rules
                rules = generate_association_rules(
                    binary_df,
                    min_support=self.min_support,
                    min_confidence=self.min_confidence,
                    min_lift=self.min_lift,
                    max_len=self.max_len,
                    use_parallel=True
                )

                # If no rules found, try with relaxed parameters
                if len(rules) == 0:
                    self.output_text.insert(tk.END, "No rules found with initial parameters. Trying with relaxed parameters...\n")
                    self.output_text.update()

                    self.min_support = 0.03
                    self.min_confidence = 0.2

                    rules = generate_association_rules(
                        binary_df,
                        min_support=self.min_support,
                        min_confidence=self.min_confidence,
                        min_lift=self.min_lift,
                        max_len=self.max_len,
                        use_parallel=True
                    )

                # Plot association rules
                if len(rules) > 0:
                    self.output_text.insert(tk.END, f"Found {len(rules)} association rules.\n")
                    self.plot_association_rules(rules, "All Data")
                else:
                    self.output_text.insert(tk.END, "No significant association rules found.\n")
            except Exception as e:
                self.output_text.insert(tk.END, f"Error during association rule mining: {str(e)}\n")
                logging.error(f"Association rules error: {str(e)}")

            # 6. Calculate baseline metrics if we have multiple years
            if len(self.datasets) > 1:
                self.output_text.insert(tk.END, "Calculating baseline metrics...\n")
                self.output_text.update()

                try:
                    baseline_metrics = self.calculate_baseline_metrics()
                    if baseline_metrics:
                        # Get the most recent year's data
                        latest_year = max(self.datasets.keys())
                        latest_data = self.datasets[latest_year]

                        # Filter by department if needed
                        if self.current_department.get() != "All Departments":
                            latest_data = latest_data[latest_data['department_name'] == self.current_department.get()]

                        # Compare with baseline
                        comparison_data = self.compare_with_baseline(latest_data, baseline_metrics)

                        # Plot baseline comparison
                        self.plot_baseline_comparison(comparison_data)
                except Exception as e:
                    self.output_text.insert(tk.END, f"Error during baseline comparison: {str(e)}\n")
                    logging.error(f"Baseline comparison error: {str(e)}")

            # 7. If we have multiple years, plot cluster trends
            if len(self.datasets) > 1:
                self.output_text.insert(tk.END, "Generating cluster trends...\n")
                self.output_text.update()

                try:
                    self.plot_cluster_trends_per_year()
                except Exception as e:
                    self.output_text.insert(tk.END, f"Error during cluster trend analysis: {str(e)}\n")
                    logging.error(f"Cluster trend error: {str(e)}")

            # Analysis complete
            self.output_text.insert(tk.END, "Analysis complete!\n")
            self.output_text.see(tk.END)  # Scroll to the end

            # Switch to the results tab
            self.tab_control.select(self.output_tab)

        except Exception as e:
            self.output_text.insert(tk.END, f"Error during analysis: {str(e)}\n")
            logging.error(f"Analysis error: {str(e)}")

    def on_department_change(self, event=None):
        """Handle department selection change"""
        if hasattr(self, 'df') and self.df is not None:
            self.run_analysis()  # Changed from update_analysis to run_analysis

    def update_department_list(self):
        """Update the list of departments in the dropdown"""
        if hasattr(self, 'df') and self.df is not None:
            # Get unique departments
            departments = sorted(self.df['department_name'].unique())
            # Update dropdown values
            self.department_dropdown['values'] = ['All Departments'] + list(departments)
            # Reset selection to "All Departments"
            self.current_department.set("All Departments")

    def get_filtered_data(self, year=None):
        """Get data filtered by selected department and optionally by year

        Args:
            year (str, optional): The year to filter data for. If None, uses the currently loaded dataset.

        Returns:
            pandas.DataFrame: The filtered dataset
        """
        # If year is specified, use that dataset, otherwise use the current df
        if year is not None and year in self.datasets:
            df_to_filter = self.datasets[year]
        else:
            df_to_filter = self.df

        if self.current_department.get() == "All Departments":
            return df_to_filter
        else:
            # Keep department_name as is, don't try to convert it to numeric
            filtered_df = df_to_filter[df_to_filter['department_name'] == self.current_department.get()].copy()

            # Convert only numeric columns, excluding department_name
            numeric_cols = [col for col in filtered_df.columns if col != 'department_name']
            for col in numeric_cols:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

            return filtered_df

    def create_navigation(self):
        """Create a navigation bar at the top of the window"""
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create left frame for main controls
        left_frame = ttk.Frame(nav_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X)

        # Create right frame for thresholds
        right_frame = ttk.Frame(nav_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.X)

        # Main control buttons (left side)
        load_btn = ttk.Button(
            left_frame,
            text="Load Data",
            command=self.load_data,
            width=15
        )
        load_btn.pack(side=tk.LEFT, padx=2)

        clear_btn = ttk.Button(
            left_frame,
            text="Clear Data",
            command=self.clear_datasets,
            width=15
        )
        clear_btn.pack(side=tk.LEFT, padx=2)

        select_features_btn = ttk.Button(
            left_frame,
            text="Select Features",
            command=self.select_features_window,
            width=15
        )
        select_features_btn.pack(side=tk.LEFT, padx=2)

        analyze_btn = ttk.Button(
            left_frame,
            text="Run Analysis",
            command=self.run_analysis,
            width=15
        )
        analyze_btn.pack(side=tk.LEFT, padx=2)

        # Add separator below navigation
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=5, pady=5)

    def clear_datasets(self):
        """Clear all loaded datasets and reset the GUI"""
        try:
            # Confirm with user
            if messagebox.askyesno("Clear Data", "Are you sure you want to clear all loaded datasets?"):
                # Clear datasets
                self.datasets.clear()
                if hasattr(self, 'df'):
                    del self.df

                # Clear selected features
                self.selected_features.clear()

                # Clear text outputs
                self.output_text.delete(1.0, tk.END)
                self.recommendations_text.delete(1.0, tk.END)

                # Clear all visualization tabs
                for tab in [self.cluster_tab,
                           self.histogram_tab, self.rules_tab]:  # Removed distribution_tab
                    for widget in tab.winfo_children():
                        widget.destroy()

                # Reset progress
                if hasattr(self, 'progress'):
                    self.progress['value'] = 0

                # Update GUI to show cleared state
                self.output_text.insert(tk.END, "All datasets have been cleared.\n")
                self.output_text.insert(tk.END, "Use 'Load Data' to load new datasets.\n")

                # Clear memory
                gc.collect()

                messagebox.showinfo("Success", "All datasets have been cleared successfully.")
        except Exception as e:
            logging.error(f"Error clearing datasets: {e}")
            messagebox.showerror("Error", f"Error clearing datasets: {str(e)}")

    def create_visualization_area(self):
        # Visualization area
        viz_frame = ttk.LabelFrame(self.right_frame, text="Visualizations", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Create tabs for different plots
        self.tab_control = ttk.Notebook(viz_frame)

        self.descriptive_tab = ttk.Frame(self.tab_control)
        self.cluster_tab = ttk.Frame(self.tab_control)
        # self.distribution_tab = ttk.Frame(self.tab_control)  # Removed distribution tab
        self.histogram_tab = ttk.Frame(self.tab_control)
        self.rules_tab = ttk.Frame(self.tab_control)
        self.output_tab = ttk.Frame(self.tab_control)
        self.recommendations_tab = ttk.Frame(self.tab_control)  # Add recommendations tab

        self.tab_control.add(self.descriptive_tab, text='Descriptive Analysis')
        self.tab_control.add(self.cluster_tab, text='Clustering')
        # self.tab_control.add(self.distribution_tab, text='Distribution')  # Removed distribution tab
        self.tab_control.add(self.histogram_tab, text='Histograms')
        self.tab_control.add(self.rules_tab, text='Association Rules')
        self.tab_control.add(self.output_tab, text='Analysis Results')
        self.tab_control.add(self.recommendations_tab, text='Recommendations')  # Add recommendations tab

        self.tab_control.pack(fill=tk.BOTH, expand=True)

        # Create scrolled text widgets for both analysis and recommendations tabs
        self.output_text = scrolledtext.ScrolledText(self.output_tab, wrap=tk.WORD, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.recommendations_text = scrolledtext.ScrolledText(self.recommendations_tab, wrap=tk.WORD, height=20)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        print("Starting load_data method")  # Debug print
        try:
            files = filedialog.askopenfilenames(
                title="Select Data Files",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            print(f"Selected files: {files}")  # Debug print

            if not files:
                print("No files selected")  # Debug print
                return

            # Track if any new datasets were loaded
            new_datasets_loaded = False

            for file_path in files:
                # Prompt user to enter the year for the current dataset
                year = self.prompt_for_year(file_path)
                if year is None:
                    print(f"Year not provided for {file_path}. Skipping this file.")
                    continue

                # Check if this year already exists in datasets
                if year in self.datasets:
                    overwrite = messagebox.askyesno("Year Exists",
                                                   f"Data for year {year} already exists. Do you want to overwrite it?")
                    if not overwrite:
                        continue

                # Load the data using pandas read_csv
                print(f"Loading data from {file_path}...")  # Debug print
                df = pd.read_csv(file_path)

                # Ensure department_name column exists
                if 'department_name' not in df.columns:
                    messagebox.showerror("Error", "The dataset must contain a 'department_name' column.")
                    continue

                # Store department_name column separately
                dept_names = df['department_name'].copy()

                # Create a new dataframe without department_name for numeric processing
                numeric_df = df.drop('department_name', axis=1)

                # Process each numeric column separately
                for col in numeric_df.columns:
                    try:
                        # Remove any leading/trailing whitespace
                        if numeric_df[col].dtype == object:
                            numeric_df[col] = numeric_df[col].str.strip()

                        # Convert to numeric, forcing errors to NaN
                        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

                        # Fill NaN values with mean of the column
                        mean_val = numeric_df[col].mean()
                        numeric_df[col].fillna(mean_val, inplace=True)

                        print(f"Successfully converted {col} to numeric")  # Debug print
                    except Exception as e:
                        print(f"Error converting column {col}: {str(e)}")
                        # If conversion fails completely, drop the column
                        numeric_df = numeric_df.drop(col, axis=1)
                        continue

                # Add department_name back to the processed dataframe
                numeric_df['department_name'] = dept_names

                print(f"Data loaded from {file_path}. Shape: {numeric_df.shape}")  # Debug print
                print(f"Columns: {numeric_df.columns.tolist()}")  # Debug print
                print(f"Column types: {numeric_df.dtypes}")  # Debug print

                # Check if data was loaded successfully
                if numeric_df.empty:
                    print(f"DataFrame from {file_path} is empty")  # Debug print
                    messagebox.showerror("Error", f"No data was loaded from the file: {file_path}")
                    continue

                # Store the dataset with the associated year
                self.datasets[year] = numeric_df
                print(f"Dataset for year {year} added successfully.")  # Debug print
                new_datasets_loaded = True

                # Show success message
                self.output_text.insert(tk.END, f"Data loaded successfully for year {year}: {len(numeric_df)} records\n")
                self.output_text.insert(tk.END, f"Columns found: {', '.join(numeric_df.columns)}\n\n")

            # Sort datasets by year
            self.sort_datasets_by_year()

            # Display summary of loaded datasets
            if self.datasets:
                self.output_text.insert(tk.END, "\nSummary of loaded datasets:\n")
                for year, dataset in self.datasets.items():
                    self.output_text.insert(tk.END, f"Year {year}: {len(dataset)} records, {len(dataset.columns)} columns\n")
                self.output_text.insert(tk.END, "\n")

            # After loading all datasets, concatenate them into self.df
            if self.datasets:
                self.df = pd.concat(self.datasets.values(), ignore_index=True)
                print(f"Combined dataset shape: {self.df.shape}")  # Debug print

                # Update department list in dropdown
                self.update_department_list()

                # Automatically prompt feature selection if new datasets were loaded
                if new_datasets_loaded:
                    self.select_features_window()

                print("Data loading process completed.")  # Debug print

        except Exception as e:
            print(f"Error in load_data: {str(e)}")  # Debug print
            messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def prompt_for_year(self, file_path):
        """
        Prompts the user to enter the year for the given dataset.

        Parameters:
        - file_path (str): The path of the file being loaded.

        Returns:
        - year (str): The year entered by the user.
        """
        try:
            year_window = tk.Toplevel(self.root)
            year_window.title("Enter Year for Dataset")
            year_window.geometry("300x100")

            ttk.Label(year_window, text=f"Enter year for {file_path}:").pack(pady=10)
            year_var = tk.StringVar()
            year_entry = ttk.Entry(year_window, textvariable=year_var)
            year_entry.pack(pady=5)
            year_entry.focus_set()

            def submit_year():
                year = year_var.get().strip()
                if not year.isdigit():
                    messagebox.showerror("Invalid Input", "Please enter a valid year (numeric).")
                    return
                year_window.destroy()
                year_window.year = year  # Attach year to the window object

            submit_btn = ttk.Button(year_window, text="Submit", command=submit_year)
            submit_btn.pack(pady=5)

            self.root.wait_window(year_window)

            return getattr(year_window, 'year', None)
        except Exception as e:
            print(f"Error in prompt_for_year: {e}")  # Debug print
            return None

    def sort_datasets_by_year(self):
        """
        Sorts the datasets stored in self.datasets by year in ascending order.
        """
        try:
            sorted_years = sorted(self.datasets.keys(), key=lambda x: int(x))
            sorted_datasets = {year: self.datasets[year] for year in sorted_years}
            self.datasets = sorted_datasets
            print("Datasets sorted by year:", list(self.datasets.keys()))  # Debug print
        except Exception as e:
            print(f"Error in sort_datasets_by_year: {e}")  # Debug print
            messagebox.showerror("Error", f"Error sorting datasets by year: {str(e)}")

    def select_features_window(self):
        print("Starting select_features_window method")  # Debug print
        try:
            feature_window = tk.Toplevel(self.root)
            feature_window.title("Select Features")
            feature_window.geometry("400x500")

            # Make window modal
            feature_window.transient(self.root)
            feature_window.grab_set()

            print("Getting columns")  # Debug print
            # Get numeric columns (excluding department_name)
            numeric_columns = []
            for col in self.df.columns:
                if col == 'department_name':  # Skip department_name column
                    continue

                if np.issubdtype(self.df[col].dtype, np.number):
                    numeric_columns.append(col)
                    print(f"Column {col} is numeric")
                else:
                    # Try converting to numeric
                    try:
                        test_series = pd.to_numeric(self.df[col], errors='coerce')
                        if test_series.notna().any():  # Check if any values can be converted
                            numeric_columns.append(col)
                            print(f"Column {col} can be converted to numeric")
                    except:
                        print(f"Column {col} is not numeric")
                        continue

            print(f"Numeric columns found: {numeric_columns}")  # Debug print

            if not numeric_columns:
                messagebox.showerror("Error", "No numeric columns found in the data.")
                feature_window.destroy()
                return

            # Create frames for column type selection
            selection_frame = ttk.Frame(feature_window)
            selection_frame.pack(fill=tk.X, padx=10, pady=5)

            instructions = ttk.Label(
                feature_window,
                text="Select numeric features to include in the analysis\n(Hold Ctrl/Cmd to select multiple)",
                wraplength=350
            )
            instructions.pack(pady=10)

            # Frame for listbox and scrollbar
            list_frame = ttk.Frame(feature_window)
            list_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Create a listbox with multiple selection enabled
            self.features_listbox = tk.Listbox(
                list_frame,
                selectmode=tk.MULTIPLE,
                height=15,
                yscrollcommand=scrollbar.set
            )
            self.features_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Configure scrollbar
            scrollbar.config(command=self.features_listbox.yview)

            # Insert numeric columns into listbox
            for col in numeric_columns:
                self.features_listbox.insert(tk.END, col)

            # Add buttons frame
            button_frame = ttk.Frame(feature_window)
            button_frame.pack(pady=10)

            # Add Select All button
            select_all_btn = ttk.Button(
                button_frame,
                text="Select All",
                command=lambda: self.features_listbox.select_set(0, tk.END)
            )
            select_all_btn.pack(side=tk.LEFT, padx=5)

            # Add Clear All button
            clear_all_btn = ttk.Button(
                button_frame,
                text="Clear All",
                command=lambda: self.features_listbox.selection_clear(0, tk.END)
            )
            clear_all_btn.pack(side=tk.LEFT, padx=5)

            # Add OK button
            ok_btn = ttk.Button(
                button_frame,
                text="OK",
                command=lambda: self.save_feature_selection(feature_window)
            )
            ok_btn.pack(side=tk.LEFT, padx=5)

            # Add Cancel button
            cancel_btn = ttk.Button(
                button_frame,
                text="Cancel",
                command=feature_window.destroy
            )
            cancel_btn.pack(side=tk.LEFT, padx=5)

        except Exception as e:
            print(f"Error in select_features_window: {str(e)}")
            messagebox.showerror("Error", f"Error creating feature selection window: {str(e)}")
            if feature_window:
                feature_window.destroy()

    def save_feature_selection(self, feature_window):
        """
        Stores the selected features for analysis and closes the selection window.
        """
        selected_indices = self.features_listbox.curselection()
        if not selected_indices:
            tk.messagebox.showerror(
                "Error",
                "Please select at least one feature for analysis."
            )
            return

        # Store selected features as a list
        self.selected_features = [self.features_listbox.get(i) for i in selected_indices]

        # Update output text
        self.output_text.insert(tk.END, "Selected features for analysis:\n")
        for feature in self.selected_features:
            self.output_text.insert(tk.END, f"- {feature}\n")
        self.output_text.insert(tk.END, "\n")

        # Close the feature selection window
        feature_window.destroy()

    def plot_clusters(self, df, kmeans, title, use_pca=True):
        """
        Plots the K-means clusters using PCA for dimensionality reduction or in original feature space.

        Args:
            df: DataFrame with cluster assignments
            kmeans: Fitted KMeans model
            title: Plot title
            use_pca: Whether to use PCA (True) or plot in original feature space (False)
        """
        if not self.selected_features:
            self.output_text.insert(tk.END, "Error: No features selected for clustering.\n")
            return

        # Get numeric features only
        numeric_features = [f for f in self.selected_features if f != 'department_name']
        if not numeric_features:
            self.output_text.insert(tk.END, "Error: No numeric features available for clustering.\n")
            return

        try:
            # Clear previous content in cluster tab
            for widget in self.cluster_tab.winfo_children():
                widget.destroy()

            # Create a canvas with scrollbar for cluster tab
            canvas = tk.Canvas(self.cluster_tab)
            scrollbar = ttk.Scrollbar(self.cluster_tab, orient="vertical", command=canvas.yview)

            # Create main frame inside canvas
            main_frame = ttk.Frame(canvas)

            # Configure the canvas
            canvas.configure(yscrollcommand=scrollbar.set)

            # Pack scrollbar and canvas
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            # Create window in canvas
            canvas.create_window((0, 0), window=main_frame, anchor="nw")

            # If multiple datasets are loaded, add a dropdown to select which year to view
            if len(self.datasets) > 1:
                year_frame = ttk.Frame(main_frame)
                year_frame.pack(fill=tk.X, padx=10, pady=5)

                ttk.Label(year_frame, text="Select Year:").pack(side=tk.LEFT, padx=5)

                year_var = tk.StringVar(value="Combined")
                year_options = ["Combined"] + sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly")
                year_dropdown.pack(side=tk.LEFT, padx=5)

                def on_year_change(event=None):
                    selected_year = year_var.get()
                    if selected_year == "Combined":
                        df_to_analyze = self.get_filtered_data()
                    else:
                        df_to_analyze = self.datasets[selected_year]
                        if self.current_department.get() != "All Departments":
                            df_to_analyze = df_to_analyze[df_to_analyze['department_name'] == self.current_department.get()]

                    # Perform clustering on the selected year's data
                    clustered_df, kmeans_model, cluster_sizes, cluster_labels = self.cluster_events(
                        df_to_analyze, self.selected_features, return_labels=True)

                    # Clear the main frame and recreate the plot
                    for widget in main_frame.winfo_children():
                        widget.destroy()

                    # Recreate the year selection dropdown
                    year_frame = ttk.Frame(main_frame)
                    year_frame.pack(fill=tk.X, padx=10, pady=5)
                    ttk.Label(year_frame, text="Select Year:").pack(side=tk.LEFT, padx=5)
                    year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly")
                    year_dropdown.pack(side=tk.LEFT, padx=5)
                    year_dropdown.bind("<<ComboboxSelected>>", on_year_change)

                    # Create the plot with the new data
                    self.create_cluster_plot(clustered_df, kmeans_model, cluster_labels, selected_year, year_frame, use_pca)

                year_dropdown.bind("<<ComboboxSelected>>", on_year_change)
            else:
                # Create year frame
                year_frame = ttk.Frame(main_frame)
                year_frame.pack(fill=tk.X, padx=5, pady=5)

                # Add year label
                year_label = ttk.Label(year_frame, text=f"Year {list(self.datasets.keys())[0]}", font=('Arial', 12, 'bold'))
                year_label.pack(pady=5)

            # Create the initial cluster plot
            cluster_labels = df['cluster_label'].tolist() if 'cluster_label' in df.columns else None
            self.create_cluster_plot(df, kmeans, cluster_labels, "Combined", year_frame, use_pca)

            # Configure scroll region
            main_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

            # Function to handle resize events
            def on_cluster_resize(event):
                # Only redraw if the event is for our window and the size actually changed
                if event.widget == self.root and (event.width != getattr(self, '_last_width_cluster', 0) or
                                                 event.height != getattr(self, '_last_height_cluster', 0)):
                    self._last_width_cluster = event.width
                    self._last_height_cluster = event.height

                    # Get the available width for plots
                    available_width = max(canvas.winfo_width() - 20, 100)  # Subtract padding and ensure minimum width

                    # Resize the plot with minimum size check
                    if hasattr(self, 'cluster_fig') and hasattr(self, 'cluster_canvas'):
                        width_inches = max(available_width/100, 1.0)  # Ensure minimum width of 1 inch
                        height_inches = max((available_width*0.75)/100, 0.75)  # Ensure minimum height

                        self.cluster_fig.set_size_inches(width_inches, height_inches)
                        self.cluster_fig.tight_layout()
                        self.cluster_canvas.draw_idle()

                        # Update scroll region
                        main_frame.update_idletasks()
                        canvas.configure(scrollregion=canvas.bbox("all"))

            # Bind the resize event
            self.root.bind("<Configure>", on_cluster_resize)

            # Trigger an initial resize
            self.root.update_idletasks()
            on_cluster_resize(type('Event', (), {'widget': self.root, 'width': self.root.winfo_width(),
                                              'height': self.root.winfo_height()})())

            # Configure scroll region when frame changes
            main_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            self.output_text.insert(tk.END, f"Error during plotting: {str(e)}\n")
            logging.error(f"Error during plotting: {str(e)}")
            return

    def create_cluster_plot(self, df, kmeans, labels, year, parent_frame, use_pca=True):
        """
        Creates a cluster plot within the given parent frame.

        Args:
            df: DataFrame with data
            kmeans: KMeans model
            labels: Cluster labels
            year: Year string for title
            parent_frame: Parent frame to place the plot in
            use_pca: Whether to use PCA for dimensionality reduction
        """
        try:
            # Get numeric features only
            numeric_features = [f for f in self.selected_features if f != 'department_name']

            # Get the feature data
            feature_data = df[numeric_features].copy()

            # Handle outliers by capping extreme values
            for col in feature_data.columns:
                q1 = feature_data[col].quantile(0.05)
                q3 = feature_data[col].quantile(0.95)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                feature_data[col] = feature_data[col].clip(lower_bound, upper_bound)

            # Create figure with dynamic sizing
            self.cluster_fig = plt.Figure(figsize=(12, 8), dpi=100)
            ax = self.cluster_fig.add_subplot(111)

            if kmeans is None or labels is None:
                # If kmeans is None, create a simple bar plot of categories
                categories = ['Needs Improvement', 'Moderately Satisfactory',
                            'Satisfactory', 'Very Satisfactory']
                counts = pd.Series(labels).value_counts()
                percentages = (counts / len(labels) * 100).reindex(categories).fillna(0)

                colors = [self.get_cluster_color(cat) for cat in categories]
                bars = ax.bar(categories, percentages, color=colors)

                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom')

                if year == "Combined":
                    ax.set_title('Rating Distribution - Combined Data')
                else:
                    ax.set_title(f'Rating Distribution - Year {year}')
                ax.set_ylabel('Percentage of Responses')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                # First standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_data)

                if use_pca:
                    # Create PCA for dimensionality reduction
                    pca = PCA(n_components=2)
                    plot_data = pca.fit_transform(scaled_data)

                    # Get explained variance
                    explained_variance = pca.explained_variance_ratio_

                    # Set axis labels
                    ax.set_xlabel(f'First Principal Component\n({explained_variance[0]:.1%} variance)')
                    ax.set_ylabel(f'Second Principal Component\n({explained_variance[1]:.1%} variance)')

                    # Transform centers to PCA space
                    if kmeans is not None:
                        centers = pca.transform(kmeans.cluster_centers_)

                    title_suffix = "PCA Space"
                else:
                    # Use the first two features for plotting in original space
                    # Choose the two most important features based on variance
                    feature_variance = feature_data.var().sort_values(ascending=False)
                    top_features = feature_variance.index[:2].tolist()

                    if len(top_features) < 2:
                        # Fallback if we don't have enough features
                        top_features = feature_data.columns[:2].tolist()

                    # Get indices of the selected features
                    feature_indices = [list(feature_data.columns).index(f) for f in top_features]

                    # Extract data for the two selected features
                    plot_data = scaled_data[:, feature_indices]

                    # Set axis labels
                    ax.set_xlabel(top_features[0])
                    ax.set_ylabel(top_features[1])

                    # Use original centers
                    if kmeans is not None:
                        centers = kmeans.cluster_centers_[:, feature_indices]

                    title_suffix = "Original Feature Space"

                # Create scatter plot with proper scaling
                scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1],
                                   c=[self.get_cluster_color(label) for label in labels],
                                   alpha=0.6)

                # Plot cluster centers
                if kmeans is not None:
                    ax.scatter(centers[:, 0], centers[:, 1],
                              c='black', marker='X', s=200, label='Cluster Centers')

                # Add title
                if year == "Combined":
                    ax.set_title(f'Cluster Distribution - Combined Data ({title_suffix})')
                else:
                    ax.set_title(f'Cluster Distribution - Year {year} ({title_suffix})')

                # Add legend
                unique_labels = sorted(set(labels))
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=self.get_cluster_color(label),
                                  label=label, markersize=10)
                                  for label in unique_labels]

                # Add cluster center to legend if kmeans is not None
                if kmeans is not None:
                    legend_elements.append(plt.Line2D([0], [0], marker='X', color='black',
                                          label='Cluster Centers', markersize=10))

                ax.legend(handles=legend_elements)

                # Add grid
                ax.grid(True, alpha=0.3)

            # Create canvas for the plot with responsive sizing
            self.cluster_canvas = FigureCanvasTkAgg(self.cluster_fig, master=parent_frame)
            self.cluster_canvas.draw()
            plot_widget = self.cluster_canvas.get_tk_widget()
            plot_widget.pack(fill=tk.BOTH, expand=True)

            # Remove toolbar code
            # No toolbar for cleaner UI

        except Exception as e:
            logging.error(f"Error creating cluster plot: {str(e)}")
            raise

    def plot_association_rules(self, rules, title_suffix):
        """Plot association rules visualization"""
        try:
            # Clear previous content in rules tab
            for widget in self.rules_tab.winfo_children():
                widget.destroy()

            # Create a canvas with scrollbar for rules tab
            canvas = tk.Canvas(self.rules_tab)
            scrollbar = ttk.Scrollbar(self.rules_tab, orient="vertical", command=canvas.yview)

            # Create a frame inside the canvas for the plots
            plot_frame = ttk.Frame(canvas)

            # Configure scrolling
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create window in canvas
            canvas.create_window((0, 0), window=plot_frame, anchor="nw")

            # Configure scroll region when plot frame changes
            plot_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

            # Store plot references for resizing
            self.rules_plots = []

            # Create a header with dataset information
            header_frame = ttk.Frame(plot_frame)
            header_frame.pack(fill=tk.X, padx=10, pady=10)

            if len(self.datasets) > 1:
                header_text = f"Association Rules - {title_suffix} - Combined Data from {len(self.datasets)} Years ({', '.join(sorted(self.datasets.keys()))})"
            else:
                year = next(iter(self.datasets.keys()))
                header_text = f"Association Rules - {title_suffix} - Year {year}"

            header_label = ttk.Label(header_frame, text=header_text, font=("Arial", 14, "bold"))
            header_label.pack(pady=5)

            # If multiple datasets are loaded, add a dropdown to select which year to view
            if len(self.datasets) > 1:
                year_frame = ttk.Frame(plot_frame)
                year_frame.pack(fill=tk.X, padx=10, pady=5)

                ttk.Label(year_frame, text="Select Year:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

                year_var = tk.StringVar(value="Combined")
                year_options = ["Combined"] + sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly", width=15, font=("Arial", 12))
                year_dropdown.pack(side=tk.LEFT, padx=5)
                year_dropdown.set("Combined")  # Default to combined view

                # Create a container frame for the rules content
                content_frame = ttk.Frame(plot_frame)
                content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                def on_year_change(event=None):
                    # Clear existing content
                    for widget in content_frame.winfo_children():
                        widget.destroy()

                    # Get data for selected year
                    selected_year = year_var.get()

                    if selected_year == "Combined":
                        # Use the combined rules that were passed to this method
                        rules_to_display = rules
                        year_title = "Combined Data"
                    else:
                        # Generate rules for the specific year
                    df_to_analyze = self.datasets[selected_year]
                    if self.current_department.get() != "All Departments":
                        df_to_analyze = df_to_analyze[df_to_analyze['department_name'] == self.current_department.get()]

                        # Prepare data for association rules
                        try:
                            binary_df = prepare_for_association_rules(df_to_analyze, self.selected_features)

                            # Generate association rules with current parameters
                            year_rules = generate_association_rules(
                                binary_df,
                                min_support=self.min_support,
                                min_confidence=self.min_confidence,
                                min_lift=self.min_lift,
                                max_len=self.max_len,
                                use_parallel=True
                            )

                            # If no rules found, try with relaxed parameters
                            if len(year_rules) == 0:
                            def try_relaxed_parameters():
                                    relaxed_support = max(0.01, self.min_support * 0.5)
                                    relaxed_confidence = max(0.1, self.min_confidence * 0.5)

                                    logging.info(f"No rules found with original parameters. Trying relaxed parameters: "
                                                f"support={relaxed_support}, confidence={relaxed_confidence}")

                                    return generate_association_rules(
                                    binary_df,
                                        min_support=relaxed_support,
                                        min_confidence=relaxed_confidence,
                                        min_lift=1.0,  # Keep minimum lift at 1.0
                                        max_len=self.max_len,
                                        use_parallel=True
                                    )

                                year_rules = try_relaxed_parameters()

                            rules_to_display = year_rules
                            year_title = f"Year {selected_year}"
                    except Exception as e:
                            # Show error message in the content frame
                            error_label = ttk.Label(content_frame, text=f"Error generating rules for year {selected_year}: {str(e)}")
                            error_label.pack(pady=20)
                            return

                    # Display rules for the selected year
                    if rules_to_display is None or len(rules_to_display) == 0:
                        no_rules_label = ttk.Label(content_frame, text=f"No association rules found for {year_title}.")
                        no_rules_label.pack(pady=20)
                        return

                    # Create title for the rules
                    rules_title = ttk.Label(content_frame, text=f"Association Rules for {year_title}", font=("Arial", 12, "bold"))
                    rules_title.pack(pady=5)

                    # Create a frame for the rules table
                    table_frame = ttk.Frame(content_frame)
                    table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                    # Create a table for the rules
                    columns = ("antecedents", "consequents", "support", "confidence", "lift")
                    tree = ttk.Treeview(table_frame, columns=columns, show="headings")

                    # Define column headings
                    tree.heading("antecedents", text="Antecedents")
                    tree.heading("consequents", text="Consequents")
                    tree.heading("support", text="Support")
                    tree.heading("confidence", text="Confidence")
                    tree.heading("lift", text="Lift")

                    # Define column widths
                    tree.column("antecedents", width=300)
                    tree.column("consequents", width=300)
                    tree.column("support", width=100)
                    tree.column("confidence", width=100)
                    tree.column("lift", width=100)

                    # Add scrollbar
                    tree_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
                    tree.configure(yscrollcommand=tree_scrollbar.set)
                    tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                    # Add rules to the table
                    for i, rule in rules_to_display.iterrows():
                        antecedents = ', '.join(list(rule['antecedents']))
                        consequents = ', '.join(list(rule['consequents']))
                        tree.insert("", "end", values=(
                            antecedents,
                            consequents,
                            f"{rule['support']:.3f}",
                            f"{rule['confidence']:.3f}",
                            f"{rule['lift']:.3f}"
                        ))

                    # Add interpretation section
                    interpretation_frame = ttk.LabelFrame(content_frame, text="Rule Interpretations")
                    interpretation_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

                    # Add text widget for interpretations
                    interpretation_text = scrolledtext.ScrolledText(interpretation_frame, wrap=tk.WORD, height=15)
                    interpretation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                    # Generate and display interpretations
                    interpretations = interpret_event_association_rules(rules_to_display)
                    interpretation_text.insert(tk.END, interpretations)
                    interpretation_text.config(state='disabled')  # Make read-only

                # Bind the dropdown callback
                year_dropdown.bind('<<ComboboxSelected>>', on_year_change)

                # Trigger the dropdown callback to show combined rules initially
                on_year_change()
                                else:
                # If only one year, display rules directly
                if rules is None or len(rules) == 0:
                    label = ttk.Label(plot_frame, text="No association rules found.")
                    label.pack(pady=20)
                    return

                # Create a frame for the rules table
                table_frame = ttk.Frame(plot_frame)
                table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Create a table for the rules
                columns = ("antecedents", "consequents", "support", "confidence", "lift")
                tree = ttk.Treeview(table_frame, columns=columns, show="headings")

                # Define column headings
                tree.heading("antecedents", text="Antecedents")
                tree.heading("consequents", text="Consequents")
                tree.heading("support", text="Support")
                tree.heading("confidence", text="Confidence")
                tree.heading("lift", text="Lift")

                # Define column widths
                tree.column("antecedents", width=300)
                tree.column("consequents", width=300)
                tree.column("support", width=100)
                tree.column("confidence", width=100)
                tree.column("lift", width=100)

                # Add scrollbar
                tree_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
                tree.configure(yscrollcommand=tree_scrollbar.set)
                tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                # Add rules to the table
                for i, rule in rules.iterrows():
                    antecedents = ', '.join(list(rule['antecedents']))
                    consequents = ', '.join(list(rule['consequents']))
                    tree.insert("", "end", values=(
                        antecedents,
                        consequents,
                        f"{rule['support']:.3f}",
                        f"{rule['confidence']:.3f}",
                        f"{rule['lift']:.3f}"
                    ))

                # Add interpretation section
                interpretation_frame = ttk.LabelFrame(plot_frame, text="Rule Interpretations")
                interpretation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Add text widget for interpretations
                interpretation_text = scrolledtext.ScrolledText(interpretation_frame, wrap=tk.WORD, height=15)
                interpretation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # Generate and display interpretations
                interpretations = interpret_event_association_rules(rules)
                interpretation_text.insert(tk.END, interpretations)
                interpretation_text.config(state='disabled')  # Make read-only

            # Function to handle resize events
            def on_rules_resize(event):
                # Only redraw if the event is for our window and the size actually changed
                if event.widget == self.root and (event.width != getattr(self, '_last_width_rules', 0) or
                                                 event.height != getattr(self, '_last_height_rules', 0)):
                    self._last_width_rules = event.width
                    self._last_height_rules = event.height

            # Bind the resize event
            self.root.bind("<Configure>", on_rules_resize)

            # Trigger an initial resize
            self.root.update_idletasks()
            on_rules_resize(type('Event', (), {'widget': self.root, 'width': self.root.winfo_width(),
                                              'height': self.root.winfo_height()})())

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            logging.error(f"Error in plot_association_rules: {str(e)}")
            for widget in self.rules_tab.winfo_children():
                widget.destroy()
            label = ttk.Label(self.rules_tab, text=f"Error plotting association rules: {str(e)}")
            label.pack(pady=20)

    def plot_descriptive(self):
        """Create descriptive statistics visualization with year selection"""
        # Clear previous content in descriptive tab
        for widget in self.descriptive_tab.winfo_children():
            widget.destroy()

        # Create a canvas with scrollbar for descriptive tab
        canvas = tk.Canvas(self.descriptive_tab, width=1500, height=800)
        scrollbar = ttk.Scrollbar(self.descriptive_tab, orient="vertical", command=canvas.yview)

        # Create a frame inside the canvas for the plots
        plot_frame = ttk.Frame(canvas, width=1500)

        # Configure scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create window in canvas
        canvas.create_window((0, 0), window=plot_frame, anchor="nw", width=canvas.winfo_width())

        # Configure the plot frame to expand to fill the canvas width
        def configure_plot_frame(event):
            canvas_width = event.width
            canvas.itemconfig(canvas.find_withtag("all")[0], width=canvas_width)

        canvas.bind("<Configure>", configure_plot_frame)

        # Store plots for later reference
        self.descriptive_plots = []

        try:
            # Get filtered data
            filtered_df = self.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                label = ttk.Label(plot_frame, text="No data available for descriptive analysis.")
                label.pack(pady=20)
                return

            # Create a header with dataset information
            header_frame = ttk.Frame(plot_frame)
            header_frame.pack(fill=tk.X, padx=10, pady=10)

            if len(self.datasets) > 1:
                header_text = f"Descriptive Statistics - Combined Data from {len(self.datasets)} Years ({', '.join(sorted(self.datasets.keys()))})"
            else:
                year = next(iter(self.datasets.keys()))
                header_text = f"Descriptive Statistics - Year {year}"

            header_label = ttk.Label(header_frame, text=header_text, font=("Arial", 14, "bold"))
            header_label.pack(pady=5)

            # If multiple datasets are loaded, add a dropdown to select which year to view
            if len(self.datasets) > 1:
                year_frame = ttk.Frame(plot_frame)
                year_frame.pack(fill=tk.X, padx=10, pady=5)

                ttk.Label(year_frame, text="Select Year:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

                year_var = tk.StringVar(value="Combined")
                year_options = ["Combined"] + sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly", width=15, font=("Arial", 12))
                year_dropdown.pack(side=tk.LEFT, padx=5)
                year_dropdown.set("Combined")  # Default to combined view

                # Create a container frame for the statistics content
                content_frame = ttk.Frame(plot_frame)
                content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                def on_year_change(event=None):
                    # Clear existing content
                    for widget in content_frame.winfo_children():
                        widget.destroy()

                    # Get data for selected year
                    selected_year = year_var.get()

                    if selected_year == "Combined":
                        # Use the combined data
                        df_to_analyze = filtered_df
                        year_title = "Combined Data"
                    else:
                        # Get data for the specific year
                        df_to_analyze = self.datasets[selected_year]
                        if self.current_department.get() != "All Departments":
                            df_to_analyze = df_to_analyze[df_to_analyze['department_name'] == self.current_department.get()]
                        year_title = f"Year {selected_year}"

                    # Create a frame for the summary statistics
                    stats_frame = ttk.Frame(content_frame)
                    stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                    # Add a label for the summary statistics section
                    summary_label = tk.Label(stats_frame, text=f"Summary Statistics for {year_title}", font=("Arial", 12, "bold"))
                    summary_label.pack(anchor="w", pady=(0, 10))

                    # Filter out non-numeric columns
                    numeric_df = df_to_analyze.select_dtypes(include=['number'])
            if 'department_name' in numeric_df.columns:
                numeric_df = numeric_df.drop('department_name', axis=1)

                    if numeric_df.empty:
                        ttk.Label(stats_frame, text="No numeric data available for analysis.").pack(pady=10)
                        return

            # Calculate summary statistics
            summary_stats = numeric_df.describe().T
                    summary_stats['cv'] = summary_stats['std'] / summary_stats['mean']  # Coefficient of variation

                    # Create a scrollable frame for the feature statistics
                    stats_canvas = tk.Canvas(stats_frame, width=1450, height=600)
                    stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=stats_canvas.yview)
                    stats_inner_frame = ttk.Frame(stats_canvas)

                    stats_canvas.configure(yscrollcommand=stats_scrollbar.set)
                    stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    stats_canvas.create_window((0, 0), window=stats_inner_frame, anchor="nw")

                    def configure_stats_scroll(event):
                        stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))

                    stats_inner_frame.bind('<Configure>', configure_stats_scroll)

                    # Add feature statistics
                    row = 0
                    for feature, stat_row in summary_stats.iterrows():
                        feature_name = feature.replace('_', ' ').title()

                        # Create a frame for each feature - using standard tk.LabelFrame instead of ttk.LabelFrame for font support
                        feature_frame = tk.LabelFrame(stats_inner_frame, text=feature_name, font=("Arial", 12, "bold"))
                        feature_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)

                        # Add statistics for this feature
                        grid_frame = ttk.Frame(feature_frame)
                        grid_frame.pack(fill=tk.X, padx=10, pady=5)

                        # Create a grid of labels for the statistics
                        ttk.Label(grid_frame, text="Mean:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
                        ttk.Label(grid_frame, text=f"{stat_row['mean']:.4f}", font=("Arial", 11)).grid(row=0, column=1, sticky="w", padx=5, pady=2)

                        ttk.Label(grid_frame, text="Median:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=2)
                        ttk.Label(grid_frame, text=f"{stat_row['50%']:.4f}", font=("Arial", 11)).grid(row=1, column=1, sticky="w", padx=5, pady=2)

                        ttk.Label(grid_frame, text="Std Dev:", font=("Arial", 11, "bold")).grid(row=0, column=2, sticky="w", padx=5, pady=2)
                        ttk.Label(grid_frame, text=f"{stat_row['std']:.4f}", font=("Arial", 11)).grid(row=0, column=3, sticky="w", padx=5, pady=2)

                        ttk.Label(grid_frame, text="Range:", font=("Arial", 11, "bold")).grid(row=1, column=2, sticky="w", padx=5, pady=2)
                        ttk.Label(grid_frame, text=f"{stat_row['min']:.2f} - {stat_row['max']:.2f}", font=("Arial", 11)).grid(row=1, column=3, sticky="w", padx=5, pady=2)

                        ttk.Label(grid_frame, text="CV:", font=("Arial", 11, "bold")).grid(row=2, column=0, sticky="w", padx=5, pady=2)
                        ttk.Label(grid_frame, text=f"{stat_row['cv']:.4f}", font=("Arial", 11)).grid(row=2, column=1, sticky="w", padx=5, pady=2)

                        row += 1

                    # Add explanation of statistical measures
                    explanation_frame = tk.LabelFrame(stats_frame, text="Statistical Measures Explained", font=("Arial", 12, "bold"))
                    explanation_frame.pack(fill=tk.X, padx=10, pady=10)

                    explanation_text = """
                    Mean: The average value of the feature across all events.
                    Median: The middle value when all values are arranged in order (50th percentile).
                    Std Dev: Standard deviation - measures the amount of variation or dispersion in the values.
                    Range: The span from minimum to maximum values observed.
                    CV: Coefficient of Variation - the ratio of the standard deviation to the mean, useful for comparing variability between features.
                    """

                    explanation_label = ttk.Label(explanation_frame, text=explanation_text, font=("Arial", 11), justify=tk.LEFT)
                    explanation_label.pack(padx=10, pady=10, anchor="w")

                # Bind the dropdown callback
                year_dropdown.bind('<<ComboboxSelected>>', on_year_change)

                # Trigger the dropdown callback to show combined statistics initially
                on_year_change()
                        else:
                # If only one year, display statistics directly
                content_frame = ttk.Frame(plot_frame)
                content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                year = next(iter(self.datasets.keys()))
                df_to_analyze = filtered_df
                year_title = f"Year {year}"

                # Create a frame for the summary statistics
                stats_frame = ttk.Frame(content_frame)
                stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # Add a label for the summary statistics section
                summary_label = tk.Label(stats_frame, text=f"Summary Statistics for {year_title}", font=("Arial", 12, "bold"))
                summary_label.pack(anchor="w", pady=(0, 10))

                # Filter out non-numeric columns
                numeric_df = df_to_analyze.select_dtypes(include=['number'])
                if 'department_name' in numeric_df.columns:
                    numeric_df = numeric_df.drop('department_name', axis=1)

                if numeric_df.empty:
                    ttk.Label(stats_frame, text="No numeric data available for analysis.").pack(pady=10)
                    return

                # Calculate summary statistics
                summary_stats = numeric_df.describe().T
                summary_stats['cv'] = summary_stats['std'] / summary_stats['mean']  # Coefficient of variation

                # Create a scrollable frame for the feature statistics
                stats_canvas = tk.Canvas(stats_frame, width=1450, height=600)
                stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=stats_canvas.yview)
                stats_inner_frame = ttk.Frame(stats_canvas)

                stats_canvas.configure(yscrollcommand=stats_scrollbar.set)
                stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                stats_canvas.create_window((0, 0), window=stats_inner_frame, anchor="nw")

                def configure_stats_scroll(event):
                    stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))

                stats_inner_frame.bind('<Configure>', configure_stats_scroll)

                # Add feature statistics
                row = 0
                for feature, stat_row in summary_stats.iterrows():
                    feature_name = feature.replace('_', ' ').title()

                    # Create a frame for each feature - using standard tk.LabelFrame instead of ttk.LabelFrame for font support
                    feature_frame = tk.LabelFrame(stats_inner_frame, text=feature_name, font=("Arial", 12, "bold"))
                    feature_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)

                    # Add statistics for this feature
                    grid_frame = ttk.Frame(feature_frame)
                    grid_frame.pack(fill=tk.X, padx=10, pady=5)

                    # Create a grid of labels for the statistics
                    ttk.Label(grid_frame, text="Mean:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(grid_frame, text=f"{stat_row['mean']:.4f}", font=("Arial", 11)).grid(row=0, column=1, sticky="w", padx=5, pady=2)

                    ttk.Label(grid_frame, text="Median:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(grid_frame, text=f"{stat_row['50%']:.4f}", font=("Arial", 11)).grid(row=1, column=1, sticky="w", padx=5, pady=2)

                    ttk.Label(grid_frame, text="Std Dev:", font=("Arial", 11, "bold")).grid(row=0, column=2, sticky="w", padx=5, pady=2)
                    ttk.Label(grid_frame, text=f"{stat_row['std']:.4f}", font=("Arial", 11)).grid(row=0, column=3, sticky="w", padx=5, pady=2)

                    ttk.Label(grid_frame, text="Range:", font=("Arial", 11, "bold")).grid(row=1, column=2, sticky="w", padx=5, pady=2)
                    ttk.Label(grid_frame, text=f"{stat_row['min']:.2f} - {stat_row['max']:.2f}", font=("Arial", 11)).grid(row=1, column=3, sticky="w", padx=5, pady=2)

                    ttk.Label(grid_frame, text="CV:", font=("Arial", 11, "bold")).grid(row=2, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(grid_frame, text=f"{stat_row['cv']:.4f}", font=("Arial", 11)).grid(row=2, column=1, sticky="w", padx=5, pady=2)

                    row += 1

                # Add explanation of statistical measures
                explanation_frame = tk.LabelFrame(stats_frame, text="Statistical Measures Explained", font=("Arial", 12, "bold"))
                explanation_frame.pack(fill=tk.X, padx=10, pady=10)

                explanation_text = """
                Mean: The average value of the feature across all events.
                Median: The middle value when all values are arranged in order (50th percentile).
                Std Dev: Standard deviation - measures the amount of variation or dispersion in the values.
                Range: The span from minimum to maximum values observed.
                CV: Coefficient of Variation - the ratio of the standard deviation to the mean, useful for comparing variability between features.
                """

                explanation_label = ttk.Label(explanation_frame, text=explanation_text, font=("Arial", 11), justify=tk.LEFT)
                explanation_label.pack(padx=10, pady=10, anchor="w")

            # Configure scroll region when plot frame changes
            def configure_scroll_region(event):
                canvas.configure(scrollregion=canvas.bbox("all"))

            plot_frame.bind('<Configure>', configure_scroll_region)

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            logging.error(f"Error creating descriptive statistics: {str(e)}")
            ttk.Label(self.descriptive_tab, text=f"Error creating descriptive statistics: {str(e)}").pack(pady=20)

    def plot_recommendations(self):
        """Create recommendations visualization with year selection"""
        try:
            # Clear previous content in recommendations tab
            for widget in self.recommendations_tab.winfo_children():
                widget.destroy()

            # Create a canvas with scrollbar for recommendations tab
            canvas = tk.Canvas(self.recommendations_tab, width=1500, height=800)
            scrollbar = ttk.Scrollbar(self.recommendations_tab, orient="vertical", command=canvas.yview)

            # Create a frame inside the canvas for the recommendations
            plot_frame = ttk.Frame(canvas, width=1500)

            # Configure scrolling
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create window in canvas
            canvas.create_window((0, 0), window=plot_frame, anchor="nw", width=canvas.winfo_width())

            # Configure the plot frame to expand to fill the canvas width
            def configure_plot_frame(event):
                canvas_width = event.width
                canvas.itemconfig(canvas.find_withtag("all")[0], width=canvas_width)

            canvas.bind("<Configure>", configure_plot_frame)

            # Get filtered data
            filtered_df = self.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                label = ttk.Label(plot_frame, text="No data available for recommendations.")
                label.pack(pady=20)
                return

            # Create a header with dataset information
            header_frame = ttk.Frame(plot_frame)
            header_frame.pack(fill=tk.X, padx=10, pady=10)

            if len(self.datasets) > 1:
                header_text = f"Recommendations - Combined Data from {len(self.datasets)} Years ({', '.join(sorted(self.datasets.keys()))})"
            else:
                year = next(iter(self.datasets.keys()))
                header_text = f"Recommendations - Year {year}"

            header_label = ttk.Label(header_frame, text=header_text, font=("Arial", 14, "bold"))
            header_label.pack(pady=5)

            # If multiple datasets are loaded, add a dropdown to select which year to view
            if len(self.datasets) > 1:
                year_frame = ttk.Frame(plot_frame)
                year_frame.pack(fill=tk.X, padx=10, pady=5)

                ttk.Label(year_frame, text="Select Year:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

                year_var = tk.StringVar(value="Combined")
                year_options = ["Combined"] + sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly", width=15, font=("Arial", 12))
                year_dropdown.pack(side=tk.LEFT, padx=5)
                year_dropdown.set("Combined")  # Default to combined view

                # Create a container frame for the recommendations content
                content_frame = ttk.Frame(plot_frame)
                content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                def on_year_change(event=None):
                    # Clear existing content
                    for widget in content_frame.winfo_children():
                        widget.destroy()

                    # Get data for selected year
                    selected_year = year_var.get()

                    if selected_year == "Combined":
                        # Use the combined data
                        df_to_analyze = filtered_df
                        year_title = "Combined Data"
                    else:
                        # Get data for the specific year
                        df_to_analyze = self.datasets[selected_year]
                        if self.current_department.get() != "All Departments":
                            df_to_analyze = df_to_analyze[df_to_analyze['department_name'] == self.current_department.get()]
                        year_title = f"Year {selected_year}"

                    # Create recommendations for the selected year
                    self.create_recommendations_for_year(content_frame, df_to_analyze, year_title)

                # Bind the dropdown callback
                year_dropdown.bind('<<ComboboxSelected>>', on_year_change)

                # Trigger the dropdown callback to show combined recommendations initially
                on_year_change()
            else:
                # If only one year, display recommendations directly
                content_frame = ttk.Frame(plot_frame)
                content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                year = next(iter(self.datasets.keys()))
                self.create_recommendations_for_year(content_frame, filtered_df, f"Year {year}")

            # Configure scroll region when plot frame changes
            def configure_scroll_region(event):
                canvas.configure(scrollregion=canvas.bbox("all"))

            plot_frame.bind('<Configure>', configure_scroll_region)

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            logging.error(f"Error in plot_recommendations: {str(e)}")
            for widget in self.recommendations_tab.winfo_children():
                widget.destroy()
            label = ttk.Label(self.recommendations_tab, text=f"Error generating recommendations: {str(e)}")
            label.pack(pady=20)

    def create_recommendations_for_year(self, parent_frame, df, title):
        """Create recommendations for a specific year"""
        try:
            # Create a title for the recommendations
            title_label = ttk.Label(parent_frame, text=f"Recommendations for {title}", font=("Arial", 12, "bold"))
            title_label.pack(pady=5)

            # Create recommendations text widget
            recommendations_text = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=30, font=("Arial", 11))
            recommendations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Calculate average scores for each feature
            numeric_df = df.copy()
            if 'department_name' in numeric_df.columns:
                numeric_df = numeric_df.drop('department_name', axis=1)

            # Convert all columns to numeric
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

            # Calculate average scores
            avg_scores = numeric_df.mean()

            # Identify low and high scoring features
            needs_improvement = avg_scores[avg_scores < 1.5]  # Below 1.5 on 0-3 scale
            very_satisfactory = avg_scores[avg_scores >= 2.25]  # Above 2.25 on 0-3 scale

            # Generate standard recommendations based on low scores
            standard_recommendations = generate_event_recommendations(needs_improvement)

            # Generate maintenance recommendations for high scores
            maintenance_recommendations = generate_event_maintenance_recommendations(very_satisfactory)

            # Generate dynamic recommendations
            dynamic_recommendations = self.generate_dynamic_recommendations(
                numeric_df, needs_improvement, very_satisfactory
            )

            # Display recommendations
            recommendations_text.insert(tk.END, "IMPROVEMENT RECOMMENDATIONS:\n\n")

            if standard_recommendations:
                for feature, recs in standard_recommendations.items():
                    feature_name = feature.replace('_', ' ').title()
                    recommendations_text.insert(tk.END, f"{feature_name}:\n")

                    for rec in recs:
                        recommendations_text.insert(tk.END, f"• {rec['text']}\n")
                        recommendations_text.insert(tk.END, f"  Action: {rec['action']}\n\n")
            else:
                recommendations_text.insert(tk.END, "No improvement recommendations identified.\n\n")

            # Add dynamic recommendations
            recommendations_text.insert(tk.END, "\nDETAILED RECOMMENDATIONS:\n\n")

            if dynamic_recommendations:
                has_recommendations = False
                for feature, recs in dynamic_recommendations.items():
                    if recs:  # Only show features with recommendations
                        has_recommendations = True
                        feature_name = feature.replace('_', ' ').title()
                        recommendations_text.insert(tk.END, f"{feature_name}:\n")

                        for rec in recs:
                            recommendations_text.insert(tk.END, f"• {rec['text']} (Priority: {rec['priority']})\n")
                            recommendations_text.insert(tk.END, f"  {rec['action']}\n\n")

                if not has_recommendations:
                    recommendations_text.insert(tk.END, "No detailed recommendations identified.\n\n")
                else:
                  recommendations_text.insert(tk.END, "No detailed recommendations identified.\n\n")

            # Add maintenance recommendations
            recommendations_text.insert(tk.END, "\nMAINTENANCE RECOMMENDATIONS:\n\n")

            if maintenance_recommendations:
                for feature, recs in maintenance_recommendations.items():
                    feature_name = feature.replace('_', ' ').title()
                    recommendations_text.insert(tk.END, f"{feature_name}:\n")

                    for rec in recs:
                        recommendations_text.insert(tk.END, f"• {rec['text']}\n")
                        recommendations_text.insert(tk.END, f"  Action: {rec['action']}\n\n")
            else:
                recommendations_text.insert(tk.END, "No maintenance recommendations identified.\n\n")

            # Make text read-only
            recommendations_text.config(state='disabled')

        except Exception as e:
            logging.error(f"Error creating recommendations for {title}: {str(e)}")
            ttk.Label(parent_frame, text=f"Error creating recommendations for {title}: {str(e)}").pack(pady=20)

    def generate_dynamic_recommendations(self, df, low_scores, high_scores):
        """
        Generate dynamic recommendations based on data analysis.

        Args:
            df: DataFrame with event data
            low_scores: Series of features with low scores
            high_scores: Series of features with high scores

        Returns:
            Dictionary of feature-specific recommendations
        """
        recommendations = {}

        try:
            # For each feature with low scores, generate detailed recommendations
            for feature in low_scores.index:
                feature_recs = []

                # Get the average score for this feature
                avg_score = low_scores[feature]

                # Determine priority based on score (lower score = higher priority)
                if avg_score < 1.0:
                    priority = "High"
                elif avg_score < 1.25:
                    priority = "Medium-High"
                else:
                    priority = "Medium"

                # Generate feature-specific recommendations
                if feature == 'content_relevance':
                    feature_recs.append({
                        'text': f"Content relevance score is low ({avg_score:.2f}/3.0).",
                        'action': "Review content selection process and implement a pre-event survey to better understand participant interests and needs.",
                        'priority': priority
                    })

                elif feature == 'speaker_quality':
                    feature_recs.append({
                        'text': f"Speaker quality score is low ({avg_score:.2f}/3.0).",
                        'action': "Implement a more rigorous speaker selection process and provide presentation training for speakers.",
                        'priority': priority
                    })

                elif feature == 'organization':
                    feature_recs.append({
                        'text': f"Organization score is low ({avg_score:.2f}/3.0).",
                        'action': "Review event planning timeline and checklist. Consider using project management software for better coordination.",
                        'priority': priority
                    })

                elif feature == 'venue_quality':
                    feature_recs.append({
                        'text': f"Venue quality score is low ({avg_score:.2f}/3.0).",
                        'action': "Evaluate alternative venues or negotiate improvements with current venue management.",
                        'priority': priority
                    })

                elif feature == 'networking_opportunities':
                    feature_recs.append({
                        'text': f"Networking opportunities score is low ({avg_score:.2f}/3.0).",
                        'action': "Add structured networking sessions and use technology to facilitate connections between participants.",
                        'priority': priority
                    })

                elif feature == 'value_for_money':
                    feature_recs.append({
                        'text': f"Value for money score is low ({avg_score:.2f}/3.0).",
                        'action': "Review pricing strategy and add more value components to the event package.",
                        'priority': priority
                    })

                elif feature == 'catering_quality':
                    feature_recs.append({
                        'text': f"Catering quality score is low ({avg_score:.2f}/3.0).",
                        'action': "Change catering provider or negotiate menu improvements with current provider.",
                        'priority': priority
                    })

                elif feature == 'audio_visual_quality':
                    feature_recs.append({
                        'text': f"Audio/visual quality score is low ({avg_score:.2f}/3.0).",
                        'action': "Upgrade A/V equipment or hire professional A/V technicians for the event.",
                        'priority': priority
                    })

                # Add the recommendations for this feature
                if feature_recs:
                    recommendations[feature] = feature_recs

            # Look for correlations between features to generate additional insights
            if len(df.columns) > 1:
                try:
                    # Calculate correlation matrix
                    corr_matrix = df.corr()

                    # Find strong correlations (positive or negative)
                    for feature in low_scores.index:
                        if feature in corr_matrix.columns:
                            # Get correlations for this feature
                            correlations = corr_matrix[feature].sort_values(ascending=False)

                            # Find strong positive correlations (excluding self-correlation)
                            strong_pos_corr = correlations[(correlations > 0.6) & (correlations < 0.99)]

                            # Find strong negative correlations
                            strong_neg_corr = correlations[correlations < -0.4]

                            # Generate recommendations based on correlations
                            if not strong_pos_corr.empty:
                                for corr_feature, corr_value in strong_pos_corr.items():
                                    if corr_feature in high_scores.index:
                                        # This is interesting - a high-scoring feature correlates with a low-scoring one
                                        if feature not in recommendations:
                                            recommendations[feature] = []

                                        recommendations[feature].append({
                                            'text': f"Strong positive correlation ({corr_value:.2f}) with high-scoring feature '{corr_feature.replace('_', ' ').title()}'.",
                                            'action': f"Leverage strategies from '{corr_feature.replace('_', ' ').title()}' to improve this area.",
                                            'priority': "Medium"
                                        })

                            if not strong_neg_corr.empty:
                                for corr_feature, corr_value in strong_neg_corr.items():
                                    if feature not in recommendations:
                                        recommendations[feature] = []

                                    recommendations[feature].append({
                                        'text': f"Strong negative correlation ({corr_value:.2f}) with feature '{corr_feature.replace('_', ' ').title()}'.",
                                        'action': f"Investigate potential trade-offs between '{feature.replace('_', ' ').title()}' and '{corr_feature.replace('_', ' ').title()}'.",
                                        'priority': "Medium"
                                    })
                except Exception as corr_e:
                    logging.warning(f"Error calculating correlations for recommendations: {str(corr_e)}")

        except Exception as e:
            logging.error(f"Error generating dynamic recommendations: {str(e)}")

        return recommendations
