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
import io
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from functools import lru_cache
import inspect

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
    Generate association rules from binary data using the Apriori algorithm.

    Parameters:
    - binary_df: DataFrame with binary encoded features
    - min_support: Minimum support threshold
    - min_confidence: Minimum confidence threshold
    - min_lift: Minimum lift threshold
    - max_len: Maximum length of itemsets
    - use_parallel: Whether to use parallel processing

    Returns:
    - rules: DataFrame containing association rules
    """
    try:
        start_time = time.time()

        # Check if we're in PDF generation mode (for optimizations)
        pdf_generation_mode = False
        frame = inspect.currentframe()
        while frame:
            if '_generating_pdf' in frame.f_locals and frame.f_locals['_generating_pdf']:
                pdf_generation_mode = True
                break
            frame = frame.f_back

        # If we're generating a PDF, use simpler parameters for speed
        if pdf_generation_mode:
            logging.info("Optimizing association rules for PDF generation")
            min_support = max(0.1, min_support)  # Increase min_support for speed
            max_len = min(3, max_len)  # Limit max_len for speed
            use_parallel = False  # Disable parallel for stability in PDF generation

        logging.info(f"Generating association rules with min_support={min_support}, "
                    f"min_confidence={min_confidence}, min_lift={min_lift}")

        # Get frequent itemsets
        frequent_itemsets = apriori(binary_df,
                                   min_support=min_support,
                                   max_len=max_len,
                                   use_colnames=True,
                                   verbose=0,
                                   low_memory=True)

        if frequent_itemsets is None or len(frequent_itemsets) == 0:
            logging.warning("No frequent itemsets found. Trying with lower support threshold...")
            # Try with lower support
            new_min_support = min_support / 2 if min_support > 0.01 else 0.01
            frequent_itemsets = apriori(binary_df,
                                       min_support=new_min_support,
                                       max_len=max_len,
                                       use_colnames=True,
                                       verbose=0,
                                       low_memory=True)

            if frequent_itemsets is None or len(frequent_itemsets) == 0:
                logging.warning("Still no frequent itemsets found.")
                return pd.DataFrame()  # Return empty DataFrame

        logging.info(f"Found {len(frequent_itemsets)} frequent itemsets")

        # Generate rules
        rules = association_rules(frequent_itemsets,
                                 metric="confidence",
                                 min_threshold=min_confidence)

        # Filter rules by lift
        rules = rules[rules['lift'] >= min_lift]

        # Sort rules by lift
        rules = rules.sort_values('lift', ascending=False)

        logging.info(f"Generated {len(rules)} rules with min_lift={min_lift}")
        logging.info(f"Association rules generation took {time.time() - start_time:.2f} seconds")

        return rules

    except Exception as e:
        logging.error(f"Error in generate_association_rules: {str(e)}")
        # Return empty DataFrame
        return pd.DataFrame()

def generate_recommendations_from_rules(self, rules, df, min_lift=1.5, max_recommendations=10):
    """
    Generate recommendations based on association rules analysis using fixed rating scale.
    Includes recommendations for features with ratings below the maximum score.
    Excludes Overall_Rating from recommendations.
    """
    recommendations = {}

    if rules.empty:
        return recommendations

    # Use fixed rating scale thresholds
    min_rating = self.RATING_SCALE['min']
    max_rating = self.RATING_SCALE['max']
    needs_improvement = self.RATING_SCALE['thresholds']['needs_improvement']
    very_satisfactory = self.RATING_SCALE['thresholds']['very_satisfactory']

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

                # Include features with ratings below the maximum (3.0)
                # Skip only if rating is very satisfactory and at or near the maximum
                if feature in avg_ratings and avg_ratings[feature] >= max_rating * 0.95:
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

                    if rating in ['Needs_Improvement', 'Moderately_Satisfactory', 'Satisfactory']:
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
                        elif rating == 'Satisfactory':
                            recommendation['action'] += f"\n• Fine-tune {feature.replace('_', ' ').lower()} elements"
                            recommendation['action'] += f"\n• Schedule quarterly improvement reviews"
                            recommendation['priority'] = 'Low'

                        if recommendation not in recommendations[feature]:
                            recommendations[feature].append(recommendation)
                            recommendation_count += 1

                            if recommendation_count >= max_recommendations:
                                return recommendations

    return recommendations

def get_common_issues_dictionary():
    """
    Returns a dictionary of common issues for features with low scores.
    This provides standardized explanations for potential problem areas.
    """
    return {
        'Overall_Rating': "Low overall rating often indicates general dissatisfaction with the event as a whole. This could be due to a combination of factors including poor organization, irrelevant content, or inadequate facilities.",

        'Objectives_Met': "Low scores on objectives met suggest that the event failed to deliver on its promised outcomes or goals. Participants may have felt the event description misrepresented what was actually delivered.",

        'Venue_Rating': "Poor venue ratings typically indicate issues with physical comfort, accessibility, or suitability of the location. This could include problems with temperature control, seating comfort, acoustics, or technical capabilities.",

        'Schedule_Rating': "Low schedule ratings suggest timing issues such as poor session pacing, inadequate breaks, inconvenient start/end times, or sessions running over their allotted time.",

        'Speaker_Rating': "Poor speaker ratings may indicate presenters who were unprepared, unengaging, difficult to understand, or lacking expertise in their subject matter.",

        'content_relevance': "Low scores for content relevance indicate that the material presented did not align with participant expectations or needs. The content may have been too basic, too advanced, or off-topic for the target audience.",

        'organization': "Poor organization scores suggest logistical problems, unclear instructions, registration issues, or general confusion about the event flow.",

        'venue_quality': "Low venue quality scores may point to inadequate facilities, poor maintenance, technical difficulties, or insufficient amenities.",

        'networking_opportunities': "Poor scores for networking opportunities indicate insufficient structured interaction between participants, inadequate time for networking, or poorly facilitated networking activities.",

        'value_for_money': "Low value for money ratings suggest that participants did not feel the event was worth the cost of attendance, whether in terms of ticket price, time investment, or travel expenses.",

        'catering_quality': "Poor catering ratings typically relate to food quality, insufficient quantity, limited options (especially for dietary restrictions), or poor service.",

        'audio_visual_quality': "Low scores for audio/visual quality suggest technical problems such as poor sound systems, visibility issues with presentations, or technical failures during the event."
    }

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

def generate_event_improvement_recommendations(moderate_scores):
    """Generate improvement recommendations for scores that are not at the highest level but not in needs improvement category."""
    improvement_recommendations = {
        'Overall_Rating': [
            {
                'text': "Identify specific areas for enhancement",
                'action': "Conduct targeted surveys to identify improvement opportunities",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Objectives_Met': [
            {
                'text': "Enhance objective clarity and achievement",
                'action': "Review and refine objective-setting process",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Venue_Rating': [
            {
                'text': "Optimize current venue setup",
                'action': "Identify specific venue improvements for next event",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Schedule_Rating': [
            {
                'text': "Fine-tune event scheduling",
                'action': "Analyze session timing and make targeted adjustments",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'Speaker_Rating': [
            {
                'text': "Enhance speaker preparation and support",
                'action': "Implement pre-event speaker coaching sessions",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'content_relevance': [
            {
                'text': "Enhance content relevance for target audience",
                'action': "Conduct pre-event content surveys with participants",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'organization': [
            {
                'text': "Streamline event organization",
                'action': "Review and optimize event logistics process",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'venue_quality': [
            {
                'text': "Enhance venue quality elements",
                'action': "Identify specific venue enhancements",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'networking_opportunities': [
            {
                'text': "Enhance networking opportunities",
                'action': "Add more structured networking activities",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ],
        'value_for_money': [
            {
                'text': "Increase perceived value",
                'action': "Add additional value elements to event package",
                'support': 1.0,
                'confidence': 1.0,
                'lift': 1.0
            }
        ]
    }
    return {k: improvement_recommendations.get(k, []) for k in moderate_scores.index if k in improvement_recommendations}

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

            # Keep reference to the navigation frame and buttons
            self.nav_frame = None
            self.nav_buttons = {}

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

            # Define color scheme and fonts
            self.colors = {
                'primary': '#1a73e8',       # Main blue color
                'secondary': '#4285f4',     # Secondary blue
                'success': '#0f9d58',       # Green for success
                'warning': '#f4b400',       # Yellow for warnings
                'danger': '#db4437',        # Red for errors
                'light_bg': '#f8f9fa',      # Light background
                'dark_bg': '#202124',       # Dark background
                'text': '#3c4043',          # Text color
                'light_text': '#5f6368',    # Secondary text
                'highlight': '#e8f0fe'      # Highlight background
            }

            # Fonts
            self.fonts = {
                'heading': ('Segoe UI', 12, 'bold'),
                'subheading': ('Segoe UI', 11, 'bold'),
                'body': ('Segoe UI', 10),
                'small': ('Segoe UI', 9),
                'code': ('Consolas', 10)
            }

            self.root = root
            self.root.title("Event Analysis Dashboard")

            # Apply a theme if ttk style is available
            self.style = ttk.Style()
            try:
                # Try to set a modern theme if available
                import ttkthemes
                self.style = ttkthemes.ThemedStyle(self.root)
                self.style.set_theme("arc")  # Modern clean theme
            except ImportError:
                # Fall back to clam theme which is available in standard ttk
                self.style.theme_use('clam')

            # Configure common styles
            self.style.configure('TButton', font=self.fonts['body'], padding=6)
            self.style.configure('TLabel', font=self.fonts['body'])
            self.style.configure('Heading.TLabel', font=self.fonts['heading'])
            self.style.configure('Subheading.TLabel', font=self.fonts['subheading'])
            self.style.configure('Action.TButton', foreground=self.colors['primary'])
            self.style.configure('Success.TButton', foreground=self.colors['success'])
            self.style.configure('Warning.TButton', foreground=self.colors['warning'])

            # Make the window fullscreen
            self.root.state('zoomed')  # For Windows
            # For Linux/Mac, use:
            # self.root.attributes('-zoomed', True)

            # Set minimum window size
            self.root.minsize(1200, 800)

            # Set application icon if available
            try:
                self.root.iconbitmap('app_icon.ico')  # You'll need to create this icon file
            except:
                pass  # Continue if icon not found

            # Create main containers with consistent padding
            self.main_frame = ttk.Frame(self.root, padding="10 10 10 10")
            self.main_frame.pack(expand=True, fill=tk.BOTH)

            # Create header with title and department filter
            self.header_frame = ttk.Frame(self.main_frame)
            self.header_frame.pack(fill=tk.X, pady=(0, 10))

            # App title
            title_label = ttk.Label(self.header_frame, text="Event Analysis Dashboard", style='Heading.TLabel')
            title_label.pack(side=tk.LEFT, padx=(0, 20))

            # Create department filter frame
            self.create_department_filter()

            # Create navigation bar - store reference to it
            self.create_navigation()

            # Create notebook for tabs with improved styling
            self.tab_control = ttk.Notebook(self.main_frame)
            self.tab_control.pack(expand=True, fill=tk.BOTH)

            # Initialize tabs
            self.output_tab = ttk.Frame(self.tab_control, padding=10)
            self.cluster_tab = ttk.Frame(self.tab_control, padding=10)
            self.rules_tab = ttk.Frame(self.tab_control, padding=10)
            self.descriptive_tab = ttk.Frame(self.tab_control, padding=10)
            self.histogram_tab = ttk.Frame(self.tab_control, padding=10)
            self.recommendations_tab = ttk.Frame(self.tab_control, padding=10)
            self.baseline_tab = ttk.Frame(self.tab_control, padding=10)
            self.cluster_trends_tab = ttk.Frame(self.tab_control, padding=10)  # Add new tab for cluster trends per year

            # Add tabs to notebook
            self.tab_control.add(self.output_tab, text='Analysis Results')
            self.tab_control.add(self.cluster_tab, text='Clustering')
            self.tab_control.add(self.rules_tab, text='Association Rules')
            self.tab_control.add(self.descriptive_tab, text='Descriptive Stats')
            self.tab_control.add(self.histogram_tab, text='Histograms')
            self.tab_control.add(self.recommendations_tab, text='Recommendations')
            self.tab_control.add(self.baseline_tab, text='Baseline Comparisons')
            self.tab_control.add(self.cluster_trends_tab, text='Cluster Trends Per Year')

            # Create scrolled text widgets for output and recommendations
            self.output_text = scrolledtext.ScrolledText(
                self.output_tab,
                height=30,
                font=self.fonts['body'],
                background=self.colors['light_bg'],
                foreground=self.colors['text'],
                padx=10,
                pady=10,
                wrap=tk.WORD
            )
            self.output_text.pack(fill=tk.BOTH, expand=True)

            self.recommendations_text = scrolledtext.ScrolledText(
                self.recommendations_tab,
                height=30,
                font=self.fonts['body'],
                background=self.colors['light_bg'],
                foreground=self.colors['text'],
                padx=10,
                pady=10,
                wrap=tk.WORD
            )
            self.recommendations_text.pack(fill=tk.BOTH, expand=True)

            # Configure text tags for formatting
            self.output_text.tag_configure('heading', font=self.fonts['heading'], foreground=self.colors['primary'])
            self.output_text.tag_configure('subheading', font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.output_text.tag_configure('normal', font=self.fonts['body'])
            self.output_text.tag_configure('success', foreground=self.colors['success'])
            self.output_text.tag_configure('warning', foreground=self.colors['warning'])
            self.output_text.tag_configure('error', foreground=self.colors['danger'])
            self.output_text.tag_configure('highlight', background=self.colors['highlight'])
            self.output_text.tag_configure('code', font=self.fonts['code'], background='#f0f0f0')
            self.output_text.tag_configure('light_text', foreground=self.colors['light_text'])

            self.recommendations_text.tag_configure('heading', font=self.fonts['heading'], foreground=self.colors['primary'])
            self.recommendations_text.tag_configure('subheading', font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.recommendations_text.tag_configure('normal', font=self.fonts['body'])
            self.recommendations_text.tag_configure('success', foreground=self.colors['success'])
            self.recommendations_text.tag_configure('warning', foreground=self.colors['warning'])
            self.recommendations_text.tag_configure('error', foreground=self.colors['danger'])
            self.recommendations_text.tag_configure('highlight', background=self.colors['highlight'])
            self.recommendations_text.tag_configure('code', font=self.fonts['code'], background='#f0f0f0')
            self.recommendations_text.tag_configure('light_text', foreground=self.colors['light_text'])

            print("AnalysisGUI initialized successfully.")

        except Exception as e:
            print(f"Error in AnalysisGUI.__init__: {e}")
            raise  # Re-raise the exception to see the full traceback

    def create_department_filter(self):
        """Create the department filter dropdown"""
        # Create a styled frame for the department filter in the header
        filter_frame = ttk.Frame(self.header_frame)
        filter_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create filter container with subtle border
        filter_container = ttk.Frame(filter_frame, style='Card.TFrame')
        filter_container.pack(side=tk.RIGHT, padx=5, pady=2)

        # Add a filter icon (optional - using Unicode character)
        filter_icon = ttk.Label(filter_container, text="🔍", font=('Segoe UI', 11))
        filter_icon.pack(side=tk.LEFT, padx=(5, 0))

        # Create label with better styling
        dept_label = ttk.Label(filter_container, text="Department:", style='Subheading.TLabel')
        dept_label.pack(side=tk.LEFT, padx=5)

        # Create department dropdown with better styling
        self.department_dropdown = ttk.Combobox(
            filter_container,
            textvariable=self.current_department,
            width=25,
            font=self.fonts['body'],
            state="readonly"  # Prevents users from entering invalid values
        )
        self.department_dropdown.pack(side=tk.LEFT, padx=5)

        # Add a refresh button
        refresh_btn = ttk.Button(
            filter_container,
            text="↻",
            width=3,
            command=self.update_department_list,
            style='Action.TButton'
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)

        # Add tooltip to explain what the refresh button does
        # This is a basic implementation - consider using a proper tooltip library
        def show_refresh_tooltip(event):
            tooltip = tk.Toplevel(self.root)
            tooltip.wm_overrideredirect(True)
            tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
            ttk.Label(tooltip, text="Refresh department list",
                     background=self.colors['highlight'],
                     padding=5).pack()
            tooltip.after(2000, tooltip.destroy)  # Auto-destroy after 2 seconds

        refresh_btn.bind("<Enter>", show_refresh_tooltip)

        # Bind department change event
        self.department_dropdown.bind('<<ComboboxSelected>>', self.on_department_change)

    def run_analysis(self):
        """
        Run the complete analysis workflow on the loaded data.
        """
        try:
            # Check if data is loaded
            if not hasattr(self, 'df') or self.df is None or self.df.empty:
                messagebox.showerror("Error", "No data loaded. Please load data first.")
                return

            # Check if features are selected
            if not self.selected_features:
                messagebox.showerror("Error", "No features selected. Please select features for analysis.")
                return

            # Get filtered data based on department selection
            filtered_df = self.get_filtered_data()

            # Make sure our critical widgets exist and are properly configured
            self.ensure_widgets_exist()

            # Clear the content of text widgets
            self.output_text.delete(1.0, tk.END)
            self.recommendations_text.delete(1.0, tk.END)

            # Clear previous visualizations - careful not to disrupt our text widgets
            for tab in [self.cluster_tab, self.histogram_tab,
                       self.rules_tab, self.descriptive_tab, self.baseline_tab,
                       self.cluster_trends_tab]:  # Removed distribution_tab
                for widget in tab.winfo_children():
                    widget.destroy()

            # Show analysis progress
            self.output_text.insert(tk.END, "Starting analysis...\n")
            self.output_text.insert(tk.END, f"Department filter: {self.current_department.get()}\n")
            self.output_text.insert(tk.END, f"Selected features: {', '.join(self.selected_features)}\n")
            self.output_text.insert(tk.END, f"Datasets loaded: {len(self.datasets)} years ({', '.join(sorted(self.datasets.keys()))})\n\n")
            self.output_text.update()

            # 1. Analyze event ratings
            self.output_text.insert(tk.END, "Analyzing event ratings...\n")
            self.output_text.update()

            avg_scores, needs_improvement, moderately_satisfactory, satisfactory, very_satisfactory = analyze_event_ratings(filtered_df)

            # Display rating analysis results
            self.output_text.insert(tk.END, "\nRating Analysis Results:\n")
            self.output_text.insert(tk.END, interpret_ratings(avg_scores))
            self.output_text.update()

            # 2. Perform clustering
            self.output_text.insert(tk.END, "\nPerforming clustering analysis...\n")
            self.output_text.update()

            try:
                clustered_df, kmeans, cluster_sizes = self.cluster_events(filtered_df, self.selected_features)

                # Display clustering results
                self.output_text.insert(tk.END, "\nClustering Results:\n")
                for label, size in cluster_sizes.items():
                    percentage = (size / len(clustered_df)) * 100
                    self.output_text.insert(tk.END, f"  {label}: {size} samples ({percentage:.1f}%)\n")

                # Plot clusters
                self.plot_clusters(clustered_df, kmeans, "Cluster Analysis", use_pca=True)

                # If multiple years exist, also plot clusters for each year separately
                if len(self.datasets) > 1:
                    self.output_text.insert(tk.END, "\nGenerating clusters for each year...\n")
                    for year, year_df in sorted(self.datasets.items()):
                        # Filter by department if needed
                        if self.current_department.get() != "All Departments":
                            year_df = year_df[year_df['department_name'] == self.current_department.get()]

                        if not year_df.empty:
                            try:
                                # Cluster this year's data
                                year_clustered_df, year_kmeans, year_labels = self.cluster_events(
                                    year_df, self.selected_features, return_labels=True)

                                # Plot clusters for this year
                                self.plot_clusters_for_year(year_clustered_df, year_kmeans,
                                                           year_labels, year, use_pca=True)
                            except Exception as year_e:
                                logging.error(f"Error clustering year {year}: {str(year_e)}")
                                continue

            except Exception as e:
                self.output_text.insert(tk.END, f"Error during clustering: {str(e)}\n")
                logging.error(f"Clustering error: {str(e)}")

            # 3. Generate association rules
            self.output_text.insert(tk.END, "\nGenerating association rules...\n")
            self.output_text.update()

            try:
                # Create a progress indicator for large datasets
                progress_frame = None
                progress_bar = None
                progress_label = None

                if len(filtered_df) > 10000:
                    # Create a progress frame
                    progress_frame = ttk.Frame(self.output_tab)
                    progress_frame.pack(fill=tk.X, padx=10, pady=5)

                    # Add a label
                    progress_label = ttk.Label(progress_frame, text="Processing association rules...")
                    progress_label.pack(side=tk.TOP, pady=2)

                    # Add a progress bar
                    progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
                    progress_bar.pack(fill=tk.X, pady=2)
                    progress_bar.start(10)  # Start the animation

                    # Update the UI
                    self.root.update()

                # Determine if we need to sample based on data size
                sample_size = None
                if len(filtered_df) > 50000:
                    sample_size = 50000  # Cap at 50k rows for very large datasets
                    self.output_text.insert(tk.END, f"Dataset is large ({len(filtered_df)} rows). Sampling to {sample_size} rows for association rule mining.\n")
                elif len(filtered_df) > 20000:
                    sample_size = 0.5  # Use 50% for moderately large datasets
                    self.output_text.insert(tk.END, f"Dataset is moderately large ({len(filtered_df)} rows). Sampling to 50% for association rule mining.\n")

                # Update progress label
                if progress_label:
                    progress_label.config(text="Preparing data for association rules...")
                    self.root.update()

                binary_df = prepare_for_association_rules(filtered_df, self.selected_features, sample_size=sample_size)

                # Add diagnostic information about the binary data
                if binary_df.empty:
                    self.output_text.insert(tk.END, "\nNo valid binary data was created. Check your feature selection.\n")
                else:
                    # Count non-zero values in each column
                    column_counts = binary_df.sum().sort_values(ascending=False)
                    total_rows = len(binary_df)

                    # Display the top 10 most frequent items
                    self.output_text.insert(tk.END, f"\nTop 10 most frequent rating categories (out of {len(column_counts)} total):\n")
                    for col, count in column_counts.head(10).items():
                        percentage = (count / total_rows) * 100
                        self.output_text.insert(tk.END, f"- {col}: {count} occurrences ({percentage:.2f}%)\n")

                    # Check if any columns have sufficient support
                    min_required = 0.05 * total_rows  # 5% minimum support
                    columns_with_support = column_counts[column_counts >= min_required].shape[0]
                    self.output_text.insert(tk.END, f"\n{columns_with_support} out of {len(column_counts)} categories meet the 5% minimum support threshold.\n")

                    if columns_with_support < 2:
                        self.output_text.insert(tk.END, "WARNING: Less than 2 categories meet the minimum support threshold. Try lowering the threshold.\n")

                # Update progress label
                if progress_label:
                    progress_label.config(text="Generating association rules...")
                    self.root.update()

                # Generate rules with optimized parameters for dataset size
                min_support = 0.05
                if len(binary_df) > 50000:
                    min_support = 0.1
                elif len(binary_df) > 20000:
                    min_support = 0.075
                elif len(binary_df) < 1000:
                    # For small datasets, use a lower minimum support
                    min_support = 0.03

                # Use parallel processing for large datasets
                use_parallel = len(binary_df) > 5000

                # Try with progressively lower support thresholds if needed
                support_thresholds = [min_support, min_support/2, min_support/4]
                rules = pd.DataFrame()

                for threshold in support_thresholds:
                    self.output_text.insert(tk.END, f"\nTrying with minimum support: {threshold:.4f}\n")
                    self.output_text.update()

                    rules = generate_association_rules(
                        binary_df,
                        min_support=threshold,
                        min_confidence=0.3,  # Lower confidence threshold to find more rules
                        min_lift=1.0,
                        use_parallel=use_parallel
                        # n_jobs parameter removed as it's not supported
                    )

                    if not rules.empty:
                        self.output_text.insert(tk.END, f"Found {len(rules)} rules with support threshold {threshold:.4f}\n")
                        break
                    else:
                        self.output_text.insert(tk.END, f"No rules found with support threshold {threshold:.4f}\n")

                # Remove progress indicator if it exists
                if progress_frame:
                    progress_bar.stop()
                    progress_frame.destroy()

                if not rules.empty:
                    # Display association rules interpretation
                    self.output_text.insert(tk.END, "\nAssociation Rules Analysis:\n", 'heading')
                    self.output_text.insert(tk.END, interpret_event_association_rules(rules))

                    # Plot association rules
                    self.plot_association_rules(rules, self.current_department.get())
                else:
                    # Clear any previous content in this section
                    self.output_text.insert(tk.END, "\n", 'normal')

                    # Create a visually distinct message for no rules found
                    self.output_text.insert(tk.END, "⚠️ No Significant Association Rules Found\n", 'heading')
                    self.output_text.insert(tk.END, "We attempted to find association rules with multiple thresholds but couldn't find any significant patterns.\n\n", 'normal')

                    # Display possible reasons with better formatting
                    self.output_text.insert(tk.END, "Possible Reasons:\n", 'subheading')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Data is too sparse or has insufficient patterns\n", 'normal')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Selected features don't exhibit strong associations\n", 'normal')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Sample size is too small after filtering\n", 'normal')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Rating distributions are too uniform\n", 'normal')

                    # Add a separator line
                    self.output_text.insert(tk.END, "\n" + "─" * 50 + "\n\n", 'light_text')

                    # Display recommendations with better formatting
                    self.output_text.insert(tk.END, "Suggested Actions:\n", 'subheading')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Select different or more relevant features\n", 'normal')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Include more data or try different departments\n", 'normal')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Check for data quality issues in your dataset\n", 'normal')
                    self.output_text.insert(tk.END, "• ", 'normal')
                    self.output_text.insert(tk.END, "Try the relaxed parameters option below\n", 'normal')

                    # Add a visible separator before the retry section
                    self.output_text.insert(tk.END, "\n", 'normal')

                    # Create a better frame for the retry option
                    retry_frame = ttk.Frame(self.output_tab, style='Card.TFrame')
                    retry_frame.pack(fill=tk.X, padx=20, pady=10)

                    # Add an info label with explanation
                    info_label = ttk.Label(
                        retry_frame,
                        text="Try with extremely relaxed parameters to find any possible patterns",
                        style='Subheading.TLabel',
                        padding=(10, 10, 10, 5)
                    )
                    info_label.pack(fill=tk.X)

                    # Add a descriptive text
                    ttk.Label(
                        retry_frame,
                        text="This will use much lower thresholds for support and confidence,\nwhich may reveal weaker patterns but could include more noise.",
                        padding=(10, 0, 10, 10),
                        foreground=self.colors['light_text']
                    ).pack(fill=tk.X)

                    # Create a button container for better layout
                    button_container = ttk.Frame(retry_frame)
                    button_container.pack(fill=tk.X, padx=10, pady=10)

                    # Create a styled retry button
                    retry_btn = ttk.Button(
                        button_container,
                        text="Try Relaxed Parameters",
                        style='Warning.TButton',
                        command=try_relaxed_parameters,
                        padding=(10, 5)
                    )
                    retry_btn.pack(side=tk.LEFT, padx=5)

                    # Add a cancel button option
                    cancel_btn = ttk.Button(
                        button_container,
                        text="Skip This Step",
                        command=lambda: retry_frame.destroy(),
                        padding=(10, 5)
                    )
                    cancel_btn.pack(side=tk.LEFT, padx=5)

                    # Add a help button with explanation
                    help_btn = ttk.Button(
                        button_container,
                        text="Help",
                        width=8,
                        command=lambda: messagebox.showinfo(
                            "About Relaxed Parameters",
                            "Using relaxed parameters lowers the thresholds for finding patterns.\n\n"
                            "This may help discover weak associations that wouldn't meet the "
                            "normal quality criteria, but be cautious about drawing strong "
                            "conclusions from these results as they may include coincidental "
                            "patterns."
                        )
                    )
                    help_btn.pack(side=tk.RIGHT, padx=5)

                    def try_relaxed_parameters():
                        self.output_text.insert(tk.END, "\nTrying with extremely relaxed parameters...\n", 'normal')
                        self.output_text.update()

                        # Create progress indicator with label
                        progress_frame = ttk.Frame(retry_frame)
                        progress_frame.pack(fill=tk.X, padx=10, pady=5)

                        # Add a progress label
                        progress_label = ttk.Label(
                            progress_frame,
                            text="Processing with lower thresholds...",
                            foreground=self.colors['secondary']
                        )
                        progress_label.pack(fill=tk.X, pady=(0, 5))

                        # Create a better progress bar
                        retry_progress = ttk.Progressbar(
                            progress_frame,
                            mode='indeterminate',
                            length=300
                        )
                        retry_progress.pack(fill=tk.X)
                        retry_progress.start(10)

                        # Update UI to show progress is happening
                        button_container.pack_forget()  # Hide buttons during processing
                        self.root.update()

                        # Try with extremely relaxed parameters
                        relaxed_rules = generate_association_rules(
                            binary_df,
                            min_support=0.01,  # Very low support
                            min_confidence=0.1,  # Very low confidence
                            min_lift=1.01,  # Very low lift
                            max_len=3,
                            use_parallel=use_parallel
                        )

                        # Clean up progress indicators
                        retry_progress.stop()
                        progress_frame.destroy()  # Remove the entire progress frame
                        button_container.pack(fill=tk.X, padx=10, pady=10)  # Restore buttons

                        if not relaxed_rules.empty:
                            self.output_text.insert(tk.END, f"\nFound {len(relaxed_rules)} rules with relaxed parameters.\n")
                            self.output_text.insert(tk.END, "Note: These rules have very low support/confidence and should be interpreted with caution.\n\n")
                            self.output_text.insert(tk.END, interpret_event_association_rules(relaxed_rules))

                            # Plot the rules
                            self.plot_association_rules(relaxed_rules, f"{self.current_department.get()} (Relaxed Parameters)")

                            # Switch to the rules tab
                            self.tab_control.select(self.rules_tab)
                        else:
                            self.output_text.insert(tk.END, "\nStill no rules found even with extremely relaxed parameters.\n")
                            self.output_text.insert(tk.END, "This suggests there may be fundamental issues with the data or feature selection.\n")

                        # Remove the retry button after use
                        retry_button.destroy()

                    retry_button = ttk.Button(retry_frame, text="Try with Extremely Relaxed Parameters", command=try_relaxed_parameters)
                    retry_button.pack(pady=5)

            except Exception as e:
                # Remove progress indicator if it exists in case of error
                if 'progress_frame' in locals() and progress_frame:
                    progress_frame.destroy()

                self.output_text.insert(tk.END, f"Error during association rules analysis: {str(e)}\n")
                logging.error(f"Association rules error: {str(e)}")

            # 4. Generate recommendations
            self.output_text.insert(tk.END, "\nGenerating recommendations...\n")
            self.output_text.update()

            try:
                # Generate standard recommendations based on low scores
                standard_recommendations = generate_event_recommendations(needs_improvement)

                # Generate maintenance recommendations for high scores
                maintenance_recommendations = generate_event_maintenance_recommendations(very_satisfactory)

                # Generate dynamic recommendations
                dynamic_recommendations = self.generate_dynamic_recommendations(
                    filtered_df, needs_improvement, very_satisfactory
                )

                # Display recommendations
                self.recommendations_text.insert(tk.END, "IMPROVEMENT RECOMMENDATIONS:\n\n")

                for feature, recs in standard_recommendations.items():
                    feature_name = feature.replace('_', ' ').title()
                    self.recommendations_text.insert(tk.END, f"{feature_name}:\n")

                    for rec in recs:
                        self.recommendations_text.insert(tk.END, f"• {rec['text']}\n")
                        self.recommendations_text.insert(tk.END, f"  Action: {rec['action']}\n\n")

                # Add dynamic recommendations
                self.recommendations_text.insert(tk.END, "\nDETAILED RECOMMENDATIONS:\n\n")

                for feature, recs in dynamic_recommendations.items():
                    if recs:  # Only show features with recommendations
                        feature_name = feature.replace('_', ' ').title()
                        self.recommendations_text.insert(tk.END, f"{feature_name}:\n")

                        for rec in recs:
                            self.recommendations_text.insert(tk.END, f"• {rec['text']} (Priority: {rec['priority']})\n")
                            self.recommendations_text.insert(tk.END, f"  {rec['action']}\n\n")

                # Add maintenance recommendations
                self.recommendations_text.insert(tk.END, "\nMAINTENANCE RECOMMENDATIONS:\n\n")

                for feature, recs in maintenance_recommendations.items():
                    feature_name = feature.replace('_', ' ').title()
                    self.recommendations_text.insert(tk.END, f"{feature_name}:\n")

                    for rec in recs:
                        self.recommendations_text.insert(tk.END, f"• {rec['text']}\n")
                        self.recommendations_text.insert(tk.END, f"  Action: {rec['action']}\n\n")

            except Exception as e:
                self.recommendations_text.insert(tk.END, f"Error generating recommendations: {str(e)}\n")
                logging.error(f"Recommendations error: {str(e)}")

            # 5. Create descriptive statistics visualizations
            try:
                self.plot_descriptive()
            except Exception as e:
                logging.error(f"Error creating descriptive statistics: {str(e)}")

            # 5.5 Create histogram visualizations - ADD THIS SECTION
            try:
                self.plot_histograms()
            except Exception as e:
                logging.error(f"Error creating histograms: {str(e)}")

            # 5.6 Create recommendations visualizations
            try:
                self.plot_recommendations()
            except Exception as e:
                logging.error(f"Error creating recommendations: {str(e)}")

            # 6. Create distribution visualizations - REMOVED
            # Distribution tab has been removed

            # 7. Create baseline comparisons if multiple years exist
            if len(self.datasets) > 1:
                try:
                    self.output_text.insert(tk.END, "\nGenerating baseline comparisons...\n")
                    self.output_text.update()

                    # Debug information
                    self.output_text.insert(tk.END, f"Available years: {list(self.datasets.keys())}\n")

                    baseline_metrics = self.calculate_baseline_metrics()
                    if baseline_metrics:
                        self.output_text.insert(tk.END, f"Baseline metrics calculated for {len(baseline_metrics)} features\n")

                        comparison_data = self.compare_with_baseline(filtered_df, baseline_metrics)
                        if comparison_data and len(comparison_data) > 0:
                            self.output_text.insert(tk.END, f"Comparison data generated for {len(comparison_data)} features\n")
                            # Store comparison data as an instance attribute for use in recommendations
                            self.baseline_comparison = comparison_data
                            self.plot_baseline_comparison(comparison_data)
                        else:
                            self.output_text.insert(tk.END, "No valid comparison data generated\n")
                            # Clear any previous baseline comparison data
                            self.baseline_comparison = None
                            # Create a simple message in the baseline tab
                            for widget in self.baseline_tab.winfo_children():
                                widget.destroy()
                            label = ttk.Label(self.baseline_tab, text="No valid comparison data could be generated.\nThis may happen if there's insufficient data or no significant changes between years.")
                            label.pack(pady=20)
                    else:
                        self.output_text.insert(tk.END, "No baseline metrics could be calculated\n")
                        # Create a simple message in the baseline tab
                        for widget in self.baseline_tab.winfo_children():
                            widget.destroy()
                        label = ttk.Label(self.baseline_tab, text="No baseline metrics could be calculated.\nPlease ensure you have loaded data for multiple years with consistent features.")
                        label.pack(pady=20)
                except Exception as e:
                    logging.error(f"Error creating baseline comparisons: {str(e)}")
                    self.output_text.insert(tk.END, f"Error in baseline comparison: {str(e)}\n")
                    # Create an error message in the baseline tab
                    for widget in self.baseline_tab.winfo_children():
                        widget.destroy()
                    label = ttk.Label(self.baseline_tab, text=f"Error creating baseline comparisons:\n{str(e)}\n\nPlease check the log for more details.")
                    label.pack(pady=20)
            else:
                # Create a simple message in the baseline tab
                for widget in self.baseline_tab.winfo_children():
                    widget.destroy()
                label = ttk.Label(self.baseline_tab, text="Baseline comparison requires data from multiple years.\nPlease load data for at least two different years to enable this feature.")
                label.pack(pady=20)
                self.output_text.insert(tk.END, "Baseline comparison skipped (requires multiple years of data)\n")

            # 8. Create cluster trends per year if multiple years exist
            if len(self.datasets) > 1:
                try:
                    self.output_text.insert(tk.END, "\nGenerating cluster trends per year...\n")
                    self.output_text.update()
                    self.plot_cluster_trends_per_year()
                except Exception as e:
                    logging.error(f"Error creating cluster trends per year: {str(e)}")
                    self.output_text.insert(tk.END, f"Error in cluster trends per year: {str(e)}\n")
            else:
                # Create a simple message in the cluster trends tab
                for widget in self.cluster_trends_tab.winfo_children():
                    widget.destroy()
                label = ttk.Label(self.cluster_trends_tab, text="Cluster trends per year requires data from multiple years.\nPlease load data for at least two different years to enable this feature.")
                label.pack(pady=20)
                self.output_text.insert(tk.END, "Cluster trends per year skipped (requires multiple years of data)\n")

            # Analysis complete
            self.output_text.insert(tk.END, "\nAnalysis complete!\n")
            self.output_text.see(tk.END)  # Scroll to the end

            # Switch to the Analysis Results tab
            self.tab_control.select(self.output_tab)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
            logging.error(f"Error in run_analysis: {str(e)}")

    def on_department_change(self, event=None):
        """Handle department selection change"""
        if hasattr(self, 'df') and self.df is not None:
            # Ensure widgets exist before running analysis that interacts with them
            self.ensure_widgets_exist()
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
        """Create a navigation bar at the top of the window with permanent button references"""
        # Check if navigation frame already exists and remove it if it does
        if hasattr(self, 'nav_frame') and self.nav_frame is not None:
            try:
                self.nav_frame.destroy()
            except:
                pass

        # Create a new navigation frame with a unique name for easier identification
        self.nav_frame = ttk.Frame(self.main_frame, name="navigation_frame")
        self.nav_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create left frame for main controls
        left_frame = ttk.Frame(self.nav_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X)

        # Clear existing button references
        self.nav_buttons = {}

        # Create and store references to buttons
        self.nav_buttons['load'] = ttk.Button(
            left_frame,
            text="Load Data",
            command=self.load_data,
            width=15
        )
        self.nav_buttons['load'].pack(side=tk.LEFT, padx=2)

        self.nav_buttons['clear'] = ttk.Button(
            left_frame,
            text="Clear Data",
            command=self.clear_datasets,
            width=15
        )
        self.nav_buttons['clear'].pack(side=tk.LEFT, padx=2)

        self.nav_buttons['features'] = ttk.Button(
            left_frame,
            text="Select Features",
            command=self.select_features_window,
            width=15
        )
        self.nav_buttons['features'].pack(side=tk.LEFT, padx=2)

        self.nav_buttons['analyze'] = ttk.Button(
            left_frame,
            text="Run Analysis",
            command=self.run_analysis,
            width=15
        )
        self.nav_buttons['analyze'].pack(side=tk.LEFT, padx=2)

        self.nav_buttons['export'] = ttk.Button(
            left_frame,
            text="Export to PDF",
            command=self.open_pdf_export_window,
            width=15
        )
        self.nav_buttons['export'].pack(side=tk.LEFT, padx=2)

        # Add separator below navigation
        self.separator = ttk.Separator(self.main_frame, orient='horizontal')
        self.separator.pack(fill=tk.X, padx=5, pady=5)

        logging.debug("Navigation bar created with buttons: %s", list(self.nav_buttons.keys()))
        return self.nav_frame

    def ensure_navigation_buttons(self):
        """Make sure all navigation buttons are present and visible"""
        try:
            # Check if our navigation frame exists and has children
            needs_recreation = False

            if not hasattr(self, 'nav_frame') or self.nav_frame is None:
                logging.debug("Navigation frame doesn't exist, will recreate")
                needs_recreation = True
            else:
                try:
                    if not self.nav_frame.winfo_exists():
                        logging.debug("Navigation frame no longer exists, will recreate")
                        needs_recreation = True
                    else:
                        # Check if all buttons exist
                        for btn_name, btn in self.nav_buttons.items():
                            if not btn.winfo_exists():
                                logging.debug(f"Button {btn_name} doesn't exist, will recreate navigation")
                                needs_recreation = True
                                break
                except:
                    needs_recreation = True

            if needs_recreation:
                logging.info("Recreating navigation bar")
                self.create_navigation()
                # Ensure proper layout - update the UI
                self.root.update_idletasks()

            return True
        except Exception as e:
            logging.error(f"Error ensuring navigation buttons: {e}")
            print(f"Error ensuring navigation buttons: {e}")
            # Try to recover by recreating the navigation bar
            try:
                self.create_navigation()
                return True
            except Exception as e2:
                logging.error(f"Failed to recreate navigation bar: {e2}")
                return False

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            # Create a tooltip window
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")

            # Create the tooltip content
            label = ttk.Label(
                tooltip,
                text=text,
                background=self.colors['highlight'],
                foreground=self.colors['text'],
                padding=5,
                wraplength=200
            )
            label.pack()

            # Store the tooltip to destroy it later
            widget.tooltip = tooltip

            # Auto-destroy after some time
            widget.after(2000, lambda: tooltip.destroy() if hasattr(widget, 'tooltip') else None)

        def leave(event):
            # Destroy the tooltip when the mouse leaves
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip

        # Bind events
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def ensure_widgets_exist(self):
        """Helper method to ensure critical widgets exist and are properly configured"""
        # Check if the navigation buttons exist
        self.ensure_navigation_buttons()

        # Check output_text widget
        try:
            if not hasattr(self, 'output_text') or not self.output_text.winfo_exists():
                # Create or recreate output_text
                for widget in self.output_tab.winfo_children():
                    widget.destroy()

                self.output_text = scrolledtext.ScrolledText(
                    self.output_tab,
                    height=30,
                    font=self.fonts['body'],
                    background=self.colors['light_bg'],
                    foreground=self.colors['text'],
                    padx=10,
                    pady=10,
                    wrap=tk.WORD
                )
                self.output_text.pack(fill=tk.BOTH, expand=True)

                # Configure tags
                self.output_text.tag_configure('heading', font=self.fonts['heading'], foreground=self.colors['primary'])
                self.output_text.tag_configure('subheading', font=self.fonts['subheading'], foreground=self.colors['secondary'])
                self.output_text.tag_configure('normal', font=self.fonts['body'])
                self.output_text.tag_configure('success', foreground=self.colors['success'])
                self.output_text.tag_configure('warning', foreground=self.colors['warning'])
                self.output_text.tag_configure('error', foreground=self.colors['danger'])
                self.output_text.tag_configure('highlight', background=self.colors['highlight'])
                self.output_text.tag_configure('code', font=self.fonts['code'], background='#f0f0f0')
                self.output_text.tag_configure('light_text', foreground=self.colors['light_text'])
        except Exception as e:
            logging.error(f"Error ensuring output_text exists: {str(e)}")

        # Check recommendations_text widget
        try:
            if not hasattr(self, 'recommendations_text') or not self.recommendations_text.winfo_exists():
                # Create or recreate recommendations_text
                for widget in self.recommendations_tab.winfo_children():
                    widget.destroy()

                self.recommendations_text = scrolledtext.ScrolledText(
                    self.recommendations_tab,
                    height=30,
                    font=self.fonts['body'],
                    background=self.colors['light_bg'],
                    foreground=self.colors['text'],
                    padx=10,
                    pady=10,
                    wrap=tk.WORD
                )
                self.recommendations_text.pack(fill=tk.BOTH, expand=True)

                # Configure tags
                self.recommendations_text.tag_configure('heading', font=self.fonts['heading'], foreground=self.colors['primary'])
                self.recommendations_text.tag_configure('subheading', font=self.fonts['subheading'], foreground=self.colors['secondary'])
                self.recommendations_text.tag_configure('normal', font=self.fonts['body'])
                self.recommendations_text.tag_configure('success', foreground=self.colors['success'])
                self.recommendations_text.tag_configure('warning', foreground=self.colors['warning'])
                self.recommendations_text.tag_configure('error', foreground=self.colors['danger'])
                self.recommendations_text.tag_configure('highlight', background=self.colors['highlight'])
                self.recommendations_text.tag_configure('code', font=self.fonts['code'], background='#f0f0f0')
                self.recommendations_text.tag_configure('light_text', foreground=self.colors['light_text'])
        except Exception as e:
            logging.error(f"Error ensuring recommendations_text exists: {str(e)}")

        # Check department dropdown
        try:
            if not hasattr(self, 'department_dropdown') or not self.department_dropdown.winfo_exists():
                # Recreate department filter
                self.create_department_filter()
        except Exception as e:
            logging.error(f"Error ensuring department_dropdown exists: {str(e)}")
            try:
                # Try to recreate it
                self.create_department_filter()
            except Exception as e:
                logging.error(f"Failed to recreate department dropdown: {str(e)}")

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

        # Store a reference to the viz_frame for later use
        self.viz_frame = viz_frame

        # The tab_control is already created in __init__,
        # so we'll just make sure it's properly displayed in the main window
        # No need to create it again here

        # Add a note about visualizations
        note_label = ttk.Label(
            viz_frame,
            text="Visualizations will appear in the tabs above when you run the analysis.",
            foreground=self.colors['light_text'],
            wraplength=400,
            justify='center'
        )
        note_label.pack(pady=20)

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

                # CRITICAL: Make sure navigation buttons exist after data loading
                self.root.after(100, self.ensure_widgets_exist)

                # Automatically prompt feature selection if new datasets were loaded
                if new_datasets_loaded:
                    # Feature selection should happen AFTER ensuring widgets exist
                    self.root.after(200, self.select_features_window)

                print("Data loading process completed.")  # Debug print

        except Exception as e:
            print(f"Error in load_data: {str(e)}")  # Debug print
            messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def ensure_navigation_buttons(self):
        """Make sure all navigation buttons are present and visible"""
        try:
            # Check if we need to recreate the navigation bar
            nav_frame = None
            for child in self.root.winfo_children():
                if isinstance(child, ttk.Frame) and child.winfo_name() == "!frame":
                    nav_frame = child
                    break

            # If navigation frame is not found or empty, recreate it
            if nav_frame is None or len(nav_frame.winfo_children()) < 2:
                print("Navigation buttons not found, recreating...")
                # Clear existing navigation if any
                for child in self.root.winfo_children():
                    if isinstance(child, ttk.Frame) and child.winfo_name() == "!frame":
                        child.destroy()

                # Recreate navigation bar
                self.create_navigation()
        except Exception as e:
            print(f"Error ensuring navigation buttons: {e}")
            logging.error(f"Error ensuring navigation buttons: {e}")

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
            # Ensure all widgets exist before showing the window
            self.ensure_widgets_exist()

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
        """
        Plots the association rules as a network graph for a specific year.
        """
        try:
            # Clear previous content in rules tab
            for widget in self.rules_tab.winfo_children():
                widget.destroy()

            # Create a canvas with scrollbar for rules tab
            canvas = tk.Canvas(self.rules_tab)
            scrollbar = ttk.Scrollbar(self.rules_tab, orient="vertical", command=canvas.yview)

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

                year_var = tk.StringVar(value=title_suffix)
                year_options = sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly")
                year_dropdown.pack(side=tk.LEFT, padx=5)

                def on_year_change(event=None):
                    selected_year = year_var.get()
                    df_to_analyze = self.datasets[selected_year]
                    if self.current_department.get() != "All Departments":
                        df_to_analyze = df_to_analyze[df_to_analyze['department_name'] == self.current_department.get()]

                    # Run analysis for the selected year
                    self.output_text.insert(tk.END, f"\nGenerating association rules for {selected_year}...\n")
                    self.output_text.update()

                    try:
                        # Create a progress indicator for large datasets
                        progress_frame = None
                        progress_bar = None

                        if len(df_to_analyze) > 10000:
                            # Create a progress frame in the rules tab
                            progress_frame = ttk.Frame(main_frame)
                            progress_frame.pack(fill=tk.X, padx=10, pady=5)

                            # Add a label
                            progress_label = ttk.Label(progress_frame, text="Processing association rules...")
                            progress_label.pack(side=tk.TOP, pady=2)

                            # Add a progress bar
                            progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
                            progress_bar.pack(fill=tk.X, pady=2)
                            progress_bar.start(10)  # Start the animation

                            # Update the UI
                            self.root.update()

                        # Determine if we need to sample based on data size
                        sample_size = None
                        if len(df_to_analyze) > 50000:
                            sample_size = 50000  # Cap at 50k rows for very large datasets
                            self.output_text.insert(tk.END, f"Dataset is large ({len(df_to_analyze)} rows). Sampling to {sample_size} rows for association rule mining.\n")
                        elif len(df_to_analyze) > 20000:
                            sample_size = 0.5  # Use 50% for moderately large datasets
                            self.output_text.insert(tk.END, f"Dataset is moderately large ({len(df_to_analyze)} rows). Sampling to 50% for association rule mining.\n")

                        # Update progress label if it exists
                        if 'progress_label' in locals() and progress_label:
                            progress_label.config(text="Preparing data for association rules...")
                            self.root.update()

                        # Prepare data for association rules
                        binary_df = prepare_for_association_rules(df_to_analyze, self.selected_features, selected_year, sample_size=sample_size)

                        # Add diagnostic information about the binary data
                        if binary_df.empty:
                            self.output_text.insert(tk.END, f"\nNo valid binary data was created for {selected_year}. Check your feature selection.\n")
                        else:
                            # Count non-zero values in each column
                            column_counts = binary_df.sum().sort_values(ascending=False)
                            total_rows = len(binary_df)

                            # Display the top 5 most frequent items
                            self.output_text.insert(tk.END, f"\nTop 5 most frequent rating categories for {selected_year}:\n")
                            for col, count in column_counts.head(5).items():
                                percentage = (count / total_rows) * 100
                                self.output_text.insert(tk.END, f"- {col}: {count} occurrences ({percentage:.2f}%)\n")

                            # Check if any columns have sufficient support
                            min_required = 0.05 * total_rows  # 5% minimum support
                            columns_with_support = column_counts[column_counts >= min_required].shape[0]
                            self.output_text.insert(tk.END, f"\n{columns_with_support} out of {len(column_counts)} categories meet the 5% minimum support threshold.\n")

                        # Update progress label if it exists
                        if 'progress_label' in locals() and progress_label:
                            progress_label.config(text="Generating association rules...")
                            self.root.update()

                        # Generate rules with optimized parameters for dataset size
                        min_support = 0.05
                        if len(binary_df) > 50000:
                            min_support = 0.1
                        elif len(binary_df) > 20000:
                            min_support = 0.075
                        elif len(binary_df) < 1000:
                            # For small datasets, use a lower minimum support
                            min_support = 0.03

                        # Use parallel processing for large datasets
                        use_parallel = len(binary_df) > 5000

                        # Try with progressively lower support thresholds if needed
                        support_thresholds = [min_support, min_support/2, min_support/4]
                        new_rules = pd.DataFrame()

                        for threshold in support_thresholds:
                            self.output_text.insert(tk.END, f"\nTrying with minimum support: {threshold:.4f} for {selected_year}\n")
                            self.output_text.update()

                            new_rules = generate_association_rules(
                                binary_df,
                                min_support=threshold,
                                min_confidence=0.3,  # Lower confidence threshold to find more rules
                                min_lift=1.0,
                                use_parallel=use_parallel
                                # n_jobs parameter removed as it's not supported
                            )

                            if not new_rules.empty:
                                self.output_text.insert(tk.END, f"Found {len(new_rules)} rules for {selected_year} with support threshold {threshold:.4f}\n")
                                break
                            else:
                                self.output_text.insert(tk.END, f"No rules found for {selected_year} with support threshold {threshold:.4f}\n")

                        # Remove progress indicator if it exists
                        if 'progress_frame' in locals() and progress_frame:
                            progress_bar.stop()
                            progress_frame.destroy()

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

                        # Plot the new rules or show a message if none found
                        if not new_rules.empty:
                            self.plot_association_rules(new_rules, selected_year)
                        else:
                            # Add a message to the main frame
                            message_frame = ttk.Frame(main_frame)
                            message_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                            ttk.Label(message_frame, text=f"No significant association rules found for {selected_year}",
                                     font=('Arial', 12, 'bold')).pack(pady=10)

                            # Add a button to try with very relaxed parameters
                            def try_relaxed_parameters():
                                self.output_text.insert(tk.END, f"\nTrying with extremely relaxed parameters for {selected_year}...\n")
                                self.output_text.update()

                                # Create progress indicator
                                retry_label = ttk.Label(message_frame, text="Processing with relaxed parameters...")
                                retry_label.pack(pady=5)
                                retry_progress = ttk.Progressbar(message_frame, mode='indeterminate')
                                retry_progress.pack(fill=tk.X, padx=20, pady=5)
                                retry_progress.start(10)
                                self.root.update()

                                # Try with extremely relaxed parameters
                                relaxed_rules = generate_association_rules(
                                    binary_df,
                                    min_support=0.01,  # Very low support
                                    min_confidence=0.1,  # Very low confidence
                                    min_lift=0.5,      # Accept even slightly negative correlations
                                    max_len=2,         # Only look for pairs to simplify
                                    use_parallel=True
                                    # n_jobs parameter removed as it's not supported
                                )

                                # Stop progress
                                retry_progress.stop()
                                retry_progress.destroy()
                                retry_label.destroy()

                                if not relaxed_rules.empty:
                                    self.output_text.insert(tk.END, f"\nFound {len(relaxed_rules)} rules for {selected_year} with relaxed parameters.\n")
                                    self.output_text.insert(tk.END, "Note: These rules have very low support/confidence and should be interpreted with caution.\n")

                                    # Clear the message frame
                                    for widget in message_frame.winfo_children():
                                        widget.destroy()

                                    # Plot the rules
                                    self.plot_association_rules(relaxed_rules, f"{selected_year} (Relaxed Parameters)")
                                else:
                                    self.output_text.insert(tk.END, f"\nStill no rules found for {selected_year} even with extremely relaxed parameters.\n")
                                    ttk.Label(message_frame, text="No rules found even with relaxed parameters.",
                                            font=('Arial', 10)).pack(pady=5)

                                # Remove the retry button after use
                                retry_button.destroy()

                            retry_button = ttk.Button(message_frame, text="Try with Extremely Relaxed Parameters",
                                                    command=try_relaxed_parameters)
                            retry_button.pack(pady=10)

                            # Add suggestions
                            suggestions = ttk.Frame(message_frame)
                            suggestions.pack(fill=tk.X, padx=10, pady=10)

                            ttk.Label(suggestions, text="Suggestions:", font=('Arial', 10, 'bold')).pack(anchor='w')
                            ttk.Label(suggestions, text="• Select different or more features").pack(anchor='w')
                            ttk.Label(suggestions, text="• Check for data quality issues").pack(anchor='w')
                            ttk.Label(suggestions, text="• Ensure sufficient data for this year").pack(anchor='w')
                    except Exception as e:
                        # Remove progress indicator if it exists in case of error
                        if 'progress_frame' in locals() and progress_frame:
                            progress_frame.destroy()

                        self.output_text.insert(tk.END, f"Error during association rules analysis: {str(e)}\n")
                        logging.error(f"Association rules error: {str(e)}")

                year_dropdown.bind("<<ComboboxSelected>>", on_year_change)
            else:
                # Create year frame
                year_frame = ttk.Frame(main_frame)
                year_frame.pack(fill=tk.X, padx=5, pady=5)

                # Add year label
                year_label = ttk.Label(year_frame, text=f"Year {title_suffix}", font=('Arial', 12, 'bold'))
                year_label.pack(pady=5)

            if rules.empty:
                fig = plt.Figure(figsize=(12, 8), dpi=100)  # Dynamic sizing
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f'No significant association rules found for {title_suffix}',
                        ha='center', va='center')
                ax.set_axis_off()
            else:
                # Create figure with dynamic sizing
                fig = plt.Figure(figsize=(12, 8), dpi=100)  # Don't set figsize, let it be determined by the container
                ax = fig.add_subplot(111)
                G = nx.Graph()

                # Add nodes and edges with weights
                edge_weights = []
                edge_pairs = []
                edge_colors = []

                # Process rules to create network graph
                for _, rule in rules.iterrows():
                    # Extract antecedents and consequents
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])

                    # Add nodes for antecedents and consequents
                    for item in antecedents:
                        if item not in G:
                            # Extract feature name and category
                            parts = item.rsplit('_', 1)
                            feature = parts[0].replace('_', ' ').title()
                            category = parts[1] if len(parts) > 1 else ''

                            # Set node color based on category
                            if category == 'High':
                                node_color = '#90EE90'  # Light green
                            elif category == 'Low':
                                node_color = '#FFA07A'  # Light salmon
                            else:
                                node_color = '#ADD8E6'  # Light blue

                            G.add_node(item, color=node_color, label=f"{feature}\n({category})")

                    for item in consequents:
                        if item not in G:
                            # Extract feature name and category
                            parts = item.rsplit('_', 1)
                            feature = parts[0].replace('_', ' ').title()
                            category = parts[1] if len(parts) > 1 else ''

                            # Set node color based on category
                            if category == 'High':
                                node_color = '#90EE90'  # Light green
                            elif category == 'Low':
                                node_color = '#FFA07A'  # Light salmon
                            else:
                                node_color = '#ADD8E6'  # Light blue

                            G.add_node(item, color=node_color, label=f"{feature}\n({category})")

                    # Add edges between antecedents and consequents
                    for a_item in antecedents:
                        for c_item in consequents:
                            # Check if edge already exists
                            if not G.has_edge(a_item, c_item):
                                # Add edge with lift as weight
                                G.add_edge(a_item, c_item, weight=rule['lift'])
                                edge_weights.append(rule['lift'])
                                edge_pairs.append((a_item, c_item))

                                # Color edges based on lift
                                if rule['lift'] > 2:
                                    edge_colors.append('#006400')  # Dark green for strong relationships
                                elif rule['lift'] > 1.5:
                                    edge_colors.append('#228B22')  # Forest green for moderate relationships
                                else:
                                    edge_colors.append('#90EE90')  # Light green for weak relationships

                # Only proceed if we have edges
                if edge_weights:
                    # Get node positions using spring layout
                    pos = nx.spring_layout(G, k=0.3, iterations=50)

                    # Get node colors
                    node_colors = [G.nodes[node].get('color', '#ADD8E6') for node in G.nodes()]

                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors, alpha=0.8, ax=ax)

                    # Draw edges with varying width based on lift
                    for i, (u, v) in enumerate(edge_pairs):
                        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_weights[i]*0.5,
                                              alpha=0.7, edge_color=edge_colors[i], ax=ax)

                    # Draw node labels
                    labels = {node: G.nodes[node].get('label', node) for node in G.nodes()}
                    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)

                    # Add title
                    ax.set_title(f"Association Rules Network - Year {title_suffix}")
                else:
                    ax.text(0.5, 0.5, f'No significant relationships found for {title_suffix}',
                            ha='center', va='center')
                    ax.set_axis_off()

            # Remove axis
            ax.set_axis_off()

            # Create canvas for the plot with responsive sizing
            plot_canvas = FigureCanvasTkAgg(fig, master=year_frame)
            plot_canvas.draw()
            plot_widget = plot_canvas.get_tk_widget()
            plot_widget.pack(fill=tk.BOTH, expand=True)

            # Remove toolbar code
            # No toolbar for cleaner UI

            # Configure scroll region
            main_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

            # Function to handle resize events
            def on_rules_resize(event):
                # Only redraw if the event is for our window and the size actually changed
                if event.widget == self.root and (event.width != getattr(self, '_last_width_rules', 0) or
                                                 event.height != getattr(self, '_last_height_rules', 0)):
                    self._last_width_rules = event.width
                    self._last_height_rules = event.height

                    # Get the available width for plots
                    available_width = max(canvas.winfo_width() - 20, 100)  # Subtract padding and ensure minimum width

                    # Resize the plot with minimum size check
                    width_inches = max(available_width/100, 1.0)  # Ensure minimum width of 1 inch
                    height_inches = max((available_width*0.75)/100, 0.75)  # Ensure minimum height

                    fig.set_size_inches(width_inches, height_inches)
                    fig.tight_layout()
                    plot_canvas.draw_idle()

                    # Update scroll region
                    main_frame.update_idletasks()
                    canvas.configure(scrollregion=canvas.bbox("all"))

            # Bind the resize event
            self.root.bind("<Configure>", on_rules_resize)

            # Trigger an initial resize
            self.root.update_idletasks()
            on_rules_resize(type('Event', (), {'widget': self.root, 'width': self.root.winfo_width(),
                                              'height': self.root.winfo_height()})())

            # Configure scroll region when frame changes
            main_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            logging.error(f"Error plotting association rules: {str(e)}")
            messagebox.showerror("Error", f"Error plotting association rules: {str(e)}")

    def plot_descriptive(self):
        # Clear previous content in descriptive tab
        for widget in self.descriptive_tab.winfo_children():
            widget.destroy()

        # Create a canvas with scrollbar for descriptive tab
        canvas = tk.Canvas(self.descriptive_tab, width=1500)  # Increased width from 1200 to 1500
        scrollbar = ttk.Scrollbar(self.descriptive_tab, orient="vertical", command=canvas.yview)

        # Create a frame inside the canvas for the plots
        plot_frame = ttk.Frame(canvas, width=1500)  # Increased width from 1200 to 1500

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

        # Store plot references for resizing
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

            header_label = ttk.Label(header_frame, text=header_text, font=("Arial", 12, "bold"))
            header_label.pack(pady=5)

            # If multiple datasets are loaded, add a dropdown to select which year to view
            # Position the year selection dropdown right after the header
            if len(self.datasets) > 1:
                year_frame = ttk.Frame(plot_frame)
                year_frame.pack(fill=tk.X, padx=10, pady=5)

                ttk.Label(year_frame, text="Select Year:").pack(side=tk.LEFT, padx=5)

                year_var = tk.StringVar(value="Combined")
                year_options = ["Combined"] + sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly", width=15)
                year_dropdown.pack(side=tk.LEFT, padx=5)
                year_dropdown.set("Combined")  # Default to combined view

            # Create summary statistics table frame
            stats_frame = ttk.LabelFrame(plot_frame, text="Summary Statistics")
            stats_frame.pack(fill=tk.X, padx=10, pady=10)

            if len(self.datasets) > 1:
                def on_year_change(event=None):
                    # Clear existing stats
                    for widget in stats_frame.winfo_children():
                        widget.destroy()

                    # Create new stats text widget with increased height
                    stats_text = tk.Text(stats_frame, height=70, width=150, state='disabled')
                    stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                    # Configure the text widget to be non-editable but allow configuration
                    stats_text.config(state='normal')

                    # Get data for selected year
                    selected_year = year_var.get()
                    if selected_year == "Combined":
                        df_to_analyze = filtered_df
                        title_text = "Combined Data"
                    else:
                        df_to_analyze = self.datasets[selected_year]
                        if self.current_department.get() != "All Departments":
                            df_to_analyze = df_to_analyze[df_to_analyze['department_name'] == self.current_department.get()]
                        title_text = f"Year {selected_year}"

                    # Create a copy of the dataframe with only numeric columns
                    numeric_df = df_to_analyze.copy()
                    if 'department_name' in numeric_df.columns:
                        numeric_df = numeric_df.drop('department_name', axis=1)

                    # Convert all columns to numeric
                    for col in numeric_df.columns:
                        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

                    # Calculate summary statistics
                    summary_stats = numeric_df.describe().T
                    summary_stats['range'] = summary_stats['max'] - summary_stats['min']
                    summary_stats['cv'] = summary_stats['std'] / summary_stats['mean'] * 100  # Coefficient of variation

                    # Format and display the statistics
                    stats_text.insert(tk.END, f"Feature Statistics for {title_text}:\n\n")

                    # Add description of statistical measures
                    stats_text.insert(tk.END, "Statistical Measures Explanation:\n")
                    stats_text.insert(tk.END, "  Mean: The average value of all ratings for this feature\n")
                    stats_text.insert(tk.END, "  Median: The middle value when all ratings are arranged in order\n")
                    stats_text.insert(tk.END, "  Std Dev: Standard deviation - measures how spread out the ratings are\n")
                    stats_text.insert(tk.END, "  Range: The difference between the highest and lowest rating\n")
                    stats_text.insert(tk.END, "  CV: Coefficient of Variation - relative variability as a percentage of the mean\n\n")
                    stats_text.insert(tk.END, "Feature Details:\n\n")

                    for feature, row in summary_stats.iterrows():
                        feature_name = feature.replace('_', ' ').title()
                        stats_text.insert(tk.END, f"{feature_name}:\n")
                        stats_text.insert(tk.END, f"  Mean: {row['mean']:.2f}\n")
                        stats_text.insert(tk.END, f"  Median: {row['50%']:.2f}\n")
                        stats_text.insert(tk.END, f"  Std Dev: {row['std']:.2f}\n")
                        stats_text.insert(tk.END, f"  Range: {row['range']:.2f}\n")
                        stats_text.insert(tk.END, f"  CV: {row['cv']:.2f}%\n\n")

                    # Make text widget read-only
                    stats_text.config(state='disabled')

                year_dropdown.bind('<<ComboboxSelected>>', on_year_change)

                # Trigger the dropdown callback to show combined stats initially
                on_year_change()
            else:
                # Create a text widget for the statistics with increased height
                stats_text = tk.Text(stats_frame, height=30, width=150, state='disabled')
                stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # Configure the text widget to be non-editable but allow configuration
                stats_text.config(state='normal')

                # Create a copy of the dataframe with only numeric columns
                numeric_df = filtered_df.copy()
                if 'department_name' in numeric_df.columns:
                    numeric_df = numeric_df.drop('department_name', axis=1)

                # Convert all columns to numeric
                for col in numeric_df.columns:
                    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

                # Calculate summary statistics
                summary_stats = numeric_df.describe().T
                summary_stats['range'] = summary_stats['max'] - summary_stats['min']
                summary_stats['cv'] = summary_stats['std'] / summary_stats['mean'] * 100  # Coefficient of variation

                # Format and display the statistics
                year = next(iter(self.datasets.keys()))
                stats_text.insert(tk.END, f"Feature Statistics for Year {year}:\n\n")

                # Add description of statistical measures
                stats_text.insert(tk.END, "Statistical Measures Explanation:\n")
                stats_text.insert(tk.END, "  Mean: The average value of all ratings for this feature\n")
                stats_text.insert(tk.END, "  Median: The middle value when all ratings are arranged in order\n")
                stats_text.insert(tk.END, "  Std Dev: Standard deviation - measures how spread out the ratings are\n")
                stats_text.insert(tk.END, "  Range: The difference between the highest and lowest rating\n")
                stats_text.insert(tk.END, "  CV: Coefficient of Variation - relative variability as a percentage of the mean\n\n")
                stats_text.insert(tk.END, "Feature Details:\n\n")

                for feature, row in summary_stats.iterrows():
                    feature_name = feature.replace('_', ' ').title()
                    stats_text.insert(tk.END, f"{feature_name}:\n")
                    stats_text.insert(tk.END, f"  Mean: {row['mean']:.2f}\n")
                    stats_text.insert(tk.END, f"  Median: {row['50%']:.2f}\n")
                    stats_text.insert(tk.END, f"  Std Dev: {row['std']:.2f}\n")
                    stats_text.insert(tk.END, f"  Range: {row['range']:.2f}\n")
                    stats_text.insert(tk.END, f"  CV: {row['cv']:.2f}%\n\n")

                # Make text widget read-only
                stats_text.config(state='disabled')

            # Configure scroll region when plot frame changes
            def configure_scroll_region(event):
                canvas.configure(scrollregion=canvas.bbox("all"))

            plot_frame.bind('<Configure>', configure_scroll_region)

            # Function to handle resize events
            def on_descriptive_resize(event):
                # Only redraw if the event is for our window and the size actually changed
                if event.widget == self.root and (event.width != getattr(self, '_last_width_desc', 0) or
                                                 event.height != getattr(self, '_last_height_desc', 0)):
                    self._last_width_desc = event.width
                    self._last_height_desc = event.height

            # Bind the resize event
            self.root.bind("<Configure>", on_descriptive_resize)

            # Trigger an initial resize
            self.root.update_idletasks()
            on_descriptive_resize(type('Event', (), {'widget': self.root, 'width': self.root.winfo_width(),
                                                    'height': self.root.winfo_height()})())

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            logging.error(f"Error in plot_descriptive: {str(e)}")
            label = ttk.Label(plot_frame, text=f"Error during descriptive analysis: {str(e)}")
            label.pack(pady=20)

    def plot_clusters_for_year(self, df, kmeans, labels, year, use_pca=True):
        try:
            # Clear previous content in cluster tab if this is the first plot
            if not hasattr(self, 'cluster_canvas'):
                for widget in self.cluster_tab.winfo_children():
                    widget.destroy()

                # Create a canvas with scrollbar for cluster tab
                self.cluster_canvas = tk.Canvas(self.cluster_tab)
                self.cluster_scrollbar = ttk.Scrollbar(self.cluster_tab, orient="vertical", command=self.cluster_canvas.yview)

                # Create a frame inside the canvas for the plots
                self.cluster_plot_frame = ttk.Frame(self.cluster_canvas)

                # Configure scrolling
                self.cluster_canvas.configure(yscrollcommand=self.cluster_scrollbar.set)
                self.cluster_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.cluster_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                # Create window in canvas
                self.cluster_canvas.create_window((0, 0), window=self.cluster_plot_frame, anchor="nw")

                # Configure scroll region when plot frame changes
                self.cluster_plot_frame.bind('<Configure>',
                    lambda e: self.cluster_canvas.configure(scrollregion=self.cluster_canvas.bbox("all")))

                # Store plot references for resizing
                self.plot_references = []

                # Add a control panel at the top with a back button and year selector
                control_frame = ttk.Frame(self.cluster_plot_frame)
                control_frame.pack(fill=tk.X, padx=5, pady=5)

                back_button = ttk.Button(control_frame, text="Back to Main View",
                                        command=lambda: self.plot_clusters(df, kmeans, "Cluster Analysis", use_pca))
                back_button.pack(side=tk.LEFT, padx=5)

                # If multiple datasets are loaded, add a dropdown to select which year to view
                if len(self.datasets) > 1:
                    ttk.Label(control_frame, text="Select Year:").pack(side=tk.LEFT, padx=5)

                    # Set the initial value based on the year parameter
                    year_var = tk.StringVar(value=str(year))
                    year_options = ["Combined"] + sorted(self.datasets.keys())
                    year_dropdown = ttk.Combobox(control_frame, textvariable=year_var, values=year_options, state="readonly")
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

                        # Clear the entire cluster tab and reset attributes
                        for widget in self.cluster_tab.winfo_children():
                            widget.destroy()

                        if hasattr(self, 'cluster_canvas'):
                            delattr(self, 'cluster_canvas')
                        if hasattr(self, 'plot_references'):
                            delattr(self, 'plot_references')
                        if hasattr(self, '_resize_bound_year'):
                            delattr(self, '_resize_bound_year')

                        # Create the plot with the new data
                        if selected_year == "Combined":
                            self.plot_clusters(clustered_df, kmeans_model, "Cluster Analysis", use_pca)
                        else:
                            self.plot_clusters_for_year(clustered_df, kmeans_model, cluster_labels, selected_year, use_pca)

                    year_dropdown.bind("<<ComboboxSelected>>", on_year_change)

            # Create new figure with dynamic sizing
            fig = plt.Figure(dpi=100)  # Don't set figsize, let it be determined by the container
            ax = fig.add_subplot(111)

            if kmeans is None:
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
                # Get feature data
                feature_data = df[self.selected_features].copy()

                # Handle outliers by capping extreme values (optional)
                for col in feature_data.columns:
                    q1 = feature_data[col].quantile(0.05)
                    q3 = feature_data[col].quantile(0.95)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    feature_data[col] = feature_data[col].clip(lower_bound, upper_bound)

                # Standardize the data
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

            # Create frame for this year's plot
            year_frame = ttk.Frame(self.cluster_plot_frame)
            year_frame.pack(fill=tk.X, padx=5, pady=5)

            # Add year label
            if year == "Combined":
                year_label = ttk.Label(year_frame, text="Combined Data", font=('Arial', 12, 'bold'))
            else:
                year_label = ttk.Label(year_frame, text=f"Year {year}", font=('Arial', 12, 'bold'))
            year_label.pack(pady=5)

            # Create canvas for this year's plot with responsive sizing
            canvas = FigureCanvasTkAgg(fig, master=year_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.X, expand=True)

            # Remove toolbar code
            # No toolbar for cleaner UI

            # Store references for resizing
            self.plot_references.append((fig, canvas))

            # Add separator
            ttk.Separator(self.cluster_plot_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)

            # Update scroll region
            self.cluster_plot_frame.update_idletasks()
            self.cluster_canvas.configure(scrollregion=self.cluster_canvas.bbox("all"))

            # Function to handle resize events for all plots
            def on_resize(event):
                # Only redraw if the event is for our window and the size actually changed
                if event.widget == self.root and (event.width != getattr(self, '_last_width_year', 0) or
                                                 event.height != getattr(self, '_last_height_year', 0)):
                    self._last_width_year = event.width
                    self._last_height_year = event.height

                    # Get the available width for plots
                    available_width = self.cluster_canvas.winfo_width() - 20  # Subtract padding

                    # Resize all plots
                    for fig_ref, canvas_ref in self.plot_references:
                        # Set width based on available space, height proportionally
                        fig_ref.set_size_inches(available_width/100, (available_width*0.6)/100)
                        fig_ref.tight_layout()
                        canvas_ref.draw_idle()

            # Bind the resize event if not already bound
            if not hasattr(self, '_resize_bound_year') or not self._resize_bound_year:
                self.root.bind("<Configure>", on_resize)
                self._resize_bound_year = True

            # Trigger an initial resize
            self.root.update_idletasks()
            on_resize(type('Event', (), {'widget': self.root, 'width': self.root.winfo_width(),
                                        'height': self.root.winfo_height()})())

        except Exception as e:
            logging.error(f"Error plotting clusters: {e}")
            raise

    def get_cluster_color(self, label):
        # Color mapping for clusters
        colors = {
            'Needs Improvement': 'red',
            'Moderately Satisfactory': 'orange',
            'Satisfactory': 'lightgreen',
            'Very Satisfactory': 'darkgreen'
        }
        return colors.get(label, 'gray')

    def cluster_events(self, filtered_df, selected_features, return_labels=False):
        """
        Perform clustering analysis on the filtered data.

        Parameters:
        - filtered_df: DataFrame with filtered data
        - selected_features: List of features to use for clustering
        - return_labels: If True, return the cluster labels as well

        Returns:
        - clustered_df: DataFrame with cluster assignments
        - kmeans: KMeans model (or None if simple categorization was used)
        - cluster_sizes: Dictionary of cluster sizes
        - labels: Cluster labels (only if return_labels=True)
        """
        try:
            # Filter out non-numeric features
            numeric_features = [f for f in selected_features if f != 'department_name']
            if not numeric_features:
                raise ValueError("No numeric features selected for clustering")

            # Check if we have enough samples for meaningful clustering
            min_samples_required = 8  # Minimum 2 samples per cluster for 4 clusters
            if len(filtered_df) < min_samples_required:
                logging.warning(f"Insufficient samples ({len(filtered_df)}) for clustering. Minimum required: {min_samples_required}")
                # Return simple categorization based on mean ratings
                simple_df = filtered_df.copy()
                mean_ratings = simple_df[numeric_features].mean(axis=1)
                simple_df['cluster'] = pd.cut(mean_ratings,
                    bins=[-float('inf'), 0.74, 1.49, 2.24, float('inf')],
                    labels=[0, 1, 2, 3])
                simple_df['cluster_label'] = simple_df['cluster'].map({
                    0: 'Needs Improvement',
                    1: 'Moderately Satisfactory',
                    2: 'Satisfactory',
                    3: 'Very Satisfactory'
                })
                cluster_sizes = simple_df['cluster_label'].value_counts().to_dict()

                if return_labels:
                    return simple_df, None, cluster_sizes, simple_df['cluster_label'].tolist()
                else:
                    return simple_df, None, cluster_sizes

            # Prepare data for clustering
            cluster_data = filtered_df[numeric_features].copy()

            # Convert to numeric and handle missing values
            for feature in numeric_features:
                cluster_data[feature] = pd.to_numeric(cluster_data[feature], errors='coerce')

            # Handle missing values with mean imputation
            cluster_data = cluster_data.fillna(cluster_data.mean())

            # Normalize data to 0-1 scale
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            # Initialize KMeans with optimal parameters
            kmeans = KMeans(
                n_clusters=4,  # Fixed for our rating categories
                init='k-means++',  # Better initialization
                n_init=20,  # Increased from 10 for better results
                max_iter=500,  # Increased from 300 for better convergence
                tol=1e-6,  # Added tighter tolerance for convergence
                random_state=42
            )

            # Fit and predict clusters
            cluster_assignments = kmeans.fit_predict(scaled_data)

            # Calculate cluster centers in original scale
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            center_means = centers.mean(axis=1)

            # Sort clusters by mean ratings
            cluster_order = np.argsort(center_means)

            # Map clusters to satisfaction levels
            cluster_mapping = {
                cluster_order[0]: 'Needs Improvement',
                cluster_order[1]: 'Moderately Satisfactory',
                cluster_order[2]: 'Satisfactory',
                cluster_order[3]: 'Very Satisfactory'
            }

            # Create output dataframe with cluster assignments
            clustered_df = filtered_df.copy()
            clustered_df['cluster'] = cluster_assignments
            clustered_df['cluster_label'] = clustered_df['cluster'].map(cluster_mapping)

            # Calculate cluster sizes
            cluster_sizes = clustered_df['cluster_label'].value_counts().to_dict()

            # Calculate silhouette score only if we have enough samples in each cluster
            min_cluster_size = min(cluster_sizes.values()) if cluster_sizes else 0
            if min_cluster_size >= 2:
                silhouette_avg = silhouette_score(scaled_data, cluster_assignments)
                logging.info(f"Silhouette Score: {silhouette_avg:.3f}")
            else:
                logging.warning("Insufficient samples per cluster for silhouette score calculation")

            # Log cluster statistics
            total_samples = len(filtered_df)
            for label, size in cluster_sizes.items():
                percentage = (size / total_samples) * 100
                logging.info(f"Cluster '{label}': {size} samples ({percentage:.1f}%)")

            if return_labels:
                return clustered_df, kmeans, cluster_sizes, clustered_df['cluster_label'].tolist()
            else:
                return clustered_df, kmeans, cluster_sizes

        except Exception as e:
            logging.error(f"Error in cluster_events: {str(e)}")
            raise

    def plot_clustering_trends(self, trend_data):
        """Plot trends of cluster distributions over years"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            years = sorted(trend_data.keys())
            categories = ['Needs Improvement', 'Moderately Satisfactory',
                         'Satisfactory', 'Very Satisfactory']
            colors = {
                'Needs Improvement': '#FF0000',
                'Moderately Satisfactory': '#FFA500',
                'Satisfactory': '#90EE90',
                'Very Satisfactory': '#00FF00'
            }

            # Plot lines for all 4 categories
            for category in categories:
                percentages = [trend_data[year][category] for year in years]
                ax.plot(years, percentages, marker='o',
                       label=category, linewidth=2,
                       markersize=8, color=colors[category])

            ax.set_xlabel('Year')
            ax.set_ylabel('Percentage of Attendees')
            ax.set_title('Rating Distribution Trends Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add value labels
            for category in categories:
                percentages = [trend_data[year][category] for year in years]
                for x, y in zip(years, percentages):
                    ax.annotate(f'{y:.1f}%',
                               (x, y),
                               textcoords="offset points",
                               xytext=(0,10),
                               ha='center',
                               color=colors[category])

            plt.tight_layout()

            # Create new tab for trends if it doesn't exist
            if not hasattr(self, 'trends_tab'):
                self.trends_tab = ttk.Frame(self.tab_control)
                self.tab_control.add(self.trends_tab, text='Clustering Trends')
            else:
                # Clear existing content
                for widget in self.trends_tab.winfo_children():
                    widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.trends_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            logging.error(f"Error in plot_clustering_trends: {e}")
            raise

    def validate_parameters(self):
        """Validate against fixed rating scale"""
        # Fixed scale is always valid
        return True

    def generate_dynamic_recommendations(self, df, low_scores_combined, high_scores_combined):
        """Generate comprehensive recommendations based on statistical analysis of features"""
        recommendations = {}

        try:
            # Skip department_name column
            if 'department_name' in df.columns:
                df = df.drop('department_name', axis=1)

            # Use fixed thresholds from rating scale
            low_threshold = self.RATING_SCALE['thresholds']['needs_improvement']
            high_threshold = self.RATING_SCALE['thresholds']['satisfactory']

            for feature in df.columns:
                try:
                    # Convert to numeric, force errors to NaN
                    feature_data = pd.to_numeric(df[feature], errors='coerce')
                    avg_rating = feature_data.mean()
                    std_dev = feature_data.std()

                    # Skip if feature has no valid numeric data
                    if pd.isna(avg_rating) or pd.isna(std_dev):
                        continue

                    # Calculate percentages for different rating categories
                    low_percent = (feature_data < low_threshold).mean() * 100
                    moderate_percent = ((feature_data >= low_threshold) &
                                     (feature_data < high_threshold)).mean() * 100
                    high_percent = (feature_data >= high_threshold).mean() * 100

                    # Initialize recommendations list for this feature
                    recommendations[feature] = []

                    if feature in low_scores_combined.index:
                        # Critical Improvement Recommendations
                        rec = {
                            'text': f"Critical Enhancement Required: {feature.replace('_', ' ').title()}",
                            'priority': 'High' if low_percent > 30 else 'Medium',
                            'current_status': {
                                'average_rating': f"{avg_rating:.2f}/{self.RATING_SCALE['max']}",
                                'low_ratings_percentage': f"{low_percent:.1f}%",
                                'variability': f"±{std_dev:.2f}"
                            },
                            'timeline': {
                                'immediate': '0-2 weeks',
                                'short_term': '1-3 months',
                                'long_term': '3-6 months'
                            },
                            'metrics_to_track': [
                                'Weekly satisfaction ratings',
                                'Improvement implementation rate',
                                'Student feedback completion rate',
                                'Issue resolution time'
                            ]
                        }

                        # Build comprehensive action plan
                        actions = [
                            "Immediate Actions (0-2 weeks):",
                            f"• Conduct emergency review of {feature.replace('_', ' ').lower()} processes",
                            "• Schedule stakeholder meeting to address concerns",
                            "• Implement daily monitoring system",
                            "• Create rapid response team",

                            "\nShort-term Strategy (1-3 months):",
                            "• Develop comprehensive improvement plan:",
                            f"  - Analyze root causes of low {feature.replace('_', ' ').lower()} ratings",
                            "  - Identify resource requirements",
                            "  - Set measurable improvement targets",
                            "  - Create timeline for implementations",

                            "\nLong-term Initiatives (3-6 months):",
                            "• Structural Improvements:",
                            "  - Redesign service delivery model",
                            "  - Implement automated monitoring systems",
                            "  - Develop staff training programs",
                            "  - Create sustainability measures",

                            "\nStakeholder Engagement:",
                            "• Student Involvement:",
                            "  - Conduct focus group discussions",
                            "  - Implement suggestion system",
                            "  - Create student advisory panel",

                            "\nResource Allocation:",
                            "• Required Resources:",
                            "  - Additional staff training",
                            "  - Technology upgrades",
                            "  - Process improvement tools",

                            "\nMonitoring and Evaluation:",
                            "• Key Performance Indicators:",
                            f"  - Target: Reduce low ratings by 50% within 3 months",
                            "  - Weekly progress reviews",
                            "  - Monthly stakeholder updates",
                            "  - Quarterly comprehensive assessments"
                        ]

                        # Add specific recommendations based on variability
                        if std_dev > 0.5:
                            actions.extend([
                                "\nVariability Management:",
                                "• Standardization Initiatives:",
                                "  - Document standard operating procedures",
                                "  - Implement quality control checkpoints",
                                "  - Create service delivery guidelines",
                                "  - Regular staff training sessions"
                            ])

                        rec['action'] = "\n".join(actions)
                        recommendations[feature].append(rec)

                    elif feature in high_scores_combined.index:
                        # Excellence Maintenance Recommendations
                        rec = {
                            'text': f"Maintain Excellence: {feature.replace('_', ' ').title()}",
                            'priority': 'Maintain',
                            'current_status': {
                                'average_rating': f"{avg_rating:.2f}/{self.RATING_SCALE['max']}",
                                'high_ratings_percentage': f"{high_percent:.1f}%",
                                'consistency_level': 'High' if std_dev < 0.3 else 'Moderate'
                            },
                            'timeline': {
                                'ongoing': 'Continuous',
                                'review': 'Monthly',
                                'update': 'Quarterly'
                            },
                            'metrics_to_track': [
                                'Satisfaction rating stability',
                                'Best practice implementation rate',
                                'Innovation metrics',
                                'Staff performance indicators'
                            ]
                        }

                        actions = [
                            "Excellence Maintenance Strategy:",
                            f"• Current Performance: {high_percent:.1f}% high ratings",

                            "\nBest Practice Documentation:",
                            "• Create comprehensive documentation:",
                            "  - Standard operating procedures",
                            "  - Success case studies",
                            "  - Staff training materials",
                            "  - Quality assurance guidelines",

                            "\nContinuous Improvement:",
                            "• Innovation Initiatives:",
                            "  - Regular service reviews",
                            "  - Technology integration assessment",
                            "  - Process optimization studies",
                            "  - Student feedback integration",

                            "\nKnowledge Management:",
                            "• Best Practice Sharing:",
                            "  - Create knowledge repository",
                            "  - Implement mentoring program",
                            "  - Regular team sharing sessions",
                            "  - Cross-department collaboration",

                            "\nQuality Assurance:",
                            "• Monitoring Systems:",
                            "  - Regular quality audits",
                            "  - Performance metrics tracking",
                            "  - Feedback analysis system",
                            "  - Early warning indicators",

                            "\nStakeholder Engagement:",
                            "• Communication Strategy:",
                            "  - Regular success sharing",
                            "  - Stakeholder updates",
                            "  - Recognition programs",
                            "  - Community engagement"
                        ]

                        if std_dev < 0.3:
                            actions.extend([
                                "\nConsistency Maintenance:",
                                "• Excellence Standardization:",
                                "  - Document current practices",
                                "  - Create training modules",
                                "  - Implement quality metrics",
                                "  - Regular staff assessments"
                            ])

                        rec['action'] = "\n".join(actions)
                        recommendations[feature].append(rec)

                    else:
                        # Optimization Recommendations for Moderate Performers
                        rec = {
                            'text': f"Strategic Enhancement: {feature.replace('_', ' ').title()}",
                            'priority': 'Medium',
                            'current_status': {
                                'average_rating': f"{avg_rating:.2f}/{self.RATING_SCALE['max']}",
                                'moderate_ratings_percentage': f"{moderate_percent:.1f}%",
                                'improvement_potential': 'Significant'
                            },
                            'timeline': {
                                'assessment': '2-4 weeks',
                                'implementation': '2-3 months',
                                'review': 'Monthly'
                            },
                            'metrics_to_track': [
                                'Rating improvement rate',
                                'Process efficiency metrics',
                                'Student satisfaction trends',
                                'Implementation effectiveness'
                            ]
                        }

                        actions = [
                            "Performance Optimization Strategy:",
                            f"• Current Status: {moderate_percent:.1f}% moderate ratings",

                            "\nAssessment Phase (2-4 weeks):",
                            "• Comprehensive Analysis:",
                            "  - Current performance review",
                            "  - Gap analysis",
                            "  - Resource assessment",
                            "  - Stakeholder feedback",

                            "\nEnhancement Planning:",
                            "• Strategic Initiatives:",
                            "  - Process optimization",
                            "  - Service quality enhancement",
                            "  - Staff development program",
                            "  - Technology integration",

                            "\nImplementation Strategy:",
                            "• Phased Approach:",
                            "  - Quick wins identification",
                            "  - Pilot programs",
                            "  - Scaled implementation",
                            "  - Progress monitoring",

                            "\nResource Optimization:",
                            "• Efficiency Measures:",
                            "  - Process streamlining",
                            "  - Resource allocation review",
                            "  - Technology utilization",
                            "  - Staff training needs",

                            "\nStakeholder Management:",
                            "• Engagement Plan:",
                            "  - Regular updates",
                            "  - Feedback collection",
                            "  - Progress reporting",
                            "  - Collaboration initiatives"
                        ]

                        if std_dev > 0.4:
                            actions.extend([
                                "\nConsistency Improvement:",
                                "• Standardization Plan:",
                                "  - Service standards development",
                                "  - Quality control measures",
                                "  - Performance monitoring",
                                "  - Regular staff training"
                            ])

                        rec['action'] = "\n".join(actions)
                        recommendations[feature].append(rec)

                except Exception as e:
                    logging.error(f"Error processing feature {feature}: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error in generate_dynamic_recommendations: {str(e)}")

        return recommendations

    def calculate_baseline_metrics(self):
        """Calculate baseline metrics from the first year's data"""
        try:
            if not self.datasets:
                return None

            # Get the earliest year's data as baseline
            baseline_year = min(self.datasets.keys())
            baseline_data = self.datasets[baseline_year]

            baseline_metrics = {}
            for feature in self.selected_features:
                if feature == 'department_name':
                    continue

                # Convert to numeric and handle missing values
                feature_data = pd.to_numeric(baseline_data[feature], errors='coerce')

                # Skip if no valid data
                if feature_data.isna().all():
                    continue

                baseline_metrics[feature] = {
                    'mean': feature_data.mean(),
                    'std': feature_data.std() if len(feature_data.dropna()) > 1 else 0.0001,  # Avoid zero std
                    'percentiles': {
                        'low': feature_data.quantile(0.25),
                        'median': feature_data.median(),
                        'high': feature_data.quantile(0.75)
                    }
                }

            return baseline_metrics
        except Exception as e:
            logging.error(f"Error calculating baseline metrics: {str(e)}")
            return None

    def compare_with_baseline(self, current_data, baseline_metrics):
        """Compare current data with baseline metrics"""
        try:
            comparison = {}
            for feature in self.selected_features:
                if feature == 'department_name':
                    continue

                if feature not in baseline_metrics:
                    continue

                if feature not in current_data.columns:
                    continue

                # Convert to numeric and handle missing values
                current_feature_data = pd.to_numeric(current_data[feature], errors='coerce')

                # Skip if no valid data
                if current_feature_data.isna().all():
                    continue

                current_mean = current_feature_data.mean()
                baseline_mean = baseline_metrics[feature]['mean']
                baseline_std = baseline_metrics[feature]['std']

                # Skip if baseline std is zero to avoid division by zero
                if baseline_std == 0:
                    continue

                # Calculate z-score
                z_score = (current_mean - baseline_mean) / baseline_std

                # Calculate percentage change
                pct_change = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0

                comparison[feature] = {
                    'current_mean': current_mean,
                    'baseline_mean': baseline_mean,
                    'pct_change': pct_change,
                    'z_score': z_score,
                    'significant': abs(z_score) > 1.96  # 95% confidence level
                }

            return comparison
        except Exception as e:
            logging.error(f"Error comparing with baseline: {str(e)}")
            return {}

    def plot_baseline_comparison(self, comparison_data):
        """Plot baseline comparison data"""
        try:
            # Clear existing content
            for widget in self.baseline_tab.winfo_children():
                widget.destroy()

            if not comparison_data or len(comparison_data) == 0:
                label = ttk.Label(self.baseline_tab, text="No baseline comparison data available.")
                label.pack(pady=20)
                return

            # Create a tab control for single year vs multi-year views
            tab_control = ttk.Notebook(self.baseline_tab)
            tab_control.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Create tabs
            single_year_tab = ttk.Frame(tab_control)
            multi_year_tab = ttk.Frame(tab_control)

            tab_control.add(single_year_tab, text="Current vs Baseline")
            tab_control.add(multi_year_tab, text="All Years")

            # Single year comparison (existing functionality)
            self._plot_single_year_baseline(single_year_tab, comparison_data)

            # Multi-year comparison (new functionality)
            self._plot_multi_year_baseline(multi_year_tab)

        except Exception as e:
            logging.error(f"Error in plot_baseline_comparison: {str(e)}")
            for widget in self.baseline_tab.winfo_children():
                widget.destroy()
            error_label = ttk.Label(self.baseline_tab, text=f"Error plotting baseline comparison: {str(e)}")
            error_label.pack(pady=20)

    def _plot_single_year_baseline(self, parent_frame, comparison_data):
        """Plot single year vs baseline comparison (original functionality)"""
        try:
            # Get features from comparison data
            features = list(comparison_data.keys())

            # Create a simple frame for the visualization
            main_frame = ttk.Frame(parent_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Add a title
            title_label = ttk.Label(main_frame, text="Baseline Comparison Analysis", font=("Arial", 14, "bold"))
            title_label.pack(pady=10)

            # Create a simple table for the comparison data
            table_frame = ttk.Frame(main_frame)
            table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            # Add headers
            headers = ["Feature", "Baseline", "Current", "Change (%)", "Significant"]
            for i, header in enumerate(headers):
                label = ttk.Label(table_frame, text=header, font=("Arial", 10, "bold"))
                label.grid(row=0, column=i, padx=5, pady=5, sticky="w")

            # Add data rows
            for i, feature in enumerate(features):
                data = comparison_data[feature]

                # Feature name
                feature_label = ttk.Label(table_frame, text=feature.replace('_', ' ').title())
                feature_label.grid(row=i+1, column=0, padx=5, pady=2, sticky="w")

                # Baseline value
                baseline_label = ttk.Label(table_frame, text=f"{data['baseline_mean']:.2f}")
                baseline_label.grid(row=i+1, column=1, padx=5, pady=2)

                # Current value
                current_label = ttk.Label(table_frame, text=f"{data['current_mean']:.2f}")
                current_label.grid(row=i+1, column=2, padx=5, pady=2)

                # Change percentage
                change_text = f"{data['pct_change']:+.1f}%"
                change_color = "green" if data['pct_change'] > 0 else "red"
                change_label = ttk.Label(table_frame, text=change_text, foreground=change_color)
                change_label.grid(row=i+1, column=3, padx=5, pady=2)

                # Significance
                sig_text = "Yes" if data['significant'] else "No"
                sig_label = ttk.Label(table_frame, text=sig_text)
                sig_label.grid(row=i+1, column=4, padx=5, pady=2)

            # Try to create the chart visualization if we have matplotlib
            try:
                # Create figure with subplots
                fig = plt.Figure(figsize=(10, 8))

                # Create a single plot for the percentage changes
                ax = fig.add_subplot(111)

                # Extract data for plotting
                feature_names = [f.replace('_', ' ').title() for f in features]
                pct_changes = [comparison_data[f]['pct_change'] for f in features]
                colors = ['green' if change > 0 else 'red' for change in pct_changes]

                # Create horizontal bar chart
                bars = ax.barh(feature_names, pct_changes, color=colors, alpha=0.7)

                # Add a vertical line at 0
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

                # Add labels
                ax.set_xlabel('Percentage Change from Baseline (%)')
                ax.set_title('Performance Change by Feature')

                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    label_x = width + (1 if width >= 0 else -1)
                    ax.text(label_x, bar.get_y() + bar.get_height()/2,
                           f'{width:+.1f}%', va='center', ha='left' if width >= 0 else 'right')

                # Adjust layout
                fig.tight_layout()

                # Create canvas for the plot
                canvas_frame = ttk.Frame(main_frame)
                canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

                canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # No toolbar for cleaner UI
            except Exception as e:
                logging.error(f"Error creating visualization: {e}")
                error_label = ttk.Label(main_frame, text=f"Error creating visualization: {str(e)}")
                error_label.pack(pady=20)
        except Exception as e:
            logging.error(f"Error in _plot_single_year_baseline: {str(e)}")
            error_label = ttk.Label(parent_frame, text=f"Error plotting comparison: {str(e)}")
            error_label.pack(pady=20)

    def _plot_multi_year_baseline(self, parent_frame):
        """Plot multi-year comparison with all years of data"""
        try:
            if not self.datasets or len(self.datasets) < 2:
                label = ttk.Label(parent_frame, text="Multiple years of data are required for multi-year comparison.")
                label.pack(pady=20)
                return

            # Create main container
            main_frame = ttk.Frame(parent_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Add title
            title_label = ttk.Label(main_frame, text="Multi-Year Comparison Analysis", font=("Arial", 14, "bold"))
            title_label.pack(pady=10)

            # Get years and features for comparison
            years = sorted(self.datasets.keys())
            baseline_year = min(years)

            # We'll use the same features as the selected features
            features = [f for f in self.selected_features if f != 'department_name']

            if not features:
                label = ttk.Label(main_frame, text="No numeric features selected for analysis.")
                label.pack(pady=20)
                return

            # Calculate means for each year and feature
            yearly_means = {}
            for year, df in self.datasets.items():
                yearly_means[year] = {}
                for feature in features:
                    if feature in df.columns:
                        # Convert to numeric and handle missing values
                        feature_data = pd.to_numeric(df[feature], errors='coerce')
                        if not feature_data.isna().all():
                            yearly_means[year][feature] = feature_data.mean()

            # Create table for year-by-year data
            table_frame = ttk.Frame(main_frame)
            table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            # Add headers (Feature, Year1, Year2, etc.)
            headers = ["Feature"] + years
            for i, header in enumerate(headers):
                label = ttk.Label(table_frame, text=header, font=("Arial", 10, "bold"))
                label.grid(row=0, column=i, padx=5, pady=5, sticky="w")

            # Add data rows with color coding for trends
            for i, feature in enumerate(features):
                # Feature name
                feature_label = ttk.Label(table_frame, text=feature.replace('_', ' ').title())
                feature_label.grid(row=i+1, column=0, padx=5, pady=2, sticky="w")

                # Values for each year
                baseline_value = None
                if baseline_year in yearly_means and feature in yearly_means[baseline_year]:
                    baseline_value = yearly_means[baseline_year][feature]

                for j, year in enumerate(years):
                    if year in yearly_means and feature in yearly_means[year]:
                        value = yearly_means[year][feature]
                        value_text = f"{value:.2f}"

                        # Color code based on trend compared to baseline
                        if baseline_value is not None and year != baseline_year:
                            pct_change = ((value - baseline_value) / baseline_value) * 100 if baseline_value != 0 else 0
                            value_text = f"{value:.2f} ({pct_change:+.1f}%)"
                            color = "green" if pct_change > 0 else ("red" if pct_change < 0 else "black")
                        else:
                            color = "black"

                        # Display value
                        value_label = ttk.Label(table_frame, text=value_text, foreground=color)
                        value_label.grid(row=i+1, column=j+1, padx=5, pady=2)
                    else:
                        # Show N/A if data not available
                        value_label = ttk.Label(table_frame, text="N/A", foreground="gray")
                        value_label.grid(row=i+1, column=j+1, padx=5, pady=2)

            # Try to create visualization if we have matplotlib
            try:
                # Create figure for the trend lines
                fig = plt.Figure(figsize=(10, 8))
                ax = fig.add_subplot(111)

                # Plot trend lines for each feature
                for feature in features:
                    feature_values = []
                    for year in years:
                        if year in yearly_means and feature in yearly_means[year]:
                            feature_values.append(yearly_means[year][feature])
                        else:
                            # Use NaN for missing values
                            feature_values.append(float('nan'))

                    # Plot the line if we have enough data points
                    if any(not np.isnan(v) for v in feature_values):
                        ax.plot(years, feature_values, marker='o', label=feature.replace('_', ' ').title())

                # Set labels and title
                ax.set_xlabel('Year')
                ax.set_ylabel('Rating Value')
                ax.set_title('Feature Ratings Across Years')
                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.7)

                # Adjust layout
                fig.tight_layout()

                # Create canvas for the plot
                canvas_frame = ttk.Frame(main_frame)
                canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

                canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            except Exception as e:
                logging.error(f"Error creating trend visualization: {e}")
                error_label = ttk.Label(main_frame, text=f"Error creating trend visualization: {str(e)}")
                error_label.pack(pady=20)

            # Add interpretation text
            self._add_multi_year_interpretation(main_frame, yearly_means, features, years)

        except Exception as e:
            logging.error(f"Error in _plot_multi_year_baseline: {str(e)}")
            error_label = ttk.Label(parent_frame, text=f"Error plotting multi-year comparison: {str(e)}")
            error_label.pack(pady=20)

    def _add_multi_year_interpretation(self, parent_frame, yearly_means, features, years):
        """Add descriptive interpretation of multi-year comparison"""
        try:
            # Create a frame for the interpretation
            interp_frame = ttk.Frame(parent_frame)
            interp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Add title
            title_label = ttk.Label(interp_frame, text="Interpretation of Multi-Year Trends", font=("Arial", 12, "bold"))
            title_label.pack(pady=10, anchor='w')

            # Create text widget for interpretations
            text_widget = ScrolledText(interp_frame, wrap=tk.WORD, height=10)
            text_widget.pack(fill=tk.BOTH, expand=True, pady=5)

            # Configure tags for formatting
            text_widget.tag_configure('heading', font=("Arial", 11, "bold"))
            text_widget.tag_configure('positive', foreground='green')
            text_widget.tag_configure('negative', foreground='red')
            text_widget.tag_configure('neutral', foreground='blue')
            text_widget.tag_configure('normal', font=("Arial", 10))

            # Generate interpretations
            interpretations = ["# Multi-Year Comparison Analysis\n\n"]

            # Get baseline year and latest year
            baseline_year = min(years)
            latest_year = max(years)

            # Add overview
            interpretations.append(f"## Overview\n")
            interpretations.append(f"This analysis compares data across {len(years)} years ")
            interpretations.append(f"from {baseline_year} to {latest_year}.\n\n")

            # Calculate overall trends
            improved_features = []
            declined_features = []
            stable_features = []

            # Analyze each feature's trend
            for feature in features:
                first_value = None
                last_value = None

                if baseline_year in yearly_means and feature in yearly_means[baseline_year]:
                    first_value = yearly_means[baseline_year][feature]

                if latest_year in yearly_means and feature in yearly_means[latest_year]:
                    last_value = yearly_means[latest_year][feature]

                if first_value is not None and last_value is not None:
                    pct_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0

                    # Categorize based on change
                    if pct_change > 5:
                        improved_features.append((feature, pct_change))
                    elif pct_change < -5:
                        declined_features.append((feature, pct_change))
                    else:
                        stable_features.append((feature, pct_change))

            # Add trend summaries
            interpretations.append(f"## Key Findings\n\n")

            # Improved features
            if improved_features:
                interpretations.append("### Improving Features\n")
                improved_features.sort(key=lambda x: x[1], reverse=True)
                for feature, change in improved_features:
                    interpretations.append(f"* {feature.replace('_', ' ').title()}: {change:+.1f}% improvement from {baseline_year} to {latest_year}\n")
                interpretations.append("\n")

            # Declined features
            if declined_features:
                interpretations.append("### Declining Features\n")
                declined_features.sort(key=lambda x: x[1])
                for feature, change in declined_features:
                    interpretations.append(f"* {feature.replace('_', ' ').title()}: {change:.1f}% decline from {baseline_year} to {latest_year}\n")
                interpretations.append("\n")

            # Stable features
            if stable_features:
                interpretations.append("### Stable Features\n")
                for feature, change in stable_features:
                    interpretations.append(f"* {feature.replace('_', ' ').title()}: Relatively stable ({change:+.1f}%)\n")
                interpretations.append("\n")

            # Analyze year-over-year changes
            interpretations.append("## Year-Over-Year Analysis\n\n")

            # For each consecutive pair of years
            for i in range(len(years) - 1):
                year1 = years[i]
                year2 = years[i + 1]

                interpretations.append(f"### {year1} to {year2} Change\n")

                year_changes = []
                for feature in features:
                    if (year1 in yearly_means and feature in yearly_means[year1] and
                        year2 in yearly_means and feature in yearly_means[year2]):
                        value1 = yearly_means[year1][feature]
                        value2 = yearly_means[year2][feature]

                        pct_change = ((value2 - value1) / value1) * 100 if value1 != 0 else 0
                        year_changes.append((feature, pct_change))

                # Sort changes by magnitude
                year_changes.sort(key=lambda x: abs(x[1]), reverse=True)

                # Show top changes
                if year_changes:
                    for feature, change in year_changes[:5]:  # Show top 5 changes
                        if change > 0:
                            interpretations.append(f"* {feature.replace('_', ' ').title()}: {change:+.1f}% improvement\n")
                        else:
                            interpretations.append(f"* {feature.replace('_', ' ').title()}: {change:.1f}% decline\n")
                else:
                    interpretations.append("* No comparable data available\n")

                interpretations.append("\n")

            # Add recommendations
            interpretations.append("## Recommendations Based on Trends\n\n")

            if declined_features:
                interpretations.append("### Areas Needing Attention\n")
                for feature, change in declined_features[:3]:  # Focus on top 3 declining areas
                    interpretations.append(f"* Focus on improving {feature.replace('_', ' ').title()} which has declined by {abs(change):.1f}%\n")
                interpretations.append("\n")

            if improved_features:
                interpretations.append("### Continue Successful Practices\n")
                for feature, change in improved_features[:3]:  # Highlight top 3 improvements
                    interpretations.append(f"* Maintain successful practices for {feature.replace('_', ' ').title()} which has improved by {change:.1f}%\n")
                interpretations.append("\n")

            # Insert the interpretation text
            interpretation_text = "".join(interpretations)
            text_widget.insert(tk.END, interpretation_text)

            # Apply formatting - this is simplified as tkinter doesn't support markdown
            # Future enhancement could parse the markdown and apply appropriate tags
            current_pos = '1.0'
            while True:
                heading_pos = text_widget.search('##', current_pos, tk.END)
                if not heading_pos:
                    break
                line_end = text_widget.search('\n', heading_pos, tk.END)
                if not line_end:
                    break
                text_widget.tag_add('heading', heading_pos, line_end)
                current_pos = line_end

            # Make text read-only
            text_widget.configure(state='disabled')

        except Exception as e:
            logging.error(f"Error creating interpretation: {e}")
            error_label = ttk.Label(parent_frame, text=f"Error creating interpretation: {str(e)}")
            error_label.pack(pady=20)

    def plot_histograms(self):
        """Plot histograms for each selected feature"""
        try:
            # Clear previous content in histogram tab
            for widget in self.histogram_tab.winfo_children():
                widget.destroy()

            # Create a frame for controls
            controls_frame = ttk.Frame(self.histogram_tab)
            controls_frame.pack(fill=tk.X, padx=10, pady=5)

            # Add year selection dropdown if multiple datasets are loaded
            selected_year = tk.StringVar()
            if len(self.datasets) > 1:
                year_label = ttk.Label(controls_frame, text="Select Year:")
                year_label.pack(side=tk.LEFT, padx=(0, 5))

                years = sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(controls_frame, textvariable=selected_year,
                                            values=years, state="readonly", width=10)
                year_dropdown.pack(side=tk.LEFT, padx=5)
                year_dropdown.set(years[-1])  # Default to most recent year

                # Create a content frame that will be updated based on year selection
                content_frame = ttk.Frame(self.histogram_tab)
                content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

                # Function to update histogram based on selected year
                def on_year_change(event=None):
                    # Clear the content frame
                    for widget in content_frame.winfo_children():
                        widget.destroy()

                    year = selected_year.get()
                    if year in self.datasets:
                        self.create_histogram_for_year(content_frame, year)

                # Bind the callback to the dropdown
                year_dropdown.bind("<<ComboboxSelected>>", on_year_change)

                # Initial plot with default year
                on_year_change()
            else:
                # If only one dataset, use it directly
                content_frame = ttk.Frame(self.histogram_tab)
                content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

                if self.datasets:
                    year = list(self.datasets.keys())[0]
                    self.create_histogram_for_year(content_frame, year)
                else:
                    label = ttk.Label(content_frame, text="No data available for histogram analysis.")
                    label.pack(pady=20)

        except Exception as e:
            logging.error(f"Error in plot_histograms: {str(e)}")
            messagebox.showerror("Error", f"Error creating histograms: {str(e)}")

    def create_histogram_for_year(self, parent_frame, year):
        """Create histogram visualization for a specific year"""
        try:
            # Get filtered data for the selected year
            filtered_df = self.get_filtered_data(year)
            if filtered_df is None or filtered_df.empty:
                label = ttk.Label(parent_frame, text="No data available for histogram analysis.")
                label.pack(pady=20)
                return

            # Filter out non-numeric columns
            numeric_features = [f for f in self.selected_features if f != 'department_name']
            if not numeric_features:
                label = ttk.Label(parent_frame, text="No numeric features selected for histogram analysis.")
                label.pack(pady=20)
                return

            # Create a figure
            fig = plt.Figure(figsize=(12, 8), dpi=100)
            ax = fig.add_subplot(111)

            # Define the rating categories
            categories = ['Needs Improvement', 'Moderately Satisfactory', 'Satisfactory', 'Very Satisfactory']
            category_colors = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2']

            # Calculate the percentage of ratings in each category for each feature
            data = []
            for feature in numeric_features:
                # Convert to numeric and handle missing values
                feature_data = pd.to_numeric(filtered_df[feature], errors='coerce')

                # Skip if no valid data
                if feature_data.isna().all():
                    continue

                # Calculate percentages for each category
                total_valid = len(feature_data.dropna())
                if total_valid > 0:
                    needs_improvement = (feature_data <= 0.74).sum() / total_valid * 100
                    moderately_satisfactory = ((feature_data > 0.74) & (feature_data <= 1.49)).sum() / total_valid * 100
                    satisfactory = ((feature_data > 1.49) & (feature_data <= 2.24)).sum() / total_valid * 100
                    very_satisfactory = (feature_data > 2.24).sum() / total_valid * 100

                    data.append({
                        'feature': feature.replace('_', ' ').title(),
                        'Needs Improvement': needs_improvement,
                        'Moderately Satisfactory': moderately_satisfactory,
                        'Satisfactory': satisfactory,
                        'Very Satisfactory': very_satisfactory,
                        'mean': feature_data.mean()
                    })

            if not data:
                label = ttk.Label(parent_frame, text="No valid data for histogram analysis.")
                label.pack(pady=20)
                return

            # Convert to DataFrame for easier plotting
            df_plot = pd.DataFrame(data)

            # Sort by mean rating (optional)
            df_plot = df_plot.sort_values('mean')

            # Set up the plot
            features = df_plot['feature'].tolist()
            bottom = np.zeros(len(features))

            # Plot stacked bars for each category
            for i, category in enumerate(categories):
                values = df_plot[category].tolist()
                ax.bar(features, values, bottom=bottom, label=category, color=category_colors[i])
                bottom += values

            # Customize the plot
            ax.set_title(f'Distribution of Ratings by Feature ({year})', fontsize=14)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Percentage of Responses', fontsize=12)
            ax.set_ylim(0, 100)

            # Add percentage labels on bars
            for i, feature in enumerate(features):
                total = 0
                for category in categories:
                    value = df_plot.loc[df_plot['feature'] == feature, category].values[0]
                    if value >= 5:  # Only show label if percentage is at least 5%
                        ax.text(i, total + value/2, f'{value:.0f}%',
                               ha='center', va='center', fontsize=9, fontweight='bold')
                    total += value

            # Add mean values as text below each bar
            for i, feature in enumerate(features):
                mean = df_plot.loc[df_plot['feature'] == feature, 'mean'].values[0]
                ax.text(i, -5, f'Mean: {mean:.2f}', ha='center', va='top', fontsize=9)

            # Add legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Adjust layout
            fig.tight_layout(rect=[0, 0.1, 1, 0.95])

            # Create canvas for the plot
            canvas = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            logging.error(f"Error in create_histogram_for_year: {str(e)}")
            messagebox.showerror("Error", f"Error creating histograms for year {year}: {str(e)}")

    def plot_cluster_trends_per_year(self):
        """Create a detailed view of cluster trends for each year"""
        try:
            # Clear previous content
            for widget in self.cluster_trends_tab.winfo_children():
                widget.destroy()

            # Check if we have multiple years of data
            if len(self.datasets) < 2:
                label = ttk.Label(self.cluster_trends_tab, text="Cluster trends require data from multiple years.\nPlease load data for at least two different years to enable this feature.")
                label.pack(pady=20)
                return

            # Create a canvas with scrollbar for the tab
            canvas = tk.Canvas(self.cluster_trends_tab)
            scrollbar = ttk.Scrollbar(self.cluster_trends_tab, orient="vertical", command=canvas.yview)

            # Create main frame inside canvas
            main_frame = ttk.Frame(canvas)

            # Configure the canvas
            canvas.configure(yscrollcommand=scrollbar.set)

            # Pack scrollbar and canvas
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            # Create window in canvas
            canvas.create_window((0, 0), window=main_frame, anchor="nw")

            # Add title
            title_label = ttk.Label(main_frame, text="Cluster Trends Analysis Per Year", font=("Arial", 14, "bold"))
            title_label.pack(pady=10)

            # Calculate cluster distributions for each year
            trend_data = {}
            for year, df in sorted(self.datasets.items()):
                # Get filtered data based on department selection
                filtered_df = df.copy()
                if hasattr(self, 'current_department') and self.current_department.get() != "All Departments":
                    filtered_df = filtered_df[filtered_df['department_name'] == self.current_department.get()]

                # Skip if no data after filtering
                if filtered_df.empty:
                    continue

                # Ensure we're only using numeric features for clustering
                numeric_features = [f for f in self.selected_features if f != 'department_name']
                if not numeric_features:
                    continue

                # Perform clustering for this year
                try:
                    clustered_df, kmeans, cluster_sizes = self.cluster_events(filtered_df, self.selected_features)

                    # Calculate percentages for each cluster
                    total = sum(cluster_sizes.values())
                    percentages = {label: (size / total * 100) for label, size in cluster_sizes.items()}

                    # Store in trend_data
                    trend_data[year] = percentages
                except Exception as e:
                    logging.error(f"Error clustering data for year {year}: {e}")
                    continue

            # If we don't have enough data, show a message
            if len(trend_data) < 2:
                label = ttk.Label(main_frame, text="Insufficient data to generate cluster trends.\nPlease ensure you have valid data for multiple years.")
                label.pack(pady=20)
                return

            # Create overall trend plot
            trend_frame = ttk.Frame(main_frame)
            trend_frame.pack(fill=tk.X, padx=10, pady=10)

            trend_label = ttk.Label(trend_frame, text="Overall Cluster Trends", font=("Arial", 12, "bold"))
            trend_label.pack(pady=5)

            # Create figure for overall trends
            fig_overall = plt.Figure(figsize=(12, 6), dpi=100)  # Lower DPI for faster rendering
            ax_overall = fig_overall.add_subplot(111)

            years = sorted(trend_data.keys())
            categories = ['Needs Improvement', 'Moderately Satisfactory', 'Satisfactory', 'Very Satisfactory']
            colors = {
                'Needs Improvement': '#FF0000',
                'Moderately Satisfactory': '#FFA500',
                'Satisfactory': '#90EE90',
                'Very Satisfactory': '#00FF00'
            }

            # Plot lines for all 4 categories
            for category in categories:
                if all(category in trend_data[year] for year in years):
                    percentages = [trend_data[year][category] for year in years]
                    ax_overall.plot(years, percentages, marker='o',
                                   label=category, linewidth=2,
                                   markersize=8, color=colors[category])

            ax_overall.set_xlabel('Year')
            ax_overall.set_ylabel('Percentage of Attendees')
            ax_overall.set_title('Rating Distribution Trends Over Time')
            ax_overall.grid(True, alpha=0.3)
            ax_overall.legend()

            # Add value labels
            for category in categories:
                if all(category in trend_data[year] for year in years):
                    percentages = [trend_data[year][category] for year in years]
                    for x, y in zip(years, percentages):
                        ax_overall.annotate(f'{y:.1f}%',
                                          (x, y),
                                          textcoords="offset points",
                                          xytext=(0,10),
                                          ha='center',
                                          color=colors[category])

            fig_overall.tight_layout()

            # Create canvas for overall trend plot
            canvas_overall = FigureCanvasTkAgg(fig_overall, master=trend_frame)
            canvas_overall.draw()
            canvas_overall_widget = canvas_overall.get_tk_widget()
            canvas_overall_widget.pack(fill=tk.X, expand=True)

            # Remove toolbar code
            # No toolbar for cleaner UI

            # Add separator
            ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)

            # Add a section for insights
            insights_frame = ttk.LabelFrame(main_frame, text="Trend Insights", padding=10)
            insights_frame.pack(fill=tk.X, padx=10, pady=10)

            # Calculate insights
            insights_text = scrolledtext.ScrolledText(insights_frame, wrap=tk.WORD, height=10)
            insights_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Add insights about trends
            insights_text.insert(tk.END, "Cluster Distribution Trend Insights:\n\n")

            # Calculate changes for each category
            for category in categories:
                if all(category in trend_data[year] for year in years):
                    first_year = years[0]
                    last_year = years[-1]
                    first_value = trend_data[first_year][category]
                    last_value = trend_data[last_year][category]
                    change = last_value - first_value

                    insights_text.insert(tk.END, f"{category}:\n")
                    if abs(change) < 1.0:
                        insights_text.insert(tk.END, f"• Remained stable from {first_year} to {last_year} ({change:.1f}% change)\n")
                    elif change > 0:
                        insights_text.insert(tk.END, f"• Increased from {first_year} to {last_year} by {change:.1f}%\n")
                    else:
                        insights_text.insert(tk.END, f"• Decreased from {first_year} to {last_year} by {abs(change):.1f}%\n")

                    # Calculate trend direction
                    values = [trend_data[year][category] for year in years]
                    if len(values) > 2:
                        # Simple trend analysis
                        increases = sum(1 for i in range(len(values)-1) if values[i+1] > values[i])
                        decreases = sum(1 for i in range(len(values)-1) if values[i+1] < values[i])

                        if increases > decreases:
                            insights_text.insert(tk.END, f"• Overall upward trend over the {len(years)} years\n")
                        elif decreases > increases:
                            insights_text.insert(tk.END, f"• Overall downward trend over the {len(years)} years\n")
                        else:
                            insights_text.insert(tk.END, f"• Fluctuating trend over the {len(years)} years\n")

                    insights_text.insert(tk.END, "\n")

            # Add overall assessment
            insights_text.insert(tk.END, "Overall Assessment:\n")

            # Check if satisfaction is improving
            if all('Very Satisfactory' in trend_data[year] and 'Needs Improvement' in trend_data[year] for year in years):
                first_year = years[0]
                last_year = years[-1]
                vs_change = trend_data[last_year]['Very Satisfactory'] - trend_data[first_year]['Very Satisfactory']
                ni_change = trend_data[last_year]['Needs Improvement'] - trend_data[first_year]['Needs Improvement']

                if vs_change > 0 and ni_change < 0:
                    insights_text.insert(tk.END, "• Overall satisfaction is improving (Very Satisfactory ratings increasing, Needs Improvement ratings decreasing)\n")
                elif vs_change < 0 and ni_change > 0:
                    insights_text.insert(tk.END, "• Overall satisfaction is declining (Very Satisfactory ratings decreasing, Needs Improvement ratings increasing)\n")
                else:
                    insights_text.insert(tk.END, "• Mixed trends in satisfaction levels\n")

            # Configure scroll region when frame changes
            main_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            logging.error(f"Error in plot_cluster_trends_per_year: {e}")
            # Create an error message
            for widget in self.cluster_trends_tab.winfo_children():
                widget.destroy()
            label = ttk.Label(self.cluster_trends_tab, text=f"Error creating cluster trends per year:\n{str(e)}\n\nPlease check the log for more details.")
            label.pack(pady=20)

    def plot_recommendations(self):
        """Create recommendations visualization with year selection"""
        try:
            # Make sure our widgets exist and are properly configured
            self.ensure_widgets_exist()

            # Clear previous content in recommendations tab but be careful not to destroy
            # the recommendations_text widget that's referenced elsewhere
            # First check if recommendations_text is a child of recommendations_tab
            tab_has_rec_text = False
            for widget in self.recommendations_tab.winfo_children():
                if widget == self.recommendations_text:
                    tab_has_rec_text = True
                else:
                    widget.destroy()

            # If the recommendations_text widget isn't in the tab for some reason,
            # we need to be careful about recreating it
            if not tab_has_rec_text:
                # Clear the recommendations tab completely
                for widget in self.recommendations_tab.winfo_children():
                    widget.destroy()

                # Check if our original recommendations_text widget still exists
                try:
                    if hasattr(self, 'recommendations_text') and self.recommendations_text.winfo_exists():
                        # The widget exists but is not in the tab, so we'll recreate it properly
                        self.recommendations_text.destroy()
                except:
                    pass  # The widget doesn't exist or can't be accessed

                # Create a fresh recommendations_text widget
                self.recommendations_text = scrolledtext.ScrolledText(
                    self.recommendations_tab,
                    height=30,
                    font=self.fonts['body'],
                    background=self.colors['light_bg'],
                    foreground=self.colors['text'],
                    padx=10,
                    pady=10,
                    wrap=tk.WORD
                )
                self.recommendations_text.pack(fill=tk.BOTH, expand=True)

                # Configure the tags for styled text
                self.recommendations_text.tag_configure('heading', font=self.fonts['heading'], foreground=self.colors['primary'])
                self.recommendations_text.tag_configure('subheading', font=self.fonts['subheading'], foreground=self.colors['secondary'])
                self.recommendations_text.tag_configure('normal', font=self.fonts['body'])
                self.recommendations_text.tag_configure('success', foreground=self.colors['success'])
                self.recommendations_text.tag_configure('warning', foreground=self.colors['warning'])
                self.recommendations_text.tag_configure('error', foreground=self.colors['danger'])
                self.recommendations_text.tag_configure('highlight', background=self.colors['highlight'])
                self.recommendations_text.tag_configure('code', font=self.fonts['code'], background='#f0f0f0')
                self.recommendations_text.tag_configure('light_text', foreground=self.colors['light_text'])

            # Clear the content of the recommendations_text widget
            self.recommendations_text.delete(1.0, tk.END)

            # The rest of the method remains the same but uses self.recommendations_text
            # instead of creating a new canvas and scrolled widget

            # Get filtered data
            filtered_df = self.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                self.recommendations_text.insert(tk.END, "No data available for recommendations.", 'heading')
                return

            # Add header information
            if len(self.datasets) > 1:
                header_text = f"Recommendations - Combined Data from {len(self.datasets)} Years ({', '.join(sorted(self.datasets.keys()))})"
            else:
                year = next(iter(self.datasets.keys()))
                header_text = f"Recommendations - Year {year}"

            self.recommendations_text.insert(tk.END, header_text + "\n\n", 'heading')

            # If multiple datasets are loaded, add a dropdown to select which year to view
            if len(self.datasets) > 1:
                # Create a selector frame above the recommendations_text
                year_frame = ttk.Frame(self.recommendations_tab)
                year_frame.pack(fill=tk.X, padx=10, pady=5)

                ttk.Label(year_frame, text="Select Year:", style='Subheading.TLabel').pack(side=tk.LEFT, padx=5)

                year_var = tk.StringVar(value="Combined")
                year_options = ["Combined"] + sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(
                    year_frame,
                    textvariable=year_var,
                    values=year_options,
                    state="readonly",
                    width=15,
                    font=self.fonts['body']
                )
                year_dropdown.pack(side=tk.LEFT, padx=5)
                year_dropdown.set("Combined")  # Default to combined view

                def on_year_change(event=None):
                    # Clear existing content except the header
                    self.recommendations_text.delete(1.0, tk.END)
                    self.recommendations_text.insert(tk.END, header_text + "\n\n", 'heading')

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

                    # Show which year we're looking at
                    self.recommendations_text.insert(tk.END, f"Showing recommendations for {year_title}\n\n", 'subheading')

                    # Generate and display recommendations directly in the text widget
                    self.display_recommendations_in_text(df_to_analyze)

                # Bind the dropdown callback
                year_dropdown.bind('<<ComboboxSelected>>', on_year_change)

                # Trigger the dropdown callback to show combined recommendations initially
                on_year_change()
            else:
                # If only one year, display recommendations directly
                year = next(iter(self.datasets.keys()))
                self.recommendations_text.insert(tk.END, f"Showing recommendations for Year {year}\n\n", 'subheading')
                self.display_recommendations_in_text(filtered_df)

        except Exception as e:
            logging.error(f"Error in plot_recommendations: {str(e)}")
            self.recommendations_text.insert(tk.END, f"Error generating recommendations: {str(e)}\n", 'error')

    def display_recommendations_in_text(self, df):
        """Display recommendations directly in the recommendations_text widget"""
        try:
            # Check if we have data
            if df is None or df.empty:
                self.recommendations_text.insert(tk.END, "No data available for recommendations.\n", 'normal')
                return

            # Analyze ratings
            avg_scores, needs_improvement, moderately_satisfactory, satisfactory, very_satisfactory = analyze_event_ratings(df)

            # Get common issues dictionary for reference
            common_issues = get_common_issues_dictionary()

            # Display common issues for low scores
            self.recommendations_text.insert(tk.END, "COMMON ISSUES FOR LOW SCORES:\n\n", 'heading')

            if not needs_improvement.empty:
                for feature in needs_improvement.index:
                    if feature in common_issues:
                        feature_name = feature.replace('_', ' ').title()
                        self.recommendations_text.insert(tk.END, f"{feature_name}:\n", 'subheading')
                        self.recommendations_text.insert(tk.END, f"{common_issues[feature]}\n\n", 'normal')
            else:
                self.recommendations_text.insert(tk.END, "No ratings in the 'Needs Improvement' category.\n\n", 'normal')

            # Generate and display improvement recommendations for low scores
            if not needs_improvement.empty:
                improvement_recs = generate_event_recommendations(needs_improvement)

                self.recommendations_text.insert(tk.END, "IMPROVEMENT RECOMMENDATIONS:\n\n", 'heading')

                if improvement_recs:
                    for feature, recs in improvement_recs.items():
                        if recs:  # Only show if we have recommendations
                            feature_name = feature.replace('_', ' ').title()
                            self.recommendations_text.insert(tk.END, f"{feature_name}:\n", 'subheading')

                            for rec in recs:
                                self.recommendations_text.insert(tk.END, f"• ", 'normal')
                                self.recommendations_text.insert(tk.END, f"{rec['text']}\n", 'normal')
                                self.recommendations_text.insert(tk.END, f"  Action: ", 'light_text')
                                self.recommendations_text.insert(tk.END, f"{rec['action']}\n\n", 'normal')
                else:
                    self.recommendations_text.insert(tk.END, "No improvement recommendations identified.\n\n", 'normal')

            # Generate and display enhancement recommendations for moderate/satisfactory scores
            combined_moderate_scores = pd.concat([moderately_satisfactory, satisfactory])
            if not combined_moderate_scores.empty:
                enhancement_recs = generate_event_improvement_recommendations(combined_moderate_scores)

                self.recommendations_text.insert(tk.END, "ENHANCEMENT RECOMMENDATIONS:\n\n", 'heading')

                if enhancement_recs:
                    for feature, recs in enhancement_recs.items():
                        if recs:  # Only show if we have recommendations
                            feature_name = feature.replace('_', ' ').title()
                            score_value = avg_scores[feature]
                            self.recommendations_text.insert(tk.END, f"{feature_name} (Current Score: {score_value:.2f}/3.00):\n", 'subheading')

                            for rec in recs:
                                self.recommendations_text.insert(tk.END, f"• ", 'normal')
                                self.recommendations_text.insert(tk.END, f"{rec['text']}\n", 'normal')
                                self.recommendations_text.insert(tk.END, f"  Action: ", 'light_text')
                                self.recommendations_text.insert(tk.END, f"{rec['action']}\n\n", 'normal')
                else:
                    self.recommendations_text.insert(tk.END, "No enhancement recommendations identified.\n\n", 'normal')

            # Generate detailed recommendations
            detailed_recs = generate_recommendations_from_rules(self.rules, df) if hasattr(self, 'rules') else {}

            self.recommendations_text.insert(tk.END, "\nDETAILED RECOMMENDATIONS:\n\n", 'heading')

            if detailed_recs:
                for feature, recs in detailed_recs.items():
                    if recs:  # Only show if we have recommendations
                        feature_name = feature.replace('_', ' ').title()
                        self.recommendations_text.insert(tk.END, f"{feature_name}:\n", 'subheading')

                        for rec in recs:
                            self.recommendations_text.insert(tk.END, f"• ", 'normal')
                            self.recommendations_text.insert(tk.END, f"{rec['text']} ", 'normal')
                            self.recommendations_text.insert(tk.END, f"(Priority: {rec['priority']})\n", 'light_text')
                            self.recommendations_text.insert(tk.END, f"  {rec['action']}\n\n", 'normal')
            elif not needs_improvement.empty:
                self.recommendations_text.insert(tk.END, "No detailed recommendations identified.\n\n", 'normal')
            else:
                self.recommendations_text.insert(tk.END, "No detailed recommendations identified.\n\n", 'normal')

            # Generate and display maintenance recommendations for high scores
            if not very_satisfactory.empty:
                maintenance_recs = generate_event_maintenance_recommendations(very_satisfactory)

                self.recommendations_text.insert(tk.END, "\nMAINTENANCE RECOMMENDATIONS:\n\n", 'heading')

                if maintenance_recs:
                    for feature, recs in maintenance_recs.items():
                        if recs:  # Only show if we have recommendations
                            feature_name = feature.replace('_', ' ').title()
                            self.recommendations_text.insert(tk.END, f"{feature_name}:\n", 'subheading')

                            for rec in recs:
                                self.recommendations_text.insert(tk.END, f"• ", 'normal')
                                self.recommendations_text.insert(tk.END, f"{rec['text']}\n", 'normal')
                                self.recommendations_text.insert(tk.END, f"  Action: ", 'light_text')
                                self.recommendations_text.insert(tk.END, f"{rec['action']}\n\n", 'normal')
                else:
                    self.recommendations_text.insert(tk.END, "No maintenance recommendations identified.\n\n", 'normal')

            # Check if we have year-over-year data to analyze
            if hasattr(self, 'baseline_comparison') and self.baseline_comparison:
                # Generate yearly change recommendations
                change_recs = generate_yearly_change_recommendations(self.baseline_comparison)

                if change_recs:
                    self.recommendations_text.insert(tk.END, "\nYEARLY CHANGE ANALYSIS:\n\n", 'heading')

                    # Areas that need attention based on negative trends
                    self.recommendations_text.insert(tk.END, "Areas Needing Attention:\n", 'subheading')
                    for feature, recs in change_recs.items():
                        feature_name = feature.replace('_', ' ').title()
                        # Show feature name with change percentage
                        if recs and 'change' in recs[0]:
                            self.recommendations_text.insert(tk.END, f"• {feature_name} ({recs[0]['change']})\n", 'normal')

                    self.recommendations_text.insert(tk.END, "\n", 'normal')

                    # Overall assessment of trends
                    if len(change_recs) > 0:
                        if len(change_recs) > 2:
                            assessment = "Concerning trend with multiple areas needing attention."
                        else:
                            assessment = "Some areas show concerning trends that require attention."
                    else:
                        assessment = "No significant negative trends identified."

                    self.recommendations_text.insert(tk.END, f"Overall Assessment: {assessment}\n\n", 'normal')

                    # Detailed recommendations for areas with negative trends
                    self.recommendations_text.insert(tk.END, "RECOMMENDATIONS FOR DECLINING AREAS:\n\n", 'heading')

                    for feature, recs in change_recs.items():
                        if recs:  # Only show if we have recommendations
                            feature_name = feature.replace('_', ' ').title()
                            self.recommendations_text.insert(tk.END, f"{feature_name} ({recs[0]['change']}):\n", 'subheading')

                            for rec in recs:
                                self.recommendations_text.insert(tk.END, f"• ", 'normal')
                                self.recommendations_text.insert(tk.END, f"{rec['text']}\n", 'normal')
                                self.recommendations_text.insert(tk.END, f"  Action: ", 'light_text')
                                self.recommendations_text.insert(tk.END, f"{rec['action']}\n", 'normal')
                                self.recommendations_text.insert(tk.END, f"  Priority: ", 'light_text')
                                self.recommendations_text.insert(tk.END, f"{rec['priority']}\n\n", 'normal')

        except Exception as e:
            logging.error(f"Error displaying recommendations: {str(e)}")
            self.recommendations_text.insert(tk.END, f"Error displaying recommendations: {str(e)}\n", 'error')

    def open_pdf_export_window(self):
        """Open a window to configure PDF export options"""
        try:
            # Check if data is loaded
            if not hasattr(self, 'df') or self.df is None or self.df.empty:
                messagebox.showerror("Error", "No data loaded. Please load data first.")
                return

            # Ensure all widgets exist before proceeding
            self.ensure_widgets_exist()

            # Create export window
            export_window = tk.Toplevel(self.root)
            export_window.title("Export to PDF")
            export_window.geometry("600x680")  # Made taller for performance warning
            export_window.transient(self.root)
            export_window.grab_set()

            # Add padding to the window
            main_frame = ttk.Frame(export_window, padding=20)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            title_label = ttk.Label(
                main_frame,
                text="Export Analysis to PDF",
                font=self.fonts['heading'],
                foreground=self.colors['primary']
            )
            title_label.pack(pady=(0, 20))

            # Performance info frame with warning
            perf_frame = ttk.Frame(main_frame, padding=10)
            perf_frame.pack(fill=tk.X, pady=5)

            # Add a warning icon
            warning_label = ttk.Label(
                perf_frame,
                text="⚠️",
                font=("Segoe UI", 16),
                foreground=self.colors['warning']
            )
            warning_label.pack(side=tk.LEFT, padx=(0, 10))

            # Add performance warning text
            perf_text = (
                "Exporting large datasets or multiple years of data may take several minutes, especially "
                "when including cluster trends analysis. You can cancel the export at any time."
            )
            perf_warning = ttk.Label(
                perf_frame,
                text=perf_text,
                foreground=self.colors['warning'],
                wraplength=480,
                justify=tk.LEFT
            )
            perf_warning.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # File path frame
            file_frame = ttk.Frame(main_frame)
            file_frame.pack(fill=tk.X, pady=10)

            ttk.Label(file_frame, text="Save PDF to:").pack(side=tk.LEFT, padx=(0, 10))

            file_path = tk.StringVar()
            file_entry = ttk.Entry(file_frame, textvariable=file_path, width=40)
            file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

            def select_file_path():
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                if filepath:
                    file_path.set(filepath)

            browse_btn = ttk.Button(file_frame, text="Browse...", command=select_file_path)
            browse_btn.pack(side=tk.RIGHT)

            # Content selection frame
            content_frame = ttk.LabelFrame(main_frame, text="Select Content to Export", padding=10)
            content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            # Create variables to track selected items for export
            export_selections = {
                "analysis_results": tk.BooleanVar(value=False),  # Changed to False by default
                "clustering": tk.BooleanVar(value=True),
                "association_rules": tk.BooleanVar(value=True),
                "descriptive_stats": tk.BooleanVar(value=True),
                "histograms": tk.BooleanVar(value=True),
                "recommendations": tk.BooleanVar(value=True),
                "baseline_comparisons": tk.BooleanVar(value=True),
                "cluster_trends": tk.BooleanVar(value=False)  # Changed to False by default since it's slow
            }

            # Create a frame for organizing selections into two columns
            selections_frame = ttk.Frame(content_frame)
            selections_frame.pack(fill=tk.BOTH, expand=True)

            # Left column
            left_col = ttk.Frame(selections_frame)
            left_col.pack(side=tk.LEFT, fill=tk.Y, expand=True)

            # Right column
            right_col = ttk.Frame(selections_frame)
            right_col.pack(side=tk.LEFT, fill=tk.Y, expand=True)

            # Add checkboxes to columns for better layout
            ttk.Checkbutton(left_col, text="Analysis Results", variable=export_selections["analysis_results"]).pack(anchor=tk.W, pady=5)
            ttk.Checkbutton(left_col, text="Clustering", variable=export_selections["clustering"]).pack(anchor=tk.W, pady=5)
            ttk.Checkbutton(left_col, text="Association Rules", variable=export_selections["association_rules"]).pack(anchor=tk.W, pady=5)
            ttk.Checkbutton(left_col, text="Descriptive Statistics", variable=export_selections["descriptive_stats"]).pack(anchor=tk.W, pady=5)

            ttk.Checkbutton(right_col, text="Histograms", variable=export_selections["histograms"]).pack(anchor=tk.W, pady=5)
            ttk.Checkbutton(right_col, text="Recommendations", variable=export_selections["recommendations"]).pack(anchor=tk.W, pady=5)
            ttk.Checkbutton(right_col, text="Baseline Comparisons", variable=export_selections["baseline_comparisons"]).pack(anchor=tk.W, pady=5)

            # Add cluster trends with a warning (at the bottom)
            cluster_trends_frame = ttk.Frame(content_frame)
            cluster_trends_frame.pack(fill=tk.X, anchor=tk.W, pady=5)
            ttk.Checkbutton(cluster_trends_frame, text="Cluster Trends", variable=export_selections["cluster_trends"]).pack(side=tk.LEFT)
            ttk.Label(cluster_trends_frame, text="(may slow down export significantly)", foreground=self.colors['warning'], font=self.fonts['small']).pack(side=tk.LEFT, padx=5)

            # Year selection frame
            year_frame = ttk.LabelFrame(main_frame, text="Select Years", padding=10)
            year_frame.pack(fill=tk.X, pady=10)

            year_selection = {}
            all_years_var = tk.BooleanVar(value=True)

            def toggle_all_years():
                """Toggle selection of all years"""
                all_selected = all_years_var.get()
                for year_var in year_selection.values():
                    year_var.set(all_selected)

            ttk.Checkbutton(year_frame, text="All Years", variable=all_years_var, command=toggle_all_years).pack(anchor=tk.W, pady=5)

            # Add individual year checkboxes if multiple years are available
            if self.datasets:
                years_frame = ttk.Frame(year_frame)
                years_frame.pack(fill=tk.X, padx=20)

                col, row = 0, 0
                for i, year in enumerate(sorted(self.datasets.keys())):
                    year_selection[year] = tk.BooleanVar(value=True)
                    ttk.Checkbutton(years_frame, text=str(year), variable=year_selection[year]).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
                    col += 1
                    if col > 2:  # 3 items per row
                        col = 0
                        row += 1

            # Department selection frame
            dept_frame = ttk.LabelFrame(main_frame, text="Select Departments", padding=10)
            dept_frame.pack(fill=tk.X, pady=10)

            dept_selection = {}
            all_depts_var = tk.BooleanVar(value=True)

            def toggle_all_depts():
                """Toggle selection of all departments"""
                all_selected = all_depts_var.get()
                for dept_var in dept_selection.values():
                    dept_var.set(all_selected)

            ttk.Checkbutton(dept_frame, text="All Departments", variable=all_depts_var, command=toggle_all_depts).pack(anchor=tk.W, pady=5)

            # Add individual department checkboxes
            if self.departments:
                depts_frame = ttk.Frame(dept_frame)
                depts_frame.pack(fill=tk.X, padx=20)

                col, row = 0, 0
                for i, dept in enumerate(sorted(self.departments)):
                    dept_selection[dept] = tk.BooleanVar(value=True)
                    ttk.Checkbutton(depts_frame, text=dept, variable=dept_selection[dept]).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
                    col += 1
                    if col > 1:  # 2 items per row
                        col = 0
                        row += 1

            # Buttons frame
            buttons_frame = ttk.Frame(main_frame)
            buttons_frame.pack(fill=tk.X, pady=(20, 0))

            def export_pdf():
                """Handle PDF export with selected options"""
                # Check if file path is provided
                if not file_path.get():
                    messagebox.showerror("Error", "Please specify a file path for the PDF.")
                    return

                # Check if at least one item is selected
                if not any(export_selections.values()):
                    messagebox.showerror("Error", "Please select at least one content type to export.")
                    return

                # Collect selected years
                if all_years_var.get():
                    selected_years = list(self.datasets.keys())
                else:
                    selected_years = [year for year, selected in year_selection.items() if selected.get()]

                # Confirm with warning if cluster trends is selected and many years
                if export_selections["cluster_trends"].get() and len(selected_years) > 3:
                    confirm = messagebox.askyesno(
                        "Performance Warning",
                        "You've selected to export cluster trends with multiple years, which may take several minutes. Continue?",
                        icon='warning'
                    )
                    if not confirm:
                        return

                # Collect selected content
                selected_content = {key: var.get() for key, var in export_selections.items()}

                # Collect selected departments
                if all_depts_var.get():
                    selected_depts = ["All Departments"] + self.departments
                else:
                    selected_depts = ["All Departments"] if all_depts_var.get() else []
                    selected_depts += [dept for dept, selected in dept_selection.items() if selected.get()]

                # Start PDF generation
                self.generate_pdf(file_path.get(), selected_content, selected_years, selected_depts)

                # Close the window
                export_window.destroy()

            cancel_btn = ttk.Button(buttons_frame, text="Cancel", command=export_window.destroy)
            cancel_btn.pack(side=tk.RIGHT, padx=5)

            export_btn = ttk.Button(
                buttons_frame,
                text="Generate PDF",
                command=export_pdf,
                style="Success.TButton"
            )
            export_btn.pack(side=tk.RIGHT, padx=5)

        except Exception as e:
            logging.error(f"Error in open_pdf_export_window: {str(e)}")
            messagebox.showerror("Error", f"Error creating PDF export window: {str(e)}")

    def generate_pdf(self, file_path, selected_content, selected_years, selected_departments):
        """Generate a PDF report with the selected content, years, and departments"""
        try:
            # Add a flag to indicate we're generating a PDF - this will be used for optimizations
            self._generating_pdf = True
            self._cancel_pdf_export = False

            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Generating PDF")
            progress_window.geometry("300x150")
            progress_window.transient(self.root)
            progress_window.grab_set()

            progress_frame = ttk.Frame(progress_window, padding=20)
            progress_frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(progress_frame, text="Generating PDF report...", font=self.fonts['subheading']).pack(pady=(0, 10))

            progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=250, mode='indeterminate')
            progress.pack(pady=10)
            progress.start(10)

            status_var = tk.StringVar(value="Initializing...")
            status_label = ttk.Label(progress_frame, textvariable=status_var)
            status_label.pack(pady=10)

            # Add cancel button to avoid getting stuck
            def cancel_export():
                self._cancel_pdf_export = True
                status_var.set("Cancelling export...")
                progress_window.update()

            cancel_btn = ttk.Button(progress_frame, text="Cancel", command=cancel_export)
            cancel_btn.pack(pady=(5, 0))

            # Update progress in a non-blocking way
            def update_status(message):
                status_var.set(message)
                progress_window.update()
                progress_window.update_idletasks()
                # Check if cancelled
                return not self._cancel_pdf_export

            # Create the PDF document
            update_status("Creating PDF document...")

            # Use landscape orientation for more space for visualizations
            doc = SimpleDocTemplate(
                file_path,
                pagesize=landscape(letter),
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch
            )

            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=18,
                spaceAfter=12,
                alignment=TA_CENTER
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=14,
                spaceAfter=6
            )
            subheading_style = ParagraphStyle(
                'CustomSubheading',
                parent=styles['Heading2'],
                fontSize=12,
                spaceAfter=6
            )
            normal_style = styles['Normal']
            interpretation_style = ParagraphStyle(
                'Interpretation',
                parent=styles['Normal'],
                fontSize=10,
                leftIndent=20,
                spaceAfter=6
            )

            # Create bullet point style
            bullet_style = ParagraphStyle(
                'BulletPoint',
                parent=styles['Normal'],
                fontSize=10,
                leftIndent=30,
                bulletIndent=15,
                spaceAfter=3
            )

            # Create the content list
            content = []

            # Add title and metadata
            content.append(Paragraph("Event Analysis Report", title_style))
            content.append(Spacer(1, 0.2*inch))

            # Add report metadata
            metadata = [
                f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Years included: {', '.join(map(str, selected_years)) if selected_years else 'None'}",
                f"Departments included: {', '.join(selected_departments) if selected_departments else 'All'}"
            ]

            for meta in metadata:
                content.append(Paragraph(meta, normal_style))

            content.append(Spacer(1, 0.3*inch))

            # Process each section based on selection
            if selected_content["analysis_results"]:
                update_status("Adding analysis results...")

                content.append(Paragraph("Analysis Results", heading_style))
                content.append(Spacer(1, 0.1*inch))

                # Get analysis results from the output text widget
                analysis_text = self.output_text.get(1.0, tk.END)
                for line in analysis_text.split('\n'):
                    if line.strip():  # Skip empty lines
                        content.append(Paragraph(line, normal_style))
                        content.append(Spacer(1, 0.05*inch))

                content.append(Spacer(1, 0.2*inch))

            if selected_content["recommendations"]:
                update_status("Adding recommendations...")

                content.append(Paragraph("Recommendations", heading_style))
                content.append(Spacer(1, 0.1*inch))

                # Get recommendations from the recommendations text widget
                recommendations_text = self.recommendations_text.get(1.0, tk.END)
                for line in recommendations_text.split('\n'):
                    if line.strip():  # Skip empty lines
                        # Format bullet points if they exist
                        if line.strip().startswith('•'):
                            content.append(Paragraph(f"&#8226; {line.strip()[1:].strip()}", bullet_style))
                        else:
                            content.append(Paragraph(line, normal_style))

                content.append(Spacer(1, 0.2*inch))

            # Add cluster trends analysis if selected
            if selected_content["cluster_trends"] and len(selected_years) > 1:
                update_status("Adding cluster trends analysis...")

                content.append(Paragraph("Cluster Trends Over Time", heading_style))
                content.append(Spacer(1, 0.1*inch))

                # Calculate cluster trends data - OPTIMIZED VERSION
                try:
                    # Create a figure for cluster trends with greatly reduced resolution
                    trend_fig = plt.Figure(figsize=(10, 6), dpi=72)  # Reduced DPI for faster rendering
                    trend_ax = trend_fig.add_subplot(111)

                    # Create a dictionary to store simplified trend data
                    trend_data = {}

                    # Analyze each year - with smaller samples and timeout protection
                    total_years = len(selected_years)

                    # Define maximum sample size - much smaller for faster processing
                    MAX_SAMPLE_SIZE = 500  # Reduced from 1000

                    # Define simplified feature set if there are many features
                    if len(self.selected_features) > 5:
                        # Use only the first 5 features for performance
                        simplified_features = [f for f in self.selected_features[:5] if f != 'department_name']
                    else:
                        simplified_features = [f for f in self.selected_features if f != 'department_name']

                    for i, year in enumerate(sorted(selected_years)):
                        if not update_status(f"Processing cluster data for year {year} ({i+1}/{total_years})..."):
                            progress_window.destroy()
                            return

                        if year in self.datasets:
                            # Get filtered data for the year
                            year_df = self.get_filtered_data(year)

                            if year_df is not None and not year_df.empty and len(simplified_features) > 0:
                                # Create clusters for this year - with extreme performance optimizations
                                try:
                                    # Always use a small sample for faster clustering
                                    sample_size = min(MAX_SAMPLE_SIZE, len(year_df))
                                    year_df_sample = year_df.sample(n=sample_size, random_state=42)

                                    # Use simplified clustering for PDF generation (fewer iterations, lower tolerance)
                                    # Extract selected features excluding 'department_name'
                                    cluster_data = year_df_sample[simplified_features].copy()

                                    # Handle missing values with mean imputation
                                    for feature in simplified_features:
                                        cluster_data[feature] = pd.to_numeric(cluster_data[feature], errors='coerce')

                                    cluster_data = cluster_data.fillna(cluster_data.mean())

                                    # Skip standardization for speed and use fixed categories
                                    # Use direct categorization based on means - much faster than KMeans
                                    mean_ratings = cluster_data.mean(axis=1)

                                    # Create simplified clusters based on rating categories
                                    labels = pd.cut(mean_ratings,
                                        bins=[-float('inf'), 0.74, 1.49, 2.24, float('inf')],
                                        labels=[0, 1, 2, 3]).astype(str)

                                    cluster_mapping = {
                                        '0': 'Needs Improvement',
                                        '1': 'Moderately Satisfactory',
                                        '2': 'Satisfactory',
                                        '3': 'Very Satisfactory'
                                    }

                                    # Map to readable labels
                                    label_names = labels.map(cluster_mapping)

                                    # Calculate distribution
                                    counts = label_names.value_counts()

                                    # Store simplified data for this year
                                    trend_data[year] = {
                                        'Needs Improvement': 0,
                                        'Moderately Satisfactory': 0,
                                        'Satisfactory': 0,
                                        'Very Satisfactory': 0
                                    }

                                    # Fill in available values
                                    total = len(label_names)
                                    for category in cluster_mapping.values():
                                        if category in counts:
                                            trend_data[year][category] = counts[category] / total * 100

                                except Exception as e:
                                    logging.error(f"Error calculating simplified cluster trends for year {year}: {e}")
                                    # Continue with other years instead of failing completely

                    # Plot trend data if we have enough years
                    if len(trend_data) > 1:
                        if not update_status("Creating trend visualizations..."):
                            progress_window.destroy()
                            return

                        try:
                            # Sort years
                            years = sorted(trend_data.keys())

                            # Plot satisfaction trends
                            categories = ['Very Satisfactory', 'Satisfactory', 'Moderately Satisfactory', 'Needs Improvement']
                            colors = ['#06D6A0', '#118AB2', '#FFD166', '#FF6B6B']

                            # Create data for categories
                            category_data = {}
                            for category in categories:
                                category_data[category] = []
                                for year in years:
                                    # Check if category exists in this year's data
                                    if category in trend_data[year]:
                                        category_data[category].append(trend_data[year][category])
                                    else:
                                        # Use a default value if missing
                                        category_data[category].append(0)

                            # Clear the figure and recreate it with low resolution
                            plt.close(trend_fig)
                            trend_fig = plt.Figure(figsize=(10, 6), dpi=72)  # Even lower DPI
                            trend_ax = trend_fig.add_subplot(111)

                            # Plot lines with simplifications
                            for i, category in enumerate(categories):
                                if len(years) == len(category_data[category]) and len(years) > 0:
                                    trend_ax.plot(years, category_data[category], marker='o', markersize=6,
                                                linewidth=2, color=colors[i], label=category)

                            trend_ax.set_title('Rating Distribution Trends Across Years', fontsize=14)
                            trend_ax.set_xlabel('Year', fontsize=12)
                            trend_ax.set_ylabel('Percentage of Ratings', fontsize=12)
                            trend_ax.legend(loc='best', fontsize=10)
                            trend_ax.grid(True, linestyle='--', alpha=0.7)

                            # Ensure y-axis shows percentages properly
                            trend_ax.set_ylim(0, 100)
                            trend_ax.set_yticks(range(0, 101, 20))  # Fewer ticks

                            # Add the plot to the PDF with low-res image capturing
                            content.append(Paragraph("Rating Distribution Trends", subheading_style))

                            # Use a more reliable way to capture the figure with low resolution
                            img_data = io.BytesIO()
                            trend_fig.tight_layout()  # Ensure everything fits
                            trend_fig.savefig(img_data, format='png', dpi=72, bbox_inches='tight')
                            img_data.seek(0)

                            # Add the image to content with explicit width/height
                            img = Image(img_data, width=8*inch, height=4.5*inch)
                            content.append(img)
                            content.append(Spacer(1, 0.2*inch))

                            # Print success message to logs
                            logging.info("Successfully added simplified cluster trends visualization to PDF")

                        except Exception as e:
                            logging.error(f"Error creating cluster trends visualization: {str(e)}")
                            # Add a message to the PDF indicating the error
                            content.append(Paragraph(f"Error creating cluster trends visualization: {str(e)}", normal_style))

                except Exception as e:
                    logging.error(f"Error creating cluster trends for PDF: {str(e)}")
                    content.append(Paragraph(f"Error generating cluster trends: {str(e)}", normal_style))

                content.append(Spacer(1, 0.3*inch))

            # Function to capture figure as an image with reduced resolution
            def capture_figure(fig):
                img_data = io.BytesIO()
                fig.savefig(img_data, format='png', dpi=72, bbox_inches='tight')  # Much lower DPI
                img_data.seek(0)
                return Image(img_data, width=9*inch, height=5*inch)

            # Process visualizations for each year and department
            for year in selected_years:
                update_status(f"Processing data for year {year}...")

                content.append(Paragraph(f"Year: {year}", subheading_style))
                content.append(Spacer(1, 0.1*inch))

                # Filter data for the current year
                year_df = self.get_filtered_data(year)

                # We need to process each department separately since the GUI shows data by department
                for dept in selected_departments:
                    update_status(f"Processing data for department {dept} in year {year}...")

                    # Set the current department
                    previous_dept = self.current_department.get()
                    self.current_department.set(dept)

                    # Get filtered data for this department and year
                    dept_df = self.get_filtered_data(year)

                    if dept_df is None or dept_df.empty:
                        continue

                    content.append(Paragraph(f"Department: {dept}", subheading_style))
                    content.append(Spacer(1, 0.1*inch))

                    # Add descriptive stats for this department and year
                    if selected_content["descriptive_stats"] and self.selected_features:
                        update_status(f"Adding descriptive statistics for {dept} in {year}...")

                        try:
                            # Create descriptive stats for the PDF
                            content.append(Paragraph(f"Descriptive Statistics", subheading_style))

                            # Filter and analyze only numeric columns
                            numeric_cols = [col for col in dept_df.columns if col != 'department_name']

                            if numeric_cols:
                                # Create a summary table
                                desc_stats = dept_df[numeric_cols].describe().round(2)

                                # Convert to a list of lists for Table
                                data = [['Metric'] + [col.replace('_', ' ').title() for col in desc_stats.columns]]
                                for idx, row in desc_stats.iterrows():
                                    data.append([idx.capitalize()] + [f"{val:.2f}" for val in row])

                                # Create the table
                                table = Table(data, colWidths=[1.5*inch] + [1.2*inch] * len(desc_stats.columns))

                                # Add table style
                                table_style = TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), '#add8e6'),  # lightblue
                                    ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),   # black
                                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), '#ffffff'),  # white
                                    ('GRID', (0, 0), (-1, -1), 1, '#000000'),     # black
                                    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                                ])
                                table.setStyle(table_style)

                                content.append(table)
                                content.append(Spacer(1, 0.1*inch))

                                # Add interpretation of descriptive stats
                                content.append(Paragraph("Interpretation:", normal_style))

                                # Calculate averages and interpret
                                avg_scores = dept_df[numeric_cols].mean()
                                interpretation = interpret_ratings(avg_scores)

                                # Format the interpretation text for the PDF
                                for line in interpretation.split('\n'):
                                    if line.strip():
                                        if ":" in line and not line.startswith(' '):
                                            # This is a feature heading
                                            content.append(Paragraph(line, normal_style))
                                        else:
                                            # This is a detail line
                                            content.append(Paragraph(line, interpretation_style))

                                content.append(Spacer(1, 0.2*inch))
                        except Exception as e:
                            logging.error(f"Error creating descriptive stats for PDF: {str(e)}")

                    # Process each selected visualization type
                    if selected_content["clustering"] and self.selected_features:
                        update_status(f"Adding clustering visualization for {dept} in {year}...")

                        # Create clustering visualization
                        try:
                            # Get or create clustering
                            clustered_df, kmeans, cluster_sizes, labels = self.cluster_events(dept_df, self.selected_features, return_labels=True)

                            # Create a figure for this plot
                            cluster_fig = plt.Figure(figsize=(10, 6), dpi=100)
                            ax = cluster_fig.add_subplot(111)

                            # Create the plot
                            if kmeans is not None and len(set(labels)) > 1:  # Only if there are multiple clusters
                                self.create_cluster_visualization(dept_df, kmeans, labels, ax)

                                # Add the plot to the PDF
                                content.append(Paragraph(f"Clustering Analysis", subheading_style))
                                content.append(capture_figure(cluster_fig))

                                # Add cluster interpretation
                                content.append(Paragraph("Cluster Interpretation:", normal_style))

                                # Get cluster centers to understand what each cluster represents
                                centers = kmeans.cluster_centers_
                                n_clusters = len(centers)

                                # Calculate interpretation for each cluster
                                for i in range(n_clusters):
                                    cluster_center = centers[i]
                                    cluster_size = np.sum(labels == i)
                                    cluster_percent = (cluster_size / len(labels)) * 100

                                    # Find top features (highest values) for this cluster
                                    numeric_features = [f for f in self.selected_features if f != 'department_name']
                                    top_features_idx = np.argsort(cluster_center)[-3:]  # Top 3 features
                                    top_features = [numeric_features[idx] for idx in top_features_idx if idx < len(numeric_features)]
                                    top_features_values = [cluster_center[idx] for idx in top_features_idx if idx < len(numeric_features)]

                                    # Format cluster interpretation
                                    cluster_text = f"Cluster {i+1} ({cluster_size} events, {cluster_percent:.1f}% of total):"
                                    content.append(Paragraph(cluster_text, normal_style))

                                    # Add top features information
                                    if top_features:
                                        content.append(Paragraph(f"Key characteristics:", interpretation_style))
                                        for j, (feature, value) in enumerate(zip(top_features, top_features_values)):
                                            feature_name = feature.replace('_', ' ').title()
                                            feature_text = f"&#8226; {feature_name}: {value:.2f}"
                                            content.append(Paragraph(feature_text, bullet_style))

                                content.append(Spacer(1, 0.2*inch))
                        except Exception as e:
                            logging.error(f"Error creating clustering visualization for PDF: {str(e)}")

                    # Reset to the previous department when done
                    self.current_department.set(previous_dept)

            # Add baseline comparison if selected
            if selected_content["baseline_comparisons"] and len(selected_years) > 1:
                update_status("Adding baseline comparison...")

                try:
                    # Create heading with visual emphasis
                    content.append(Paragraph("Baseline Comparison Analysis", heading_style))
                    content.append(Spacer(1, 0.1*inch))

                    # Add explanatory text
                    content.append(Paragraph("This analysis compares the earliest year (baseline) with the most recent year to identify significant changes in ratings.", normal_style))
                    content.append(Spacer(1, 0.1*inch))

                    # Get baseline metrics from the first year and current metrics from the latest year
                    baseline_year = min(selected_years)
                    current_year = max(selected_years)

                    content.append(Paragraph(f"Baseline Year: {baseline_year} | Current Year: {current_year}", normal_style))
                    content.append(Spacer(1, 0.1*inch))

                    if baseline_year in self.datasets and current_year in self.datasets:
                        # Get filtered data for baseline and current year
                        baseline_df = self.get_filtered_data(baseline_year)
                        current_df = self.get_filtered_data(current_year)

                        if baseline_df is not None and not baseline_df.empty and current_df is not None and not current_df.empty:
                            # Calculate baseline metrics
                            numeric_cols = [col for col in baseline_df.columns if col != 'department_name']
                            baseline_metrics = {}

                            for col in numeric_cols:
                                if col in baseline_df.columns:
                                    baseline_metrics[col] = baseline_df[col].mean()

                            # Calculate current metrics and comparison
                            comparison_data = {}

                            # Create a figure for visual comparison
                            comp_fig = plt.Figure(figsize=(10, 6), dpi=100)
                            comp_ax = comp_fig.add_subplot(111)

                            # Data for the bar chart
                            features = []
                            baseline_values = []
                            current_values = []

                            for col in numeric_cols:
                                if col in current_df.columns and col in baseline_metrics:
                                    current_val = current_df[col].mean()
                                    baseline_val = baseline_metrics[col]

                                    # For visualization
                                    features.append(col.replace('_', ' ').title())
                                    baseline_values.append(baseline_val)
                                    current_values.append(current_val)

                                    # Calculate percent change
                                    if baseline_val > 0:
                                        percent_change = ((current_val - baseline_val) / baseline_val) * 100
                                    else:
                                        percent_change = 0

                                    # Determine significance
                                    significant = abs(percent_change) > 5  # More than 5% change

                                    comparison_data[col] = {
                                        'baseline': baseline_val,
                                        'current': current_val,
                                        'percent_change': percent_change,
                                        'significant': significant
                                    }

                            # Create comparison visualization
                            if features:
                                # Set up bar positions
                                x = np.arange(len(features))
                                width = 0.35

                                # Create bars
                                comp_ax.bar(x - width/2, baseline_values, width, label=f'Baseline ({baseline_year})')
                                comp_ax.bar(x + width/2, current_values, width, label=f'Current ({current_year})')

                                # Add labels and styling
                                comp_ax.set_ylabel('Rating Value', fontsize=12)
                                comp_ax.set_title('Baseline vs. Current Comparison', fontsize=14)
                                comp_ax.set_xticks(x)
                                comp_ax.set_xticklabels(features, rotation=45, ha='right')
                                comp_ax.legend()
                                comp_ax.grid(axis='y', linestyle='--', alpha=0.7)

                                # Set y-axis to standard rating scale
                                comp_ax.set_ylim(0, 3)

                                # Adjust layout
                                comp_fig.tight_layout()

                                # Add the visualization to the PDF
                                img_data = io.BytesIO()
                                comp_fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                                img_data.seek(0)
                                content.append(Image(img_data, width=8*inch, height=4.5*inch))
                                content.append(Spacer(1, 0.2*inch))

                            # Create a table for the comparison data
                            data = [['Feature', 'Baseline', 'Current', 'Change (%)', 'Significant']]

                            for feature, values in comparison_data.items():
                                feature_name = feature.replace('_', ' ').title()
                                sign = '+' if values['percent_change'] > 0 else ''
                                data.append([
                                    feature_name,
                                    f"{values['baseline']:.2f}",
                                    f"{values['current']:.2f}",
                                    f"{sign}{values['percent_change']:.1f}%",
                                    'Yes' if values['significant'] else 'No'
                                ])

                            # Create the table
                            table = Table(data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])

                            # Add table style with color coding for changes
                            table_style = TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), '#add8e6'),  # lightblue header
                                ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),   # black text in header
                                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), '#ffffff'),  # white background for data
                                ('GRID', (0, 0), (-1, -1), 1, '#000000'),     # black grid
                                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 1), (-1, -1), 9),
                            ])

                            # Apply table style
                            table.setStyle(table_style)

                            # Add color to positive and negative changes in the table
                            for i, (feature, values) in enumerate(comparison_data.items(), 1):
                                if values['significant']:
                                    if values['percent_change'] > 0:
                                        # Green for positive changes
                                        table_style.add('BACKGROUND', (3, i), (3, i), '#c6ecc6')  # light green
                                    else:
                                        # Red for negative changes
                                        table_style.add('BACKGROUND', (3, i), (3, i), '#ffcccc')  # light red

                            # Apply the updated style
                            table.setStyle(table_style)

                            content.append(Paragraph("Detailed Comparison:", subheading_style))
                            content.append(table)
                            content.append(Spacer(1, 0.2*inch))

                            # Add interpretation text
                            content.append(Paragraph("Interpretation of Changes:", subheading_style))

                            # Look for significant improvements and declines
                            improvements = []
                            declines = []

                            for feature, values in comparison_data.items():
                                if values['significant']:
                                    feature_name = feature.replace('_', ' ').title()
                                    if values['percent_change'] > 0:
                                        improvements.append(f"{feature_name} (+{values['percent_change']:.1f}%)")
                                    else:
                                        declines.append(f"{feature_name} ({values['percent_change']:.1f}%)")

                            if improvements:
                                content.append(Paragraph("Significant Improvements:", interpretation_style))
                                for item in improvements:
                                    content.append(Paragraph(f"&#8226; {item}", bullet_style))
                                content.append(Spacer(1, 0.1*inch))

                            if declines:
                                content.append(Paragraph("Areas Needing Attention:", interpretation_style))
                                for item in declines:
                                    content.append(Paragraph(f"&#8226; {item}", bullet_style))
                                content.append(Spacer(1, 0.1*inch))

                            if not improvements and not declines:
                                content.append(Paragraph("No significant changes detected between baseline and current year.", interpretation_style))
                            else:
                                # Add overall assessment
                                if len(improvements) > len(declines):
                                    content.append(Paragraph("Overall Assessment: Positive trend with more improvements than areas of concern.", normal_style))
                                elif len(improvements) < len(declines):
                                    content.append(Paragraph("Overall Assessment: Concerning trend with more areas needing attention than improvements.", normal_style))
                                else:
                                    content.append(Paragraph("Overall Assessment: Mixed results with equal numbers of improvements and areas needing attention.", normal_style))

                            content.append(Spacer(1, 0.3*inch))

                            # Log success
                            logging.info("Successfully added baseline comparison to PDF")

                            # Add multi-year analysis if we have more than 2 years
                            if len(selected_years) > 2:
                                update_status("Adding multi-year comparison analysis...")

                                # Add a heading
                                content.append(Paragraph("Multi-Year Trend Analysis", heading_style))
                                content.append(Spacer(1, 0.1*inch))

                                # Add explanatory text
                                content.append(Paragraph(
                                    f"This analysis compares data across all {len(selected_years)} years " +
                                    f"from {min(selected_years)} to {max(selected_years)}.",
                                    normal_style))
                                content.append(Spacer(1, 0.1*inch))

                                # Get years and features for comparison
                                years = sorted(selected_years)
                                baseline_year = min(years)
                                features = [f for f in self.selected_features if f != 'department_name']

                                if features:
                                    # Calculate means for each year and feature
                                    yearly_means = {}
                                    for year in years:
                                        df = self.get_filtered_data(year)
                                        if df is not None and not df.empty:
                                            yearly_means[year] = {}
                                            for feature in features:
                                                if feature in df.columns:
                                                    # Convert to numeric and handle missing values
                                                    feature_data = pd.to_numeric(df[feature], errors='coerce')
                                                    if not feature_data.isna().all():
                                                        yearly_means[year][feature] = feature_data.mean()

                                    # Create a figure for trend lines
                                    trend_fig = plt.Figure(figsize=(10, 6), dpi=100)
                                    trend_ax = trend_fig.add_subplot(111)

                                    # Plot trend lines for each feature
                                    for feature in features:
                                        feature_values = []
                                        for year in years:
                                            if year in yearly_means and feature in yearly_means[year]:
                                                feature_values.append(yearly_means[year][feature])
                                            else:
                                                # Use NaN for missing values
                                                feature_values.append(float('nan'))

                                        # Plot the line if we have enough data points
                                        if any(not np.isnan(v) for v in feature_values):
                                            trend_ax.plot(years, feature_values, marker='o',
                                                         label=feature.replace('_', ' ').title())

                                    # Set labels and title
                                    trend_ax.set_xlabel('Year', fontsize=12)
                                    trend_ax.set_ylabel('Rating Value', fontsize=12)
                                    trend_ax.set_title('Feature Ratings Across Years', fontsize=14)
                                    trend_ax.legend(loc='best')
                                    trend_ax.grid(True, linestyle='--', alpha=0.7)

                                    # Set y-axis to standard rating scale
                                    trend_ax.set_ylim(0, 3)

                                    # Adjust layout
                                    trend_fig.tight_layout()

                                    # Add the visualization to the PDF
                                    trend_img_data = io.BytesIO()
                                    trend_fig.savefig(trend_img_data, format='png', dpi=150, bbox_inches='tight')
                                    trend_img_data.seek(0)
                                    content.append(Image(trend_img_data, width=8*inch, height=4.5*inch))
                                    content.append(Spacer(1, 0.2*inch))

                                    # Create a table for year-by-year data
                                    year_table_data = [['Feature'] + years]

                                    for feature in features:
                                        row = [feature.replace('_', ' ').title()]
                                        for year in years:
                                            if year in yearly_means and feature in yearly_means[year]:
                                                row.append(f"{yearly_means[year][feature]:.2f}")
                                            else:
                                                row.append("N/A")
                                        year_table_data.append(row)

                                    # Add the year-by-year table to the PDF
                                    year_table = Table(year_table_data)
                                    year_table_style = TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), '#add8e6'),  # lightblue header
                                        ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),   # black text in header
                                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                        ('BACKGROUND', (0, 1), (-1, -1), '#ffffff'),  # white background for data
                                        ('GRID', (0, 0), (-1, -1), 1, '#000000'),     # black grid
                                        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                                    ])
                                    year_table.setStyle(year_table_style)
                                    content.append(year_table)
                                    content.append(Spacer(1, 0.2*inch))

                                    # Add interpretation of multi-year trends
                                    content.append(Paragraph("Interpretation of Multi-Year Trends:", subheading_style))
                                    content.append(Spacer(1, 0.1*inch))

                                    # Calculate overall trends
                                    improved_features = []
                                    declined_features = []
                                    stable_features = []

                                    # Analyze each feature's trend
                                    for feature in features:
                                        first_value = None
                                        last_value = None

                                        if baseline_year in yearly_means and feature in yearly_means[baseline_year]:
                                            first_value = yearly_means[baseline_year][feature]

                                        if max(years) in yearly_means and feature in yearly_means[max(years)]:
                                            last_value = yearly_means[max(years)][feature]

                                        if first_value is not None and last_value is not None:
                                            pct_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0

                                            # Categorize based on change
                                            if pct_change > 5:
                                                improved_features.append((feature, pct_change))
                                            elif pct_change < -5:
                                                declined_features.append((feature, pct_change))
                                            else:
                                                stable_features.append((feature, pct_change))

                                    # Add improved features
                                    if improved_features:
                                        content.append(Paragraph("Improving Features:", interpretation_style))

                                        improved_features.sort(key=lambda x: x[1], reverse=True)
                                        for feature, change in improved_features:
                                            feature_name = feature.replace('_', ' ').title()
                                            content.append(Paragraph(f"&#8226; {feature_name}: {change:+.1f}% improvement from {baseline_year} to {max(years)}", bullet_style))

                                        content.append(Spacer(1, 0.1*inch))

                                    # Add declined features
                                    if declined_features:
                                        content.append(Paragraph("Declining Features:", interpretation_style))

                                        declined_features.sort(key=lambda x: x[1])
                                        for feature, change in declined_features:
                                            feature_name = feature.replace('_', ' ').title()
                                            content.append(Paragraph(f"&#8226; {feature_name}: {change:.1f}% decline from {baseline_year} to {max(years)}", bullet_style))

                                        content.append(Spacer(1, 0.1*inch))

                                    # Add stable features
                                    if stable_features:
                                        content.append(Paragraph("Stable Features:", interpretation_style))

                                        for feature, change in stable_features:
                                            feature_name = feature.replace('_', ' ').title()
                                            content.append(Paragraph(f"&#8226; {feature_name}: Relatively stable ({change:+.1f}%)", bullet_style))

                                    content.append(Spacer(1, 0.2*inch))

                                    # Add year-over-year analysis
                                    content.append(Paragraph("Year-Over-Year Changes:", subheading_style))
                                    content.append(Spacer(1, 0.1*inch))

                                    # For each consecutive pair of years
                                    for i in range(len(years) - 1):
                                        year1 = years[i]
                                        year2 = years[i + 1]

                                        content.append(Paragraph(f"{year1} to {year2} Change:", interpretation_style))

                                        year_changes = []
                                        for feature in features:
                                            if (year1 in yearly_means and feature in yearly_means[year1] and
                                                year2 in yearly_means and feature in yearly_means[year2]):
                                                value1 = yearly_means[year1][feature]
                                                value2 = yearly_means[year2][feature]

                                                pct_change = ((value2 - value1) / value1) * 100 if value1 != 0 else 0
                                                year_changes.append((feature, pct_change))

                                        # Sort changes by magnitude
                                        year_changes.sort(key=lambda x: abs(x[1]), reverse=True)

                                        # Show top changes
                                        if year_changes:
                                            for feature, change in year_changes[:5]:  # Show top 5 changes
                                                feature_name = feature.replace('_', ' ').title()
                                                if change > 0:
                                                    content.append(Paragraph(f"&#8226; {feature_name}: {change:+.1f}% improvement", bullet_style))
                                                else:
                                                    content.append(Paragraph(f"&#8226; {feature_name}: {change:.1f}% decline", bullet_style))
                                        else:
                                            content.append(Paragraph("&#8226; No comparable data available", bullet_style))

                                        content.append(Spacer(1, 0.1*inch))

                                    # Add recommendations based on trends
                                    content.append(Paragraph("Recommendations Based on Trends:", subheading_style))
                                    content.append(Spacer(1, 0.1*inch))

                                    if declined_features:
                                        content.append(Paragraph("Areas Needing Attention:", interpretation_style))

                                        for feature, change in declined_features[:3]:  # Focus on top 3 declining areas
                                            feature_name = feature.replace('_', ' ').title()
                                            content.append(Paragraph(f"&#8226; Focus on improving {feature_name} which has declined by {abs(change):.1f}%", bullet_style))

                                        content.append(Spacer(1, 0.1*inch))

                                    if improved_features:
                                        content.append(Paragraph("Continue Successful Practices:", interpretation_style))

                                        for feature, change in improved_features[:3]:  # Highlight top 3 improvements
                                            feature_name = feature.replace('_', ' ').title()
                                            content.append(Paragraph(f"&#8226; Maintain successful practices for {feature_name} which has improved by {change:.1f}%", bullet_style))

                                content.append(Spacer(1, 0.3*inch))
                                logging.info("Successfully added multi-year trend analysis to PDF")
                        else:
                            content.append(Paragraph("Could not generate baseline comparison: insufficient data in baseline or current year.", normal_style))
                    else:
                        content.append(Paragraph(f"Could not generate baseline comparison: missing data for baseline year ({baseline_year}) or current year ({current_year}).", normal_style))

                except Exception as e:
                    logging.error(f"Error creating baseline comparison for PDF: {str(e)}")
                    content.append(Paragraph(f"Error creating baseline comparison: {str(e)}", normal_style))

            # Build the PDF
            update_status("Building PDF document...")
            doc.build(content)

            # Close progress window
            progress_window.destroy()

            # Show success message
            messagebox.showinfo("Export Successful", f"Report has been successfully exported to:\n{file_path}")

            # Reset PDF generation flag
            self._generating_pdf = False
            self._cancel_pdf_export = False

        except Exception as e:
            # Reset PDF generation flag
            self._generating_pdf = False
            self._cancel_pdf_export = False

            logging.error(f"Error in generate_pdf: {str(e)}")
            messagebox.showerror("Error", f"Error generating PDF: {str(e)}")

            # Close progress window if it exists
            try:
                progress_window.destroy()
            except:
                pass

    def create_cluster_visualization(self, df, kmeans, labels, ax):
        """Create a cluster visualization on the given axes for PDF export"""
        try:
            # Check if we have enough dimensions for clustering
            if len(self.selected_features) < 2:
                return

            # Filter out non-numeric columns
            numeric_features = [f for f in self.selected_features if f != 'department_name']

            if len(numeric_features) < 2:
                return

            # Extract features for visualization
            X = df[numeric_features].values

            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Use PCA for visualization if we have more than 2 dimensions
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                # Create a plot
                centers_pca = pca.transform(scaler.transform(kmeans.cluster_centers_))

                # Plot each cluster with a different color
                for i in range(len(np.unique(labels))):
                    ax.scatter(
                        X_pca[labels == i, 0],
                        X_pca[labels == i, 1],
                        s=50,
                        label=f'Cluster {i+1}'
                    )

                # Plot cluster centers
                ax.scatter(
                    centers_pca[:, 0],
                    centers_pca[:, 1],
                    s=200,
                    marker='X',
                    c='red',
                    label='Centroids'
                )

                # Add labels and title
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('Event Clusters (PCA projection)')
                ax.legend()

            else:
                # Use the first two features directly
                # Plot each cluster with a different color
                for i in range(len(np.unique(labels))):
                    ax.scatter(
                        X_scaled[labels == i, 0],
                        X_scaled[labels == i, 1],
                        s=50,
                        label=f'Cluster {i+1}'
                    )

                # Plot cluster centers
                centers_scaled = scaler.transform(kmeans.cluster_centers_)
                ax.scatter(
                    centers_scaled[:, 0],
                    centers_scaled[:, 1],
                    s=200,
                    marker='X',
                    c='red',
                    label='Centroids'
                )

                # Add labels and title
                ax.set_xlabel(numeric_features[0])
                ax.set_ylabel(numeric_features[1])
                ax.set_title('Event Clusters')
                ax.legend()

        except Exception as e:
            logging.error(f"Error in create_cluster_visualization: {str(e)}")

def interpret_ratings(avg_scores):
    """
    Interpret ratings based on fixed 0-3 scale with corresponding remarks.

    Scale:
    0.00-0.74: Needs Improvement
    0.75-1.49: Moderately Satisfactory
    1.50-2.24: Satisfactory
    2.25-3.00: Very Satisfactory
    """
    interpretations = []

    for feature, score in avg_scores.items():
        feature_name = feature.replace('_', ' ')

        # Determine rating category and remarks
        if score <= 0.74:
            category = "Needs Improvement"
            remarks = "Critical attention required. Immediate action needed."
        elif score <= 1.49:
            category = "Moderately Satisfactory"
            remarks = "Shows potential but requires enhancement."
        elif score <= 2.24:
            category = "Satisfactory"
            remarks = "Meets expectations. Minor improvements possible."
        else:
            category = "Very Satisfactory"
            remarks = "Excellent performance. Maintain current standards."

        # Format the interpretation
        interpretation = (
            f"{feature_name}:\n"
            f"  Score: {score:.2f}/3.00\n"
            f"  Rating: {category}\n"
            f"  Remarks: {remarks}\n"
        )
        interpretations.append(interpretation)

    # Calculate overall rating
    overall_score = avg_scores.mean()
    if overall_score <= 0.74:
        overall_category = "Needs Improvement"
        overall_remarks = "Event requires significant improvements across multiple areas."
    elif overall_score <= 1.49:
        overall_category = "Moderately Satisfactory"
        overall_remarks = "Event shows promise but needs systematic enhancements."
    elif overall_score <= 2.24:
        overall_category = "Satisfactory"
        overall_remarks = "Event meets basic requirements. Consider targeted improvements."
    else:
        overall_category = "Very Satisfactory"
        overall_remarks = "Event excels in most areas. Focus on maintaining high standards."

    # Add overall rating
    interpretations.append(
        f"\nOVERALL EVENT RATING:\n"
        f"  Score: {overall_score:.2f}/3.00\n"
        f"  Rating: {overall_category}\n"
        f"  Remarks: {overall_remarks}\n"
    )

    return "\n".join(interpretations)

def generate_yearly_change_recommendations(comparison_data, threshold=-15.0):
    """
    Generate recommendations based on significant negative yearly changes in ratings.

    Parameters:
    - comparison_data (dict): Dictionary containing year-over-year changes for each feature
    - threshold (float): Percentage threshold for considering a change significant negative (default: -15%)

    Returns:
    - dict: Dictionary of recommendations keyed by feature name
    """
    recommendations = {}

    if not comparison_data:
        return recommendations

    # Standard recommendations for features with negative changes
    base_recommendations = {
        'Overall_Rating': [
            {
                'text': "Address declining overall satisfaction",
                'action': "Form a task force to identify and address the root causes of declining satisfaction",
                'priority': 'High'
            }
        ],
        'Objectives_Met': [
            {
                'text': "Review objectives and delivery methods",
                'action': "Conduct focused surveys to identify why objectives are not being met as effectively",
                'priority': 'High'
            }
        ],
        'Venue_Rating': [
            {
                'text': "Reassess venue suitability",
                'action': "Conduct site evaluations and gather specific feedback about venue concerns",
                'priority': 'Medium'
            }
        ],
        'Schedule_Rating': [
            {
                'text': "Evaluate schedule effectiveness",
                'action': "Review timing and duration of events, considering attendee feedback",
                'priority': 'Medium'
            }
        ],
        'Speaker_Rating': [
            {
                'text': "Review speaker selection process",
                'action': "Implement enhanced speaker training and selection criteria",
                'priority': 'High'
            }
        ],
        'Content_Rating': [
            {
                'text': "Refresh content strategy",
                'action': "Update content to include more current and relevant material",
                'priority': 'High'
            }
        ],
        'Materials_Rating': [
            {
                'text': "Improve quality of materials",
                'action': "Redesign materials with professional assistance",
                'priority': 'Medium'
            }
        ],
        'Engagement_Rating': [
            {
                'text': "Enhance engagement strategies",
                'action': "Incorporate more interactive elements and activities",
                'priority': 'High'
            }
        ],
        'Relevance_Rating': [
            {
                'text': "Update content for increased relevance",
                'action': "Conduct needs assessment to align content with current needs",
                'priority': 'High'
            }
        ],
        'Food_Rating': [
            {
                'text': "Review catering service and options",
                'action': "Consider alternative catering providers or menu options",
                'priority': 'Low'
            }
        ],
        'Technology_Rating': [
            {
                'text': "Upgrade technology infrastructure",
                'action': "Assess technical requirements and implement improvements",
                'priority': 'Medium'
            }
        ],
        'Networking_Rating': [
            {
                'text': "Enhance networking opportunities",
                'action': "Design structured networking activities",
                'priority': 'Medium'
            }
        ],
        'Value_Rating': [
            {
                'text': "Improve perceived value proposition",
                'action': "Conduct cost-benefit analysis and adjust pricing or offerings",
                'priority': 'High'
            }
        ],
        'Allowance_Rating': [
            {
                'text': "Review allowance structure and amounts",
                'action': "Benchmark allowances against industry standards and adjust accordingly",
                'priority': 'Medium'
            }
        ]
    }

    # Generic recommendation for any feature not specifically covered
    generic_recommendation = [
        {
            'text': "Address declining satisfaction in this area",
            'action': "Investigate causes and implement targeted improvements",
            'priority': 'Medium'
        }
    ]

    # Create recommendations for features with significant negative changes
    for feature, data in comparison_data.items():
        pct_change = data.get('pct_change', 0)

        # If the change is below the threshold (significant negative change)
        if pct_change < threshold:
            # Get feature-specific recommendations or use generic ones
            feature_recs = base_recommendations.get(feature, generic_recommendation)

            # Add change percentage to recommendations for context
            for rec in feature_recs:
                rec['change'] = f"{pct_change:.1f}%"

            recommendations[feature] = feature_recs

    return recommendations

def main():
    try:
        root = tk.Tk()
        app = AnalysisGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
