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

def prepare_for_association_rules(df, selected_features, year=None):
    """
    Convert selected features to transactions with ratings using fixed rating scale.
    Optimized for memory efficiency and data quality.
    """
    try:
        logging.info("Starting preparation of association rules data")
        logging.debug(f"Selected features: {selected_features}")

        if df.empty or not selected_features:
            logging.error("Empty dataframe or no features selected")
            return pd.DataFrame()

        binary_df = pd.DataFrame()

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
                categories = numeric_values.apply(categorize_rating)

                # Create binary columns for each rating category
                for category in rating_categories.keys():
                    col_name = f"{feature}_{category}"
                    binary_df[col_name] = (categories == category).astype(int)

                # Log feature processing
                valid_count = numeric_values.notna().sum()
                logging.debug(f"Processed {feature}: {valid_count} valid values")

        # Validate final binary dataframe
        if binary_df.empty:
            logging.warning("No valid binary columns created")
            return pd.DataFrame()

        logging.info(f"Created binary dataframe with {len(binary_df.columns)} columns")
        return binary_df

    except Exception as e:
        logging.error(f"Error in prepare_for_association_rules: {e}")
        return pd.DataFrame()

def generate_association_rules(binary_df, min_support=0.05):
    """
    Generate association rules from binary data with optimized parameters and validation.
    """
    try:
        logging.info("Starting association rules generation")

        if binary_df.empty:
            logging.warning("Empty DataFrame provided for association rules")
            return pd.DataFrame()

        # Log data statistics for debugging
        logging.debug("Binary data statistics:")
        support_counts = binary_df.sum()
        for col in binary_df.columns:
            support_pct = (support_counts[col] / len(binary_df)) * 100
            logging.debug(f"{col}: {support_counts[col]} occurrences ({support_pct:.2f}%)")

        # Optimize min_support based on data size
        adjusted_min_support = max(min_support, 2 / len(binary_df))
        logging.info(f"Using adjusted min_support: {adjusted_min_support}")

        # Generate frequent itemsets with optimized parameters
        try:
            frequent_itemsets = apriori(
                binary_df,
                min_support=adjusted_min_support,
                use_colnames=True,
                max_len=3,  # Limit to 3-item sets for efficiency
                verbose=1
            )

            if frequent_itemsets is None or frequent_itemsets.empty:
                logging.warning(f"No frequent itemsets found with min_support={adjusted_min_support}")
                return pd.DataFrame()

            logging.info(f"Found {len(frequent_itemsets)} frequent itemsets")

            # Generate rules with optimized confidence threshold
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=0.5,  # Higher confidence threshold for better quality
                support_only=False
            )

            if rules.empty:
                logging.warning("No rules generated with current thresholds")
                return pd.DataFrame()

            # Filter rules by lift for significance
            significant_rules = rules[rules['lift'] > 1.0]

            # Sort rules by lift and confidence
            significant_rules = significant_rules.sort_values(
                ['lift', 'confidence'],
                ascending=[False, False]
            )

            # Add support percentage for better interpretation
            significant_rules['support_pct'] = significant_rules['support'] * 100
            significant_rules['confidence_pct'] = significant_rules['confidence'] * 100

            logging.info(f"Generated {len(significant_rules)} significant rules")
            return significant_rules

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
            self.distribution_tab = ttk.Frame(self.tab_control)
            self.recommendations_tab = ttk.Frame(self.tab_control)
            self.baseline_tab = ttk.Frame(self.tab_control)
            self.cluster_trends_tab = ttk.Frame(self.tab_control)  # Add new tab for cluster trends per year

            # Add tabs to notebook
            self.tab_control.add(self.output_tab, text='Analysis Results')
            self.tab_control.add(self.cluster_tab, text='Clustering')
            self.tab_control.add(self.rules_tab, text='Association Rules')
            self.tab_control.add(self.descriptive_tab, text='Descriptive Stats')
            self.tab_control.add(self.histogram_tab, text='Histograms')
            self.tab_control.add(self.distribution_tab, text='Distribution')
            self.tab_control.add(self.recommendations_tab, text='Recommendations')
            self.tab_control.add(self.baseline_tab, text='Baseline Comparisons')
            self.tab_control.add(self.cluster_trends_tab, text='Cluster Trends Per Year')  # Add new tab

            # Create scrolled text widgets for output and recommendations
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

            # Clear previous output
            self.output_text.delete(1.0, tk.END)
            self.recommendations_text.delete(1.0, tk.END)

            # Clear previous visualizations
            for tab in [self.cluster_tab, self.distribution_tab, self.histogram_tab,
                       self.rules_tab, self.descriptive_tab, self.baseline_tab,
                       self.cluster_trends_tab]:
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
                # Prepare data for association rules
                binary_df = prepare_for_association_rules(filtered_df, self.selected_features)

                # Generate rules
                rules = generate_association_rules(binary_df)

                if not rules.empty:
                    # Display association rules interpretation
                    self.output_text.insert(tk.END, "\nAssociation Rules Analysis:\n")
                    self.output_text.insert(tk.END, interpret_event_association_rules(rules))

                    # Plot association rules
                    self.plot_association_rules(rules, self.current_department.get())
                else:
                    self.output_text.insert(tk.END, "\nNo significant association rules found.\n")

            except Exception as e:
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

            # 6. Create distribution visualizations
            try:
                # Calculate distribution results
                distribution_results = {}
                for feature in self.selected_features:
                    if feature != 'department_name':
                        distribution_results[feature] = {
                            'mean': filtered_df[feature].mean(),
                            'std': filtered_df[feature].std(),
                            'min': filtered_df[feature].min(),
                            'max': filtered_df[feature].max()
                        }

                self.plot_distribution_comparison(distribution_results)
            except Exception as e:
                logging.error(f"Error creating distribution visualizations: {str(e)}")

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
                            self.plot_baseline_comparison(comparison_data)
                        else:
                            self.output_text.insert(tk.END, "No valid comparison data generated\n")
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

    def get_filtered_data(self):
        """Get data filtered by selected department"""
        if self.current_department.get() == "All Departments":
            return self.df
        else:
            # Keep department_name as is, don't try to convert it to numeric
            filtered_df = self.df[self.df['department_name'] == self.current_department.get()].copy()

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

                # Clear all plots
                for tab in [self.cluster_tab, self.distribution_tab,
                           self.histogram_tab, self.rules_tab]:
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
        self.distribution_tab = ttk.Frame(self.tab_control)
        self.histogram_tab = ttk.Frame(self.tab_control)
        self.rules_tab = ttk.Frame(self.tab_control)
        self.analysis_tab = ttk.Frame(self.tab_control)
        self.recommendations_tab = ttk.Frame(self.tab_control)  # Add recommendations tab

        self.tab_control.add(self.descriptive_tab, text='Descriptive Analysis')
        self.tab_control.add(self.cluster_tab, text='Clustering')
        self.tab_control.add(self.distribution_tab, text='Distribution')
        self.tab_control.add(self.histogram_tab, text='Histograms')
        self.tab_control.add(self.rules_tab, text='Association Rules')
        self.tab_control.add(self.analysis_tab, text='Analysis Results')
        self.tab_control.add(self.recommendations_tab, text='Recommendations')  # Add recommendations tab

        self.tab_control.pack(fill=tk.BOTH, expand=True)

        # Create scrolled text widgets for both analysis and recommendations tabs
        self.output_text = scrolledtext.ScrolledText(self.analysis_tab, wrap=tk.WORD, height=20)
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
                        # Prepare data for association rules
                        binary_df = prepare_for_association_rules(df_to_analyze, self.selected_features, selected_year)

                        # Generate rules
                        new_rules = generate_association_rules(binary_df)

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

                        # Plot the new rules
                        self.plot_association_rules(new_rules, selected_year)
                    except Exception as e:
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
            if len(self.datasets) > 1:
                year_frame = ttk.Frame(plot_frame)
                year_frame.pack(fill=tk.X, padx=10, pady=5)

                ttk.Label(year_frame, text="Select Year:").pack(side=tk.LEFT, padx=5)

                year_var = tk.StringVar(value="Combined")
                year_options = ["Combined"] + sorted(self.datasets.keys())
                year_dropdown = ttk.Combobox(year_frame, textvariable=year_var, values=year_options, state="readonly")
                year_dropdown.pack(side=tk.LEFT, padx=5)

                def on_year_change(event=None):
                    # Clear existing stats
                    for widget in stats_frame.winfo_children():
                        widget.destroy()

                    # Create new stats text widget
                    stats_text = tk.Text(stats_frame, height=20, width=150, state='disabled')
                    stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                    # Configure the text widget to be non-editable but allow configuration
                    stats_text.config(state='normal')

                    # Get data for selected year
                    selected_year = year_var.get()
                    if selected_year == "Combined":
                        df_to_analyze = filtered_df
                    else:
                        df_to_analyze = self.datasets[selected_year]
                        if self.current_department.get() != "All Departments":
                            df_to_analyze = df_to_analyze[df_to_analyze['department_name'] == self.current_department.get()]

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
                    stats_text.insert(tk.END, f"Feature Statistics for {selected_year}:\n\n")

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

            # Create summary statistics table
            stats_frame = ttk.LabelFrame(plot_frame, text="Summary Statistics")
            stats_frame.pack(fill=tk.X, padx=10, pady=10)

            # Create a text widget for the statistics with increased height
            stats_text = tk.Text(stats_frame, height=20, width=150, state='disabled')
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
            if len(self.datasets) > 1:
                stats_text.insert(tk.END, "Feature Statistics for Combined Data:\n\n")
            else:
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

    def plot_distribution_comparison(self, distribution_results):
        """Plot distribution comparison with robust handling of dictionary input"""
        try:
            # Clear previous content
            for widget in self.distribution_tab.winfo_children():
                widget.destroy()

            # Create a canvas with scrollbar for distribution tab
            canvas = tk.Canvas(self.distribution_tab)
            scrollbar = ttk.Scrollbar(self.distribution_tab, orient="vertical", command=canvas.yview)

            # Create main frame inside canvas
            main_frame = ttk.Frame(canvas)

            # Configure the canvas
            canvas.configure(yscrollcommand=scrollbar.set)

            # Pack scrollbar and canvas
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            # Create window in canvas
            canvas.create_window((0, 0), window=main_frame, anchor="nw")

            # Get features from the distribution results
            features = list(distribution_results.keys())
            if not features:
                logging.warning("No features available for distribution plot")
                return

            # Plot for each year
            for year, df in sorted(self.datasets.items()):
                # Create year frame
                year_frame = ttk.Frame(main_frame)
                year_frame.pack(fill=tk.X, padx=5, pady=5)

                # Add year label
                year_label = ttk.Label(year_frame, text=f"Year {year}", font=('Arial', 12, 'bold'))
                year_label.pack(pady=5)

                # Create figure
                fig = plt.Figure(figsize=(12, 8), dpi=100)
                ax = fig.add_subplot(111)

                # Position for bars
                x = np.arange(len(features))
                width = 0.2

                # Colors for categories
                colors = {
                    'Needs Improvement': '#FF0000',
                    'Moderately Satisfactory': '#FFA500',
                    'Satisfactory': '#90EE90',
                    'Very Satisfactory': '#00FF00'
                }

                # Calculate distribution for this year's data
                year_distribution = {}
                for feature in features:
                    feature_data = pd.to_numeric(df[feature], errors='coerce')
                    total_valid = len(feature_data.dropna())
                    if total_valid > 0:
                        year_distribution[feature] = {
                            'Needs Improvement': (feature_data <= 0.74).sum() / total_valid * 100,
                            'Moderately Satisfactory': ((feature_data > 0.74) & (feature_data <= 1.49)).sum() / total_valid * 100,
                            'Satisfactory': ((feature_data > 1.49) & (feature_data <= 2.24)).sum() / total_valid * 100,
                            'Very Satisfactory': (feature_data > 2.24).sum() / total_valid * 100
                        }

                # Plot bars for each feature
                for i, feature in enumerate(features):
                    if feature in year_distribution:
                        feature_dist = year_distribution[feature]

                        # Plot bars for each category
                        ax.bar(x[i] - width*1.5, feature_dist['Needs Improvement'], width,
                              label='Needs Improvement' if i == 0 else "",
                              color=colors['Needs Improvement'], alpha=0.7)
                        ax.bar(x[i] - width/2, feature_dist['Moderately Satisfactory'], width,
                              label='Moderately Satisfactory' if i == 0 else "",
                              color=colors['Moderately Satisfactory'], alpha=0.7)
                        ax.bar(x[i] + width/2, feature_dist['Satisfactory'], width,
                              label='Satisfactory' if i == 0 else "",
                              color=colors['Satisfactory'], alpha=0.7)
                        ax.bar(x[i] + width*1.5, feature_dist['Very Satisfactory'], width,
                              label='Very Satisfactory' if i == 0 else "",
                              color=colors['Very Satisfactory'], alpha=0.7)

                # Customize plot
                ax.set_ylabel('Percentage of Responses')
                ax.set_title(f'Distribution of Ratings by Feature - Year {year}')
                ax.set_xticks(x)
                ax.set_xticklabels([f.replace('_', ' ').title() for f in features],
                                  rotation=45, ha='right')

                # Add legend
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                # Add grid
                ax.grid(True, axis='y', alpha=0.3)

                # Adjust layout
                plt.tight_layout()

                # Create canvas for this year's plot
                canvas_widget = FigureCanvasTkAgg(fig, master=year_frame)
                canvas_widget.draw()
                canvas_widget.get_tk_widget().pack(fill=tk.X)

                # Add separator
                ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)

            # Configure the canvas to update scroll region when the frame changes
            def configure_scroll_region(event):
                canvas.configure(scrollregion=canvas.bbox("all"))

            main_frame.bind('<Configure>', configure_scroll_region)

            # Add mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            canvas.bind_all("<MouseWheel>", on_mousewheel)

        except Exception as e:
            logging.error(f"Error in plot_distribution_comparison: {str(e)}")
            messagebox.showerror("Error", f"Error creating distribution plot: {str(e)}")

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
        """Create visualization comparing current performance with baseline"""
        try:
            # Clear existing content
            for widget in self.baseline_tab.winfo_children():
                widget.destroy()

            if not comparison_data or len(comparison_data) == 0:
                label = ttk.Label(self.baseline_tab, text="No baseline comparison data available.")
                label.pack(pady=20)
                return

            # Get features from comparison data
            features = list(comparison_data.keys())

            # Create a simple frame for the visualization
            main_frame = ttk.Frame(self.baseline_tab)
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

            # Try to create the chart visualization iinf we have matplotlib
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

                # Remove toolbar code
                # No toolbar for cleaner UI

            except Exception as e:
                logging.error(f"Error creating baseline chart: {str(e)}")
                error_label = ttk.Label(main_frame, text=f"Could not create chart visualization: {str(e)}")
                error_label.pack(pady=10)

            # Add summary text
            summary_frame = ttk.LabelFrame(main_frame, text="Summary")
            summary_frame.pack(fill=tk.X, pady=10)

            summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD)
            summary_text.pack(fill=tk.X, padx=5, pady=5)

            # Add summary of findings
            summary = "Baseline Comparison Summary:\n\n"
            for feature in features:
                data = comparison_data[feature]
                change = data['pct_change']
                significant = data['significant']

                summary += f"• {feature.replace('_', ' ').title()}: {change:+.1f}% change from baseline"
                if significant:
                    summary += " (Statistically Significant)\n"
                else:
                    summary += " (Not Statistically Significant)\n"

            summary_text.insert(tk.END, summary)
            summary_text.config(state='disabled')

        except Exception as e:
            logging.error(f"Error plotting baseline comparison: {str(e)}")
            for widget in self.baseline_tab.winfo_children():
                widget.destroy()
            error_label = ttk.Label(self.baseline_tab, text=f"Error creating baseline comparison:\n{str(e)}")
            error_label.pack(pady=20)

    def plot_histograms(self):
        """Create a more readable histogram visualization for all selected features"""
        try:
            # Clear previous content in histogram tab
            for widget in self.histogram_tab.winfo_children():
                widget.destroy()

            # Get filtered data
            filtered_df = self.get_filtered_data()
            if filtered_df is None or filtered_df.empty:
                label = ttk.Label(self.histogram_tab, text="No data available for histogram analysis.")
                label.pack(pady=20)
                return

            # Filter out non-numeric columns
            numeric_features = [f for f in self.selected_features if f != 'department_name']
            if not numeric_features:
                label = ttk.Label(self.histogram_tab, text="No numeric features selected for histogram analysis.")
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
                label = ttk.Label(self.histogram_tab, text="No valid data for histogram analysis.")
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
            ax.set_title('Distribution of Ratings by Feature', fontsize=14)
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
            canvas = FigureCanvasTkAgg(fig, master=self.histogram_tab)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Remove toolbar code
            # No toolbar for cleaner UI

        except Exception as e:
            logging.error(f"Error in plot_histograms: {str(e)}")
            messagebox.showerror("Error", f"Error creating histograms: {str(e)}")

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
            fig_overall = plt.Figure(figsize=(12, 6))
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

def main():
    try:
        root = tk.Tk()
        app = AnalysisGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
