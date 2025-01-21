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

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add to global variables section
simulation_tab = None

def analyze_event_ratings(df, low_threshold, high_threshold):
    """Analyze event ratings and identify areas needing improvement and high-performing areas."""
    avg_scores = df.mean().round(2)
    low_scores = avg_scores[avg_scores < low_threshold]
    high_scores = avg_scores[avg_scores >= high_threshold]
    return avg_scores, low_scores, high_scores

def prepare_for_association_rules(df, selected_features, year=None):
    """
    Convert selected features to transactions with ratings categorized as Low/Medium/High.
    
    Parameters:
    - df (DataFrame): The event ratings data
    - selected_features (list): Selected features for analysis
    - year (str, optional): The year for the dataset
    
    Returns:
    - binary_df (DataFrame): Binary representation for each rating level
    """
    def categorize_rating(x):
        if pd.isna(x):
            return 'Missing'
        elif x <= 1.5:
            return 'Low'
        elif x <= 2.5:
            return 'Medium'
        else:
            return 'High'
    
    binary_df = pd.DataFrame()
    for feature in selected_features:
        categories = df[feature].apply(categorize_rating)
        for category in ['Low', 'Medium', 'High']:
            col_name = f"{feature}_{category}"
            binary_df[col_name] = (categories == category).astype(int)
    
    return binary_df

def generate_association_rules(binary_df, min_support=0.05):
    """
    Generate association rules from binary data.
    
    Parameters:
    - binary_df (DataFrame): Binary event features.
    - min_support (float): Minimum support for the apriori algorithm.
    
    Returns:
    - rules (DataFrame): Generated association rules.
    """
    try:
        # Generate frequent itemsets with debug logging
        logging.debug(f"Generating frequent itemsets with min_support={min_support}")
        frequent_itemsets = apriori(binary_df, 
                                  min_support=min_support, 
                                  use_colnames=True,
                                  max_len=None)  # Allow any length itemsets
        
        if frequent_itemsets is None or frequent_itemsets.empty:
            logging.warning("No frequent itemsets found with current support threshold")
            return pd.DataFrame()
            
        logging.debug(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Ensure frequent_itemsets has the required format
        if 'support' not in frequent_itemsets.columns:
            logging.error("Frequent itemsets missing 'support' column")
            return pd.DataFrame()
            
        if 'itemsets' not in frequent_itemsets.columns:
            logging.error("Frequent itemsets missing 'itemsets' column")
            return pd.DataFrame()
        
        # Generate rules with minimum confidence of 0.1
        try:
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence",
                                    min_threshold=0.1)
        except TypeError as te:
            logging.error(f"TypeError in association_rules: {te}")
            # Try alternative format if needed
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence",
                                    min_threshold=0.1,
                                    support_only=True)
        
        if rules.empty:
            logging.warning("No rules generated with current thresholds")
            return pd.DataFrame()
        
        # Ensure rules have required columns
        required_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        if not all(col in rules.columns for col in required_columns):
            logging.error(f"Rules missing required columns. Found: {rules.columns.tolist()}")
            return pd.DataFrame()
        
        # Sort rules by lift for most interesting relationships first
        rules = rules.sort_values('lift', ascending=False)
        
        logging.debug(f"Generated {len(rules)} association rules")
        return rules
        
    except Exception as e:
        logging.error(f"Error generating rules: {str(e)}")
        return pd.DataFrame()

def generate_recommendations_from_rules(self, rules, df, min_lift=1.5, max_recommendations=10):
    """
    Generate recommendations based on association rules analysis.
    Only includes top recommendations based on lift and confidence for features with low ratings.
    Excludes Overall_Rating from recommendations.
    """
    recommendations = {}
    
    if rules.empty:
        return recommendations
    
    # Get rating scale from GUI
    try:
        min_rating = float(self.min_rating_var.get())
        max_rating = float(self.max_rating_var.get())
        low_threshold = float(self.low_threshold_var.get())
    except ValueError:
        logging.error("Invalid rating scale values")
        return recommendations
    
    # Calculate average ratings for each feature
    avg_ratings = df.mean()
    
    # Sort rules by lift and confidence for strongest associations first
    sorted_rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    
    recommendation_count = 0
    
    for _, rule in sorted_rules.iterrows():
        if rule['lift'] >= min_lift and recommendation_count < max_recommendations:
            for antecedent in rule['antecedents']:
                feature, rating = antecedent.rsplit('_', 1)
                
                # Skip if the feature is Overall_Rating
                if feature == 'Overall_Rating':
                    continue
                
                # Skip if feature's average rating is not low
                if feature in avg_ratings and avg_ratings[feature] >= low_threshold:
                    continue
                
                if feature not in recommendations:
                    recommendations[feature] = []
                
                for consequent in rule['consequents']:
                    cons_feature, cons_rating = consequent.rsplit('_', 1)
                    
                    if cons_feature == 'Overall_Rating':
                        continue
                    
                    if rating in ['Low', 'Medium']:
                        recommendation = {
                            'text': f"Improve {feature} (current avg: {avg_ratings[feature]:.2f}/{max_rating:.1f}) to enhance {cons_feature}",
                            'action': f"• Implement targeted improvements in {feature} to achieve {cons_rating} {cons_feature}",
                            'support': rule['support'],
                            'confidence': rule['confidence'],
                            'lift': rule['lift']
                        }
                        
                        if rating == 'Low':
                            recommendation['action'] += f"\n• Conduct immediate review of {feature} processes"
                            recommendation['action'] += f"\n• Set up weekly monitoring of {feature} metrics"
                            recommendation['priority'] = 'High'
                        elif rating == 'Medium':
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
    """
    Generate human-readable interpretations for the association rules.
    
    Parameters:
    - rules (DataFrame): Association rules generated from the data.
    
    Returns:
    - interpretations (str): Interpretations of the association rules.
    """
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
            ant_str.append(f"'{feature}' is {rating}")
        
        cons_str = []
        for cons in consequents:
            feature, rating = cons.rsplit('_', 1)
            cons_str.append(f"'{feature}' is {rating}")
        
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
            self.root = root
            self.root.title("Event Analysis Tool")
            
            # Create navigation bar
            self.create_navigation()
            
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
            
            # Add tabs to notebook
            self.tab_control.add(self.output_tab, text='Analysis Results')
            self.tab_control.add(self.cluster_tab, text='Clustering')
            self.tab_control.add(self.rules_tab, text='Association Rules')
            self.tab_control.add(self.descriptive_tab, text='Descriptive Stats')
            self.tab_control.add(self.histogram_tab, text='Histograms')
            self.tab_control.add(self.distribution_tab, text='Distribution')
            self.tab_control.add(self.recommendations_tab, text='Recommendations')
            
            # Create scrolled text widgets for output and recommendations
            self.output_text = scrolledtext.ScrolledText(self.output_tab, height=30)
            self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.recommendations_text = scrolledtext.ScrolledText(self.recommendations_tab, height=30)
            self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Initialize data storage
            self.datasets = {}
            self.df = None
            self.selected_features = []
            
            print("AnalysisGUI initialized successfully.")
        except Exception as e:
            print(f"Error in AnalysisGUI.__init__: {e}")

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
        
        # Threshold controls (right side)
        # Rating scale frame
        scale_frame = ttk.LabelFrame(right_frame, text="Rating Scale")
        scale_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(scale_frame, text="Min:").pack(side=tk.LEFT, padx=2)
        self.min_rating_var = tk.StringVar(value="1")
        min_entry = ttk.Entry(scale_frame, textvariable=self.min_rating_var, width=5)
        min_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(scale_frame, text="Max:").pack(side=tk.LEFT, padx=2)
        self.max_rating_var = tk.StringVar(value="5")
        max_entry = ttk.Entry(scale_frame, textvariable=self.max_rating_var, width=5)
        max_entry.pack(side=tk.LEFT, padx=2)
        
        # Threshold frame
        threshold_frame = ttk.LabelFrame(right_frame, text="Thresholds")
        threshold_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(threshold_frame, text="Low:").pack(side=tk.LEFT, padx=2)
        self.low_threshold_var = tk.StringVar(value="3")
        low_entry = ttk.Entry(threshold_frame, textvariable=self.low_threshold_var, width=5)
        low_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(threshold_frame, text="High:").pack(side=tk.LEFT, padx=2)
        self.high_threshold_var = tk.StringVar(value="4")
        high_entry = ttk.Entry(threshold_frame, textvariable=self.high_threshold_var, width=5)
        high_entry.pack(side=tk.LEFT, padx=2)
        
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
            
            for file_path in files:
                # Prompt user to enter the year for the current dataset
                year = self.prompt_for_year(file_path)
                if year is None:
                    print(f"Year not provided for {file_path}. Skipping this file.")
                    continue
                
                # Load the data using chunks for large files
                print(f"Loading data from {file_path}...")  # Debug print
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                
                # Clean memory after processing
                del chunks
                gc.collect()
                
                # Try to convert all columns to numeric where possible, skipping the header
                numeric_data = {}
                for col in df.columns:
                    try:
                        # Convert column to numeric, preserving the original values if conversion fails
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        # Only update if we have some valid numeric values
                        if numeric_series.notna().any():
                            numeric_data[col] = numeric_series
                        else:
                            print(f"Column {col} in {file_path} has no valid numeric values")
                    except Exception as e:
                        print(f"Could not convert column {col} in {file_path}: {str(e)}")
                        continue
                
                # Update dataframe with numeric columns
                for col, series in numeric_data.items():
                    df[col] = series
                
                print(f"Data loaded from {file_path}. Shape: {df.shape}")  # Debug print
                print(f"Columns: {df.columns.tolist()}")  # Debug print
                print(f"Column types: {df.dtypes}")  # Debug print
                
                # Check if data was loaded successfully
                if df.empty:
                    print(f"DataFrame from {file_path} is empty")  # Debug print
                    messagebox.showerror("Error", f"No data was loaded from the file: {file_path}")
                    continue
                
                # Store the dataset with the associated year
                self.datasets[year] = df
                print(f"Dataset for year {year} added successfully.")  # Debug print
                
                # Show success message
                self.output_text.insert(tk.END, f"Data loaded successfully for year {year}: {len(df)} records\n")
                self.output_text.insert(tk.END, f"Columns found: {', '.join(df.columns)}\n\n")
            
            # After loading all datasets, concatenate them into self.df
            if self.datasets:
                self.df = pd.concat(self.datasets.values(), ignore_index=True)
                print(f"Combined dataset shape: {self.df.shape}")  # Debug print
                
                # Automatically prompt feature selection after loading all datasets
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
            # Get numeric columns (including those that can be converted to numeric)
            numeric_columns = []
            for col in self.df.columns:
                # Check if column is already numeric
                if np.issubdtype(self.df[col].dtype, np.number):
                    numeric_columns.append(col)
                    print(f"Column {col} is numeric")
                else:
                    # Try converting to numeric
                    try:
                        if pd.to_numeric(self.df[col], errors='coerce').notna().any():
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
            
            # Populate listbox
            for column in numeric_columns:
                self.features_listbox.insert(tk.END, column)
            
            print("Creating buttons")  # Debug print
            # Button frame
            button_frame = ttk.Frame(feature_window)
            button_frame.pack(pady=20, padx=10, fill=tk.X)
            
            # Add Select All and Clear All buttons
            select_all_btn = ttk.Button(
                button_frame,
                text="Select All",
                command=lambda: self.features_listbox.select_set(0, tk.END)
            )
            select_all_btn.pack(side=tk.LEFT, padx=5, expand=True)
            
            clear_all_btn = ttk.Button(
                button_frame,
                text="Clear All",
                command=lambda: self.features_listbox.selection_clear(0, tk.END)
            )
            clear_all_btn.pack(side=tk.LEFT, padx=5, expand=True)
            
            submit_btn = ttk.Button(
                feature_window,
                text="Submit",
                command=lambda: self.submit_feature_selection(feature_window)
            )
            submit_btn.pack(pady=10)
            
            print("Feature selection window created successfully")  # Debug print
            
        except Exception as e:
            print(f"Error in select_features_window: {str(e)}")  # Debug print
            messagebox.showerror("Error", f"Error creating feature selection window: {str(e)}")
    
    def submit_feature_selection(self, feature_window):
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
    
    def plot_clusters(self, df, kmeans, title):
        """
        Plots the K-means clusters using PCA for dimensionality reduction.
        """
        if not self.selected_features:
            self.output_text.insert(tk.END, "Error: No features selected for clustering.\n")
            return

        # Get rating scale parameters
        min_rating = float(self.min_rating_var.get())
        max_rating = float(self.max_rating_var.get())

        # Apply PCA to reduce dimensions to 2 for visualization
        pca = PCA(n_components=2)
        try:
            # Normalize the data before PCA
            normalized_data = df[self.selected_features].copy()
            for feature in self.selected_features:
                normalized_data[feature] = (normalized_data[feature] - min_rating) / (max_rating - min_rating)
            
            # Use downsampling for large datasets
            if len(df) > 10000:
                sample_size = 10000
                df_sample = normalized_data.sample(n=sample_size, random_state=42)
                X_pca = pca.fit_transform(df_sample)
            else:
                X_pca = pca.fit_transform(normalized_data)
            
            # Calculate the explained variance ratio
            explained_variance = pca.explained_variance_ratio_
            
        except Exception as e:
            self.output_text.insert(tk.END, f"Error during PCA: {str(e)}\n")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # Define colors for satisfaction levels
        colors = {
            0: '#FF0000',  # Red for Low
            1: '#FFA500',  # Orange for Moderate
            2: '#00FF00'   # Green for High
        }

        # Create scatter plot with custom colors
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=[colors[label] for label in df['cluster']],
            alpha=0.8,
            label='Attendee Clusters'
        )

        # Plot cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        
        for i, (x, y) in enumerate(centers_pca):
            ax.scatter(
                x, y,
                c=colors[i],
                marker='X',
                s=200,
                linewidths=2,
                edgecolor='black',
                label=f'Cluster {i+1} Center'
            )

        # Add labels and title with explained variance
        ax.set_xlabel(f'First Principal Component\n({explained_variance[0]:.1%} variance explained)')
        ax.set_ylabel(f'Second Principal Component\n({explained_variance[1]:.1%} variance explained)')
        ax.set_title('Attendee Satisfaction Clustering')

        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Add legend with satisfaction levels
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], label='Low Satisfaction', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], label='Moderate Satisfaction', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], label='High Satisfaction', markersize=10),
            plt.Line2D([0], [0], marker='X', color='black', label='Cluster Centers', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='best')

        # Add percentage of variance explained to the plot
        total_variance = sum(explained_variance)
        ax.text(0.02, 0.98, 
                f'Total variance explained: {total_variance:.1%}',
                transform=ax.transAxes,
                verticalalignment='top')

        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.cluster_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_association_rules(self, rules, year):
        """
        Plots the association rules as a network graph for a specific year.
        """
        # Create a frame to hold both the canvas and scrollbar
        frame = ttk.Frame(self.rules_tab)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
        scroll_canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=scroll_canvas.yview)
        
        # Create a frame inside the canvas for the plot
        plot_frame = ttk.Frame(scroll_canvas)
        
        if rules.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f'No significant association rules found for year {year}', 
                    ha='center', va='center')
            ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes and edges with weights
            edge_weights = []
            edge_pairs = []
            edge_colors = []
            
            # Take top 15 rules to show more relationships
            for _, rule in rules.head(15).iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                for ant in antecedents:
                    for cons in consequents:
                        weight = float(rule['lift'])
                        G.add_edge(ant, cons, weight=weight)
                        edge_weights.append(weight)
                        edge_pairs.append((ant, cons))
                        
                        # Determine color based on lift value
                        if weight < 1.2:  # Weak association
                            edge_colors.append('#ffcccc')  # Light red
                        elif weight < 1.5:  # Moderate association
                            edge_colors.append('#ff6666')  # Medium red
                        else:  # Strong association
                            edge_colors.append('#cc0000')  # Dark red
            
            if not edge_weights:
                ax.text(0.5, 0.5, f'No significant associations found for year {year}', 
                        ha='center', va='center')
                ax.set_axis_off()
            else:
                # Draw the graph
                pos = nx.spring_layout(G, k=1)
                
                # Draw edges with varying widths and colors
                for (u, v), weight, color in zip(edge_pairs, edge_weights, edge_colors):
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[(u, v)],
                        width=weight / max(edge_weights) * 5,
                        edge_color=color
                    )
                
                # Draw nodes
                nx.draw_networkx_nodes(
                    G, pos, 
                    node_color='lightblue',
                    node_size=2000,
                    alpha=0.7
                )
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
                
                ax.set_title(f'Association Rules Network - Year {year}\n(Darker red indicates stronger association)', pad=20)
                
                # Add legend with colored lines
                legend_elements = [
                    plt.Line2D([0], [0], color='#ffcccc', linewidth=2, label='Weak (lift < 1.2)'),
                    plt.Line2D([0], [0], color='#ff6666', linewidth=2, label='Moderate (lift 1.2-1.5)'),
                    plt.Line2D([0], [0], color='#cc0000', linewidth=2, label='Strong (lift > 1.5)'),
                    plt.scatter([0], [0], c='lightblue', s=100, label='Rating Category')
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            
            # Remove axis
            ax.set_axis_off()
            
            # Adjust layout
            plt.tight_layout()
            
            # Create the FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure scrolling
        scroll_canvas.create_window((0, 0), window=plot_frame, anchor="nw")
        plot_frame.bind("<Configure>", 
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        
        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)
        
        # Configure the scroll_canvas to use the scrollbar
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
    
    def plot_descriptive(self):
        # Clear previous content in descriptive tab
        for widget in self.descriptive_tab.winfo_children():
            widget.destroy()

        # Create a canvas with scrollbar for descriptive tab
        canvas = tk.Canvas(self.descriptive_tab)
        scrollbar = ttk.Scrollbar(self.descriptive_tab, orient="vertical", command=canvas.yview)
        
        # Create main frame inside canvas
        main_frame = ttk.Frame(canvas)
        
        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas
        canvas.create_window((0, 0), window=main_frame, anchor="nw")

        # Create left frame for summary
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)

        # Create right frame for plot
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)

        # Ensure data is numeric and handle any non-numeric values
        numeric_df = self.df[self.selected_features].apply(pd.to_numeric, errors='coerce')

        # Calculate and display summary statistics using numeric data
        summary_stats = numeric_df.describe()
        target_column = self.selected_features[-1]  # Last selected feature
        
        # Get rating scale and threshold parameters
        min_rating = float(self.min_rating_var.get())
        max_rating = float(self.max_rating_var.get())
        low_threshold = float(self.low_threshold_var.get())
        high_threshold = float(self.high_threshold_var.get())
        
        # Calculate percentages based on thresholds
        low_rate = (numeric_df[target_column] < low_threshold).mean() * 100
        moderate_rate = ((numeric_df[target_column] >= low_threshold) & 
                        (numeric_df[target_column] < high_threshold)).mean() * 100
        high_rate = (numeric_df[target_column] >= high_threshold).mean() * 100

        summary_text = f"""Descriptive Statistics Summary:

📊 Overall Response Analysis:
• Total Responses: {len(numeric_df)}
• Rating Distribution:
  - Low Ratings (<{low_threshold}): {low_rate:.1f}%
  - Moderate Ratings ({low_threshold}-{high_threshold}): {moderate_rate:.1f}%
  - High Ratings (≥{high_threshold}): {high_rate:.1f}%
• Rating Scale: {min_rating:.1f} to {max_rating:.1f}

📈 Distribution Analysis:"""

        for col in self.selected_features:
            col_name = col.replace('_', ' ').title()
            mean = summary_stats.loc['mean', col]
            median = summary_stats.loc['50%', col]
            mode = numeric_df[col].mode().iloc[0]
            std = summary_stats.loc['std', col]
            min_val = summary_stats.loc['min', col]
            max_val = summary_stats.loc['max', col]
            range_val = max_val - min_val
            
            # Calculate percentages for this feature
            low_percent = (numeric_df[col] < low_threshold).mean() * 100
            high_percent = (numeric_df[col] >= high_threshold).mean() * 100
            
            summary_text += f"\n\n• {col_name}:"
            summary_text += f"\n  Central Tendency:"
            summary_text += f"\n    - Mean: {mean:.2f}/{max_rating:.1f}"
            summary_text += f"\n    - Median: {median:.2f}/{max_rating:.1f}"
            summary_text += f"\n    - Mode: {mode:.1f}/{max_rating:.1f}"
            summary_text += f"\n  Variability:"
            summary_text += f"\n    - Range: {range_val:.1f} ({min_val:.1f} to {max_val:.1f})"
            summary_text += f"\n    - Standard Deviation: ±{std:.2f}"
            summary_text += f"\n  Distribution:"
            summary_text += f"\n    - Below {low_threshold}: {low_percent:.1f}%"
            summary_text += f"\n    - Above {high_threshold}: {high_percent:.1f}%"
            
            # Add interpretation based on thresholds
            if mean < low_threshold:
                summary_text += f"\n  ⚠ Area needs attention: {col_name.lower()} (below threshold of {low_threshold})"
            elif mean >= high_threshold:
                summary_text += f"\n  ✓ High performance in {col_name.lower()} (above threshold of {high_threshold})"
            else:
                summary_text += f"\n  → Moderate performance in {col_name.lower()} (between thresholds)"
            
            # Add consistency interpretation based on scale range
            scale_range = max_rating - min_rating
            if std < (scale_range * 0.15):
                summary_text += f"\n  📊 Very consistent ratings"
            elif std < (scale_range * 0.25):
                summary_text += f"\n  📊 Fairly consistent ratings"
            else:
                summary_text += f"\n  📊 Varied ratings - mixed experiences"

        summary_label = ttk.Label(
            left_frame,
            text=summary_text,
            wraplength=400,
            justify="left",
            font=("Arial", 11)
        )
        summary_label.pack(pady=10)

        # Configure the canvas to update scroll region when the frame changes
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        main_frame.bind('<Configure>', configure_scroll_region)

        # Plot histograms in the histogram tab
        self.plot_histograms(numeric_df)

    def plot_histograms(self, numeric_df):
        # Clear previous content in histogram tab
        for widget in self.histogram_tab.winfo_children():
            widget.destroy()

        # Create a canvas with scrollbar for histogram tab
        canvas = tk.Canvas(self.histogram_tab)
        scrollbar = ttk.Scrollbar(self.histogram_tab, orient="vertical", command=canvas.yview)
        
        # Create main frame inside canvas
        main_frame = ttk.Frame(canvas)
        
        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas
        canvas.create_window((0, 0), window=main_frame, anchor="nw")

        # Calculate the number of rows and columns
        n_features = len(self.selected_features)
        n_rows = 4  # Fixed to four rows
        n_cols = (n_features + n_rows - 1) // n_rows  # Ceiling division to get number of columns needed

        # Get rating scale parameters
        min_rating = float(self.min_rating_var.get())
        max_rating = float(self.max_rating_var.get())
        
        # Create bins based on rating scale
        # Add 1 to include the max_rating in the last bin
        bins = np.linspace(min_rating, max_rating, int((max_rating - min_rating) * 2) + 1)

        # Create and display plot using numeric data
        fig = plt.figure(figsize=(12, 3 * n_rows))  # Adjusted figure size for four rows
        
        for idx, feature in enumerate(self.selected_features, 1):
            # Calculate row and column position
            row = (idx - 1) % n_rows  # This ensures we fill rows first
            col = (idx - 1) // n_rows  # Move to next column after filling a row
            plt.subplot(n_rows, n_cols, row * n_cols + col + 1)
            
            numeric_df[feature].hist(bins=bins, edgecolor='black', range=(min_rating, max_rating))
            plt.title(feature.replace('_', ' ').title(), fontsize=10)
            plt.xlabel('Rating', fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tick_params(labelsize=8)
            
            # Set x-axis limits and ticks
            plt.xlim(min_rating, max_rating)
            plt.xticks(np.arange(min_rating, max_rating + 1, 1.0))
        
        plt.suptitle("Distribution of Ratings", fontsize=16, y=1.02)
        plt.tight_layout()
        
        canvas_widget = FigureCanvasTkAgg(fig, master=main_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill='both', expand=True)

        # Configure the canvas to update scroll region when the frame changes
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        main_frame.bind('<Configure>', configure_scroll_region)

        # Add mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)

    def analyze_correlation_strength(self, df, feature):
        """Analyze and interpret correlation strength for a feature"""
        try:
            # Calculate correlations with other features
            correlations = df[self.selected_features].corr()[feature].abs()
            # Remove self-correlation
            correlations = correlations[correlations.index != feature]
            
            # Find correlations at different levels
            strong_correlations = correlations[correlations > 0.5]
            moderate_correlations = correlations[(correlations > 0.3) & (correlations <= 0.5)]
            weak_correlations = correlations[correlations <= 0.3]
            
            interpretation = []
            
            # List features by correlation strength
            if not strong_correlations.empty:
                interpretation.append("Strong Interdependence Found:")
                for other_feature, corr in strong_correlations.items():
                    feature_name = other_feature.replace('_', ' ').title()
                    interpretation.append(f"• Strong link with {feature_name} ({corr:.2f})")
                    interpretation.append(f"  - Improvements in {feature_name} likely affect {feature.replace('_', ' ').title()}")
                    interpretation.append("  - Consider joint improvement initiatives")
            
            if not moderate_correlations.empty:
                if interpretation:
                    interpretation.append("\n")
                interpretation.append("Moderate Relationships Found:")
                for other_feature, corr in moderate_correlations.items():
                    feature_name = other_feature.replace('_', ' ').title()
                    interpretation.append(f"• Moderate link with {feature_name} ({corr:.2f})")
                    interpretation.append(f"  - Some connection exists with {feature_name}")
                    interpretation.append("  - Monitor for potential relationships")
            
            if len(weak_correlations) > 0:
                if interpretation:
                    interpretation.append("\n")
                interpretation.append("Independence Analysis:")
                interpretation.append("Independent from these features:")
                for other_feature, corr in weak_correlations.items():
                    feature_name = other_feature.replace('_', ' ').title()
                    interpretation.append(f"• {feature_name} ({corr:.2f})")
            
            # Add balanced improvement strategy based on actual relationships
            interpretation.extend([
                "\nBalanced Improvement Strategy:",
                "1. Coordinated Improvements:",
            ])
            
            if not strong_correlations.empty:
                interpretation.append("   Strong relationships - Joint actions needed for:")
                for other_feature, _ in strong_correlations.items():
                    feature_name = other_feature.replace('_', ' ').title()
                    interpretation.append(f"   • {feature.replace('_', ' ').title()} with {feature_name}")
                    interpretation.append("     - Create combined improvement plans")
                    interpretation.append("     - Set joint performance targets")
            else:
                interpretation.append("   No strong relationships found - Focus on independent improvements")
            
            interpretation.append("\n2. Moderate Attention Areas:")
            if not moderate_correlations.empty:
                for other_feature, _ in moderate_correlations.items():
                    feature_name = other_feature.replace('_', ' ').title()
                    interpretation.append(f"   • Monitor {feature_name}")
                    interpretation.append("     - Track for emerging patterns")
                    interpretation.append("     - Consider indirect effects")
            else:
                interpretation.append("   No moderate relationships found")
            
            interpretation.append("\n3. Independent Focus Areas:")
            if weak_correlations.empty:
                interpretation.append("   No fully independent features found")
            else:
                interpretation.append("   Separate improvement plans needed for:")
                for other_feature, _ in weak_correlations.items():
                    feature_name = other_feature.replace('_', ' ').title()
                    interpretation.append(f"   • {feature_name}")
                    interpretation.append("     - Develop targeted strategies")
                    interpretation.append("     - Set independent metrics")
            
            # Add practical implementation steps
            interpretation.extend([
                "\nImplementation Plan:",
                "1. Resource Allocation:",
                f"   • {len(strong_correlations)} features need coordinated resources",
                f"   • {len(moderate_correlations)} features need monitoring",
                f"   • {len(weak_correlations)} features need independent focus",
                "\n2. Action Steps:",
                "   • For Linked Features:",
                "     - Create cross-functional teams",
                "     - Develop integrated improvement plans",
                "     - Set joint performance metrics",
                "   • For Independent Features:",
                "     - Assign dedicated resources",
                "     - Create focused action plans",
                "     - Track individual progress",
                "\n3. Monitoring Strategy:",
                "   • Track correlations over time",
                "   • Adjust plans based on emerging patterns",
                "   • Regular review of improvement impacts"
            ])
            
            return "\n".join(interpretation)
        
        except Exception as e:
            logging.error(f"Error analyzing correlations: {e}")
            return "Error analyzing correlations"

    def run_analysis(self):
        # Add progress bar
        progress = ttk.Progressbar(self.root, mode='determinate')
        progress.pack(fill=tk.X, padx=10, pady=5)
        
        try:
            total_steps = len(self.datasets) * 3  # 3 operations per dataset
            current_step = 0
            
            # Clear previous output and plots
            self.output_text.delete(1.0, tk.END)
            # Clear cluster and rules tabs
            for widget in self.cluster_tab.winfo_children():
                widget.destroy()
            for widget in self.rules_tab.winfo_children():
                widget.destroy()
            
            # Check if features are selected
            if not self.selected_features:
                self.output_text.insert(tk.END, "No features selected. Prompting to select features.\n")
                self.select_features_window()
                if not self.selected_features:
                    self.output_text.insert(tk.END, "No features selected. Analysis aborted.\n")
                    progress.destroy()
                    return  # Exit the method if no features are selected
            
            # Get parameters
            low_threshold = float(self.low_threshold_var.get())
            high_threshold = float(self.high_threshold_var.get())
            
            # Check if datasets are loaded
            if not self.datasets or len(self.datasets) == 0:
                self.output_text.insert(tk.END, "Error: Please load at least one dataset before running analysis.\n")
                progress.destroy()
                return
            
            # Perform clustering on the combined dataset
            self.output_text.insert(tk.END, f"=== Performing Clustering Analysis ===\n")
            progress['value'] = (current_step / total_steps) * 100
            self.root.update_idletasks()
            
            avg_scores_combined, low_scores_combined, high_scores_combined = analyze_event_ratings(
                self.df, low_threshold, high_threshold
            )
            current_step += 1
            
            # Ensure clustering is applied to the entire dataset of attendees
            progress['value'] = (current_step / total_steps) * 100
            self.root.update_idletasks()
            
            clustered_df, kmeans, cluster_sizes = self.cluster_events(self.df, self.selected_features)
            self.output_text.insert(tk.END, "Clustering completed.\n\n")
            current_step += 1
            
            # Plot clustering results
            self.plot_clusters(clustered_df, kmeans, "Combined Dataset")
            
            # Display average scores
            self.output_text.insert(tk.END, "Average Event Ratings:\n")
            self.output_text.insert(tk.END, str(avg_scores_combined) + "\n\n")
            
            # Year-by-year analysis
            for year, df in self.datasets.items():
                progress['value'] = (current_step / total_steps) * 100
                self.root.update_idletasks()
                
                self.output_text.insert(tk.END, f"=== Analysis for Year {str(year)} ===\n")
                
                # Perform event-specific analysis
                avg_scores, low_scores, high_scores = analyze_event_ratings(
                    df, low_threshold, high_threshold
                )
                
                # Generate association rules and plot them
                binary_df = prepare_for_association_rules(df, self.selected_features)
                rules = generate_association_rules(binary_df)
                self.plot_association_rules(rules, year)  # Make sure this line is present
                
                # Display results (without association rules text)
                self.output_text.insert(tk.END, "Average Event Ratings by Category:\n")
                self.output_text.insert(tk.END, str(avg_scores) + "\n\n")
                
                if not low_scores.empty:
                    self.output_text.insert(tk.END, "\n🔍 Areas Needing Improvement:\n")
                    self.output_text.insert(tk.END, "================================\n")
                    
                    for feature in low_scores.index:
                        score = low_scores[feature]
                        self.output_text.insert(tk.END, f"\n⚠️ {feature}: {score:.2f}\n")
                        
                        # Add correlation analysis
                        correlation_insight = self.analyze_correlation_strength(df, feature)
                        self.output_text.insert(tk.END, "\nCorrelation Analysis:\n")
                        self.output_text.insert(tk.END, correlation_insight + "\n")
                        
                        # Find relevant association rules
                        relevant_rules = rules[rules['consequents'].apply(
                            lambda x: any(f"{feature}_High" in item for item in x)
                        )]
                        
                        if not relevant_rules.empty:
                            # Sort rules by lift to get strongest relationships first
                            relevant_rules = relevant_rules.sort_values('lift', ascending=False)
                            
                            self.output_text.insert(tk.END, "\nImprovement Suggestions:\n")
                            for _, rule in relevant_rules.head(3).iterrows():
                                antecedents = [item.rsplit('_', 1)[0] for item in rule['antecedents']]
                                antecedents_str = ', '.join(antecedents)
                                
                                self.output_text.insert(tk.END, 
                                    f"  • Focus on improving {antecedents_str}\n"
                                    f"    - This has shown a strong correlation (lift: {rule['lift']:.2f}) "
                                    f"with higher {feature} ratings\n"
                                    f"    - Confidence: {rule['confidence']:.1%}\n"
                                )
                        else:
                            self.output_text.insert(tk.END, 
                                "\nImprovement Strategy for Independent Feature:\n"
                                "• This feature shows independence from other measures\n"
                                "• Focus on targeted improvements specific to this area\n"
                                "• Consider collecting detailed feedback about this aspect\n"
                                "• Review industry best practices for this specific feature\n"
                            )
                        
                        self.output_text.insert(tk.END, "\n---\n")
                
                if not high_scores.empty:
                    self.output_text.insert(tk.END, "\nHigh Performing Areas:\n")
                    self.output_text.insert(tk.END, str(high_scores) + "\n\n")
                
                # Separate analyses by lines
                self.output_text.insert(tk.END, "\n" + "="*50 + "\n\n")
                current_step += 1
            
            # Add descriptive analysis after loading data
            self.plot_descriptive()
            
            # Add cluster distribution to Analysis Results
            self.output_text.insert(tk.END, "\n=== Attendee Clustering Analysis ===\n")
            self.output_text.insert(tk.END, "Respondent Performance Categories:\n")
            self.output_text.insert(tk.END, "--------------------------------\n")
            
            total_respondents = len(self.df)
            for category in ['Low', 'Moderate', 'High']:
                count = cluster_sizes.get(category, 0)
                percentage = (count / total_respondents) * 100
                
                # Format the output with category-specific indicators
                if category == 'Low':
                    indicator = "!"
                elif category == 'High':
                    indicator = "*"
                else:
                    indicator = "-"
                
                self.output_text.insert(tk.END, f"{indicator} {category} Performance Respondents:\n")
                self.output_text.insert(tk.END, f"   • Count: {count} respondents\n")
                self.output_text.insert(tk.END, f"   • Percentage: {percentage:.1f}%\n\n")
            
            # Add summary insight
            major_category = max(cluster_sizes.items(), key=lambda x: x[1])[0]
            major_count = cluster_sizes[major_category]
            major_percentage = (major_count / total_respondents) * 100
            
            self.output_text.insert(tk.END, "Overall Distribution Insight:\n")
            self.output_text.insert(tk.END, f"Majority of respondents ({major_percentage:.1f}%) fall in the {major_category} performance category.\n\n")
            
            # Plot cluster distribution
            self.plot_clusters_for_year(clustered_df, kmeans, clustered_df['cluster_label'], "Combined")

            # Display recommendations in the recommendations tab
            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, "=== Improvement Recommendations ===\n\n")
            
            # Generate dynamic recommendations
            recommendations = self.generate_dynamic_recommendations(
                self.df, 
                low_scores_combined, 
                high_scores_combined,
                float(self.low_threshold_var.get()),
                float(self.high_threshold_var.get())
            )
            
            # Display recommendations
            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, "=== Recommendations ===\n\n")
            
            for feature, recs in recommendations.items():
                self.recommendations_text.insert(tk.END, f"{feature.replace('_', ' ').title()}:\n")
                self.recommendations_text.insert(tk.END, "-" * 40 + "\n")
                
                for rec in recs:
                    self.recommendations_text.insert(tk.END, f"Priority: {rec['priority']}\n")
                    self.recommendations_text.insert(tk.END, f"Recommendation: {rec['text']}\n\n")
                    self.recommendations_text.insert(tk.END, "Actions:\n")
                    self.recommendations_text.insert(tk.END, f"{rec['action']}\n\n")
            
            # Generate maintenance recommendations for high-performing areas
            maintenance_recommendations = generate_event_maintenance_recommendations(high_scores_combined)
            
            self.recommendations_text.insert(tk.END, "\n\n=== Maintenance Recommendations ===\n")
            self.recommendations_text.insert(tk.END, "For high-performing areas:\n\n")
            
            for feature, recs in maintenance_recommendations.items():
                self.recommendations_text.insert(tk.END, f"\n{feature.replace('_', ' ')}:\n")
                self.recommendations_text.insert(tk.END, "------------------------\n")
                for rec in recs:
                    self.recommendations_text.insert(tk.END, f"\n{rec['text']}\n")
                    self.recommendations_text.insert(tk.END, f"Action Steps:\n{rec['action']}\n")
            
            # Calculate and display trends
            trend_data = {}
            for year, df in sorted(self.datasets.items()):
                clustered_df, _, cluster_sizes = self.cluster_events(df, self.selected_features)
                total = len(df)
                
                # Calculate percentages for each category
                trend_data[str(year)] = {
                    category: (cluster_sizes.get(category, 0) / total * 100)
                    for category in ['Low', 'Moderate', 'High']
                }
            
            # Display trends
            for category in ['Low', 'Moderate', 'High']:
                self.output_text.insert(tk.END, f"\n{category} Satisfaction Group Trend:\n")
                for year in sorted(trend_data.keys()):
                    percentage = trend_data[year][category]
                    self.output_text.insert(tk.END, f"   • Year {year}: {percentage:.1f}%\n")
                
                # Calculate year-over-year change if multiple years
                if len(trend_data) > 1:
                    years = sorted(trend_data.keys())
                    first_year = trend_data[years[0]][category]
                    last_year = trend_data[years[-1]][category]
                    change = last_year - first_year
                    self.output_text.insert(tk.END, f"   • Overall Change: {change:+.1f}%\n")
            
            # Plot overall trend
            self.plot_clustering_trends(trend_data)
            
            # Calculate baseline metrics
            baseline_metrics = self.calculate_baseline_performance(self.datasets)
            
            # Compare current performance against baseline
            current_year = max(self.datasets.keys())
            current_data = self.datasets[current_year]
            comparison_results = self.compare_against_baseline(current_data, baseline_metrics)
            
            # Add baseline comparison to output
            self.output_text.insert(tk.END, "\n=== Baseline Performance Comparison ===\n")
            for feature, results in comparison_results.items():
                self.output_text.insert(tk.END, f"\n{feature}:\n")
                self.output_text.insert(tk.END, f"• Performance: {results['performance']}\n")
                self.output_text.insert(tk.END, f"• Change from baseline: {results['pct_change']:+.1f}%\n")
                
                # Add specific insights based on performance
                if results['performance'] == 'Significant Improvement':
                    self.output_text.insert(tk.END, "• Successfully exceeded baseline expectations\n")
                    self.output_text.insert(tk.END, "• Document and share successful strategies\n")
                elif results['performance'] == 'Significant Decline':
                    self.output_text.insert(tk.END, "• Requires immediate attention and intervention\n")
                    self.output_text.insert(tk.END, "• Review changes from baseline period\n")
            
            # Create baseline comparison visualization
            self.plot_baseline_comparison(comparison_results)
            
        except Exception as e:
            logging.error(f"Error in run_analysis: {e}")
            messagebox.showerror("Error", f"Error during analysis: {str(e)}")
        finally:
            progress.destroy()

    def plot_clusters_for_year(self, clustered_df, kmeans, cluster_labels, year):
        """Plots the distribution of clusters for a specific year"""
        try:
            if not self.selected_features:
                self.output_text.insert(tk.END, f"Error: No features selected for clustering in year {year}.\n")
                return

            # Clear previous content in distribution tab
            for widget in self.distribution_tab.winfo_children():
                widget.destroy()

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Define consistent colors
            colors = {
                'Low': '#FF0000',      # Red
                'Moderate': '#FFA500',  # Orange
                'High': '#00FF00'      # Green
            }

            # Distribution plot
            cluster_sizes = clustered_df['cluster_label'].value_counts()
            bars = ax.bar(cluster_sizes.index, cluster_sizes.values)
            
            # Color the bars according to category
            for bar, category in zip(bars, cluster_sizes.index):
                bar.set_color(colors[category])
                bar.set_alpha(0.7)
        
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')

            # Customize plot
            ax.set_title(f'Distribution of Attendees Across Satisfaction Levels (Year {year})', pad=20)
            ax.set_xlabel('Satisfaction Level')
            ax.set_ylabel('Number of Attendees')
        
            # Add grid for better readability
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_axisbelow(True)  # Put grid behind bars

            plt.tight_layout()

            # Embed plot in GUI
            canvas = FigureCanvasTkAgg(fig, master=self.distribution_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            logging.error(f"Error in plot_clusters_for_year: {e}")
            raise

    def cluster_events(self, df, selected_features):
        """Perform K-means clustering on the attendee feedback data"""
        try:
            # Get rating scale parameters
            min_rating = float(self.min_rating_var.get())
            max_rating = float(self.max_rating_var.get())
            
            # Normalize data based on rating scale
            cluster_data = df[selected_features].copy()
            for feature in selected_features:
                cluster_data[feature] = (cluster_data[feature] - min_rating) / (max_rating - min_rating)
            
            # Use 3 clusters for interpretability of attendee satisfaction levels
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_assignments = kmeans.fit_predict(cluster_data)
            
            # Calculate cluster centers and sort them to assign meaningful labels
            centers = kmeans.cluster_centers_ * (max_rating - min_rating) + min_rating
            center_means = centers.mean(axis=1)
            cluster_order = np.argsort(center_means)
            
            # Create mapping from original cluster numbers to satisfaction levels
            cluster_mapping = {
                cluster_order[0]: 'Low',
                cluster_order[1]: 'Moderate',
                cluster_order[2]: 'High'
            }
            
            # Create a new DataFrame with cluster assignments
            clustered_df = df.copy()
            clustered_df['cluster'] = cluster_assignments
            clustered_df['cluster_label'] = clustered_df['cluster'].map(cluster_mapping)
            
            # Calculate cluster sizes using value_counts
            cluster_sizes = clustered_df['cluster_label'].value_counts().to_dict()
            
            return clustered_df, kmeans, cluster_sizes
            
        except Exception as e:
            logging.error(f"Error in cluster_events: {str(e)}")
            raise

    def plot_clustering_trends(self, trend_data):
        """Plot trends of cluster distributions over years"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            years = sorted(trend_data.keys())
            categories = ['Low', 'Moderate', 'High']
            colors = {
                'Low': '#FF0000',      # Red
                'Moderate': '#FFA500',  # Orange
                'High': '#00FF00'      # Green
            }
            
            for category in categories:
                percentages = [trend_data[year][category] for year in years]
                ax.plot(years, percentages, marker='o', 
                       label=f'{category} Satisfaction',
                       linewidth=2, markersize=8, 
                       color=colors[category])
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Percentage of Attendees')
            ax.set_title('Satisfaction Group Trends Over Time')
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
        """Validate all input parameters before analysis"""
        try:
            min_rating = float(self.min_rating_var.get())
            max_rating = float(self.max_rating_var.get())
            low_threshold = float(self.low_threshold_var.get())
            high_threshold = float(self.high_threshold_var.get())
            
            if not (min_rating < low_threshold < high_threshold < max_rating):
                raise ValueError("Invalid threshold values. Must satisfy: min < low < high < max")
            
            return True
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return False

    def generate_dynamic_recommendations(self, df, low_scores, high_scores, low_threshold, high_threshold):
        """Generate detailed recommendations based on comparative analysis of high vs low performing events"""
        recommendations = {}
        
        # Get rating scale parameters
        max_rating = float(self.max_rating_var.get())
        
        for feature in df.columns:
            avg_rating = df[feature].mean()
            std_dev = df[feature].std()
            
            # Separate high and low performing events for comparison
            high_performing = df[df[feature] >= high_threshold]
            low_performing = df[df[feature] < low_threshold]
            
            if feature in low_scores.index:
                recommendations[feature] = []
                
                # Calculate detailed metrics
                low_percent = (df[feature] < low_threshold).mean() * 100
                
                # Compare characteristics between high and low performing events
                comparative_analysis = {
                    'timing': self.compare_event_timing(high_performing, low_performing),
                    'resources': self.compare_resource_utilization(high_performing, low_performing),
                    'participation': self.compare_participation_metrics(high_performing, low_performing)
                }
                
                # Generate specific recommendations based on comparative analysis
                rec = {
                    'text': f"Improve {feature.replace('_', ' ').title()} (current: {avg_rating:.2f}/{max_rating})",
                    'priority': 'High' if low_percent > 30 else 'Medium',
                }
                
                actions = [
                    f"Current Performance Metrics:",
                    f"• Average rating: {avg_rating:.2f}/{max_rating}",
                    f"• {low_percent:.1f}% of events need improvement",
                    f"\nKey Differences in High vs Low Performing Events:"
                ]
                
                # Add specific insights from comparative analysis
                if comparative_analysis['timing']:
                    actions.extend([
                        "\nTiming and Schedule:",
                        f"• {comparative_analysis['timing']}"
                    ])
                
                if comparative_analysis['resources']:
                    actions.extend([
                        "\nResource Management:",
                        f"• {comparative_analysis['resources']}"
                    ])
                
                if comparative_analysis['participation']:
                    actions.extend([
                        "\nParticipation Patterns:",
                        f"• {comparative_analysis['participation']}"
                    ])
                
                # Add specific improvement actions
                actions.extend([
                    "\nRecommended Actions:",
                    "1. Short-term Improvements:",
                    "   • Implement best practices from high-performing events",
                    f"   • Focus on {feature.replace('_', ' ').lower()} standardization",
                    "   • Establish clear success metrics",
                    "",
                    "2. Long-term Strategy:",
                    "   • Develop comprehensive improvement plan",
                    "   • Regular monitoring and evaluation",
                    "   • Staff training and development"
                ])
                
                rec['action'] = "\n".join(actions)
                recommendations[feature].append(rec)
                
            elif feature in high_scores.index:
                recommendations[feature] = []
                
                # Generate maintenance recommendations for high-performing areas
                high_percent = (df[feature] >= high_threshold).mean() * 100
                
                rec = {
                    'text': f"Maintain excellence in {feature.replace('_', ' ').title()}",
                    'priority': 'Maintain',
                    'support': high_percent / 100,
                    'confidence': 0.9,
                    'lift': 1.5
                }
                
                actions = [
                    f"• Strong performance: {high_percent:.1f}% gave high ratings",
                    f"• Current average rating: {avg_rating:.2f}/{max_rating}",
                    "• Document successful practices",
                    f"• Share {feature.replace('_', ' ').lower()} best practices with team",
                    "• Continue monitoring for consistency"
                ]
                
                if std_dev < 0.3:  # Very consistent high ratings
                    actions.extend([
                        "• Excellent consistency in delivery",
                        "• Use as benchmark for other areas"
                    ])
                
                rec['action'] = "\n".join(actions)
                recommendations[feature].append(rec)
            
            else:
                # Moderate performing areas
                recommendations[feature] = []
                
                rec = {
                    'text': f"Optimize {feature.replace('_', ' ').title()}",
                    'priority': 'Medium',
                    'support': 0.5,
                    'confidence': 0.7,
                    'lift': 1.1
                }
                
                actions = [
                    f"• Current average rating: {avg_rating:.2f}/{max_rating}",
                    "• Moderate performance area",
                    f"• Review {feature.replace('_', ' ').lower()} processes for optimization",
                    "• Identify specific enhancement opportunities"
                ]
                
                if std_dev > 0.4:  # Moderate variability
                    actions.extend([
                        "• Address consistency in delivery",
                        "• Implement standardized procedures"
                    ])
                
                rec['action'] = "\n".join(actions)
                recommendations[feature].append(rec)
        
        return recommendations

    def compare_event_timing(self, high_performing, low_performing):
        """Compare timing aspects between high and low performing events"""
        try:
            timing_insights = []
            
            if 'Event_Duration' in high_performing.columns:
                high_duration = high_performing['Event_Duration'].mean()
                low_duration = low_performing['Event_Duration'].mean()
                
                if abs(high_duration - low_duration) > 0.5:  # Significant difference threshold
                    timing_insights.append(
                        f"Successful events average {high_duration:.1f} hours vs "
                        f"{low_duration:.1f} hours for lower-rated events"
                    )
            
            if 'Time_Management' in high_performing.columns:
                high_time_mgmt = high_performing['Time_Management'].mean()
                low_time_mgmt = low_performing['Time_Management'].mean()
                
                if high_time_mgmt > low_time_mgmt:
                    timing_insights.append(
                        "Higher-rated events show better time management scores"
                    )
            
            return "\n• ".join(timing_insights) if timing_insights else None
            
        except Exception as e:
            logging.error(f"Error in compare_event_timing: {e}")
            return None

    def compare_resource_utilization(self, high_performing, low_performing):
        """Compare resource utilization between high and low performing events"""
        try:
            resource_insights = []
            
            resource_columns = [
                'Budget_Utilization', 'Resource_Efficiency',
                'Staff_Allocation', 'Equipment_Usage'
            ]
            
            for col in resource_columns:
                if col in high_performing.columns:
                    high_resource = high_performing[col].mean()
                    low_resource = low_performing[col].mean()
                    
                    diff_percent = ((high_resource - low_resource) / low_resource) * 100
                    
                    if abs(diff_percent) > 10:  # Significant difference threshold
                        resource_insights.append(
                            f"{col.replace('_', ' ')}: {diff_percent:+.1f}% difference "
                            "in high vs low performing events"
                        )
            
            return "\n• ".join(resource_insights) if resource_insights else None
            
        except Exception as e:
            logging.error(f"Error in compare_resource_utilization: {e}")
            return None

    def compare_participation_metrics(self, high_performing, low_performing):
        """Compare participation metrics between high and low performing events"""
        try:
            participation_insights = []
            
            if 'Attendance_Rate' in high_performing.columns:
                high_attendance = high_performing['Attendance_Rate'].mean()
                low_attendance = low_performing['Attendance_Rate'].mean()
                
                diff_percent = ((high_attendance - low_attendance) / low_attendance) * 100
                
                if abs(diff_percent) > 5:  # Significant difference threshold
                    participation_insights.append(
                        f"Attendance rate is {diff_percent:+.1f}% higher "
                        "in successful events"
                    )
            
            if 'Engagement_Score' in high_performing.columns:
                high_engagement = high_performing['Engagement_Score'].mean()
                low_engagement = low_performing['Engagement_Score'].mean()
                
                if high_engagement > low_engagement:
                    participation_insights.append(
                        f"Participant engagement scores are {((high_engagement/low_engagement)-1)*100:.1f}% "
                        "higher in successful events"
                    )
            
            return "\n• ".join(participation_insights) if participation_insights else None
            
        except Exception as e:
            logging.error(f"Error in compare_participation_metrics: {e}")
            return None

    def plot_distribution_comparison(self, results):
        """Plot distribution comparison visualization"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Define consistent colors
            colors = {
                'Low': '#FF0000',      # Red
                'Moderate': '#FFA500',  # Orange
                'High': '#00FF00'      # Green
            }
            
            features = list(results.keys())
            x = np.arange(len(features))
            width = 0.25  # Width of bars
            
            # Plot bars for each category
            for i, category in enumerate(['Low', 'Moderate', 'High']):
                percentages = [results[f].get(category, 0) for f in features]
                ax.bar(x + i*width, percentages, width, 
                      label=category, color=colors[category], alpha=0.7)
                
                # Add percentage labels on bars
                for j, v in enumerate(percentages):
                    ax.text(x[j] + i*width, v, f'{v:.1f}%',
                           ha='center', va='bottom')
            
            ax.set_ylabel('Percentage of Attendees')
            ax.set_title('Distribution of Satisfaction Levels by Feature')
            ax.set_xticks(x + width)
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            
            # Create canvas in distribution tab
            canvas = FigureCanvasTkAgg(fig, master=self.distribution_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            logging.error(f"Error in plot_distribution_comparison: {e}")
            raise

    def calculate_baseline_performance(self, historical_data):
        """
        Calculate baseline performance metrics from historical data.
        
        Parameters:
        - historical_data (dict): Dictionary of DataFrames containing historical event data
        
        Returns:
        - dict: Baseline metrics for each feature
        """
        try:
            # Get the earliest years that will form the baseline
            sorted_years = sorted(historical_data.keys())
            baseline_years = sorted_years[:2]  # Use first two years as baseline
            
            baseline_metrics = {}
            
            # Combine data from baseline years
            baseline_data = pd.concat([historical_data[year] for year in baseline_years])
            
            for feature in self.selected_features:
                baseline_metrics[feature] = {
                    'mean': baseline_data[feature].mean(),
                    'std': baseline_data[feature].std(),
                    'percentiles': {
                        'low': baseline_data[feature].quantile(0.25),
                        'median': baseline_data[feature].median(),
                        'high': baseline_data[feature].quantile(0.75)
                    },
                    'years': baseline_years
                }
                
            return baseline_metrics
            
        except Exception as e:
            logging.error(f"Error calculating baseline performance: {e}")
            raise

    def compare_against_baseline(self, current_data, baseline_metrics):
        """
        Compare current performance against baseline metrics.
        
        Parameters:
        - current_data (DataFrame): Current event data
        - baseline_metrics (dict): Baseline metrics from calculate_baseline_performance
        
        Returns:
        - dict: Comparison results for each feature
        """
        try:
            comparison_results = {}
            
            for feature in self.selected_features:
                current_mean = current_data[feature].mean()
                baseline_mean = baseline_metrics[feature]['mean']
                baseline_std = baseline_metrics[feature]['std']
                
                # Calculate z-score to determine significance of change
                z_score = (current_mean - baseline_mean) / baseline_std if baseline_std != 0 else 0
                
                # Calculate percentage change
                pct_change = ((current_mean - baseline_mean) / baseline_mean) * 100
                
                # Determine performance category
                if z_score > 1.96:  # 95% confidence level
                    performance = 'Significant Improvement'
                elif z_score < -1.96:
                    performance = 'Significant Decline'
                elif abs(z_score) <= 0.5:
                    performance = 'Stable'
                elif z_score > 0:
                    performance = 'Slight Improvement'
                else:
                    performance = 'Slight Decline'
                
                comparison_results[feature] = {
                    'current_mean': current_mean,
                    'baseline_mean': baseline_mean,
                    'pct_change': pct_change,
                    'z_score': z_score,
                    'performance': performance,
                    'baseline_percentiles': baseline_metrics[feature]['percentiles']
                }
            
            return comparison_results
            
        except Exception as e:
            logging.error(f"Error comparing against baseline: {e}")
            raise

    def plot_baseline_comparison(self, comparison_results):
        """
        Create visualization comparing current performance against baseline.
        """
        try:
            # Clear previous content in the tab
            if not hasattr(self, 'baseline_tab'):
                self.baseline_tab = ttk.Frame(self.tab_control)
                self.tab_control.add(self.baseline_tab, text='Baseline Comparison')
            else:
                for widget in self.baseline_tab.winfo_children():
                    widget.destroy()
            
            # Create figure with subplots
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(2, 2)
            
            # 1. Performance Change Plot
            ax1 = fig.add_subplot(gs[0, 0])
            features = list(comparison_results.keys())
            pct_changes = [results['pct_change'] for results in comparison_results.values()]
            
            colors = ['#2ecc71' if pct > 0 else '#e74c3c' for pct in pct_changes]
            bars = ax1.bar(features, pct_changes, color=colors)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_title('Performance Change from Baseline')
            ax1.set_ylabel('Percentage Change (%)')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:+.1f}%',
                        ha='center', va='bottom' if height > 0 else 'top')
            
            # 2. Current vs Baseline Comparison
            ax2 = fig.add_subplot(gs[0, 1])
            x = np.arange(len(features))
            width = 0.35
            
            current_means = [results['current_mean'] for results in comparison_results.values()]
            baseline_means = [results['baseline_mean'] for results in comparison_results.values()]
            
            ax2.bar(x - width/2, baseline_means, width, label='Baseline', color='#3498db', alpha=0.7)
            ax2.bar(x + width/2, current_means, width, label='Current', color='#e67e22', alpha=0.7)
            
            ax2.set_title('Current vs Baseline Ratings')
            ax2.set_xticks(x)
            ax2.set_xticklabels(features, rotation=45, ha='right')
            ax2.legend()
            
            # 3. Performance Distribution
            ax3 = fig.add_subplot(gs[1, :])
            
            # Create box plot data
            box_data = []
            labels = []
            for feature in features:
                baseline_percentiles = comparison_results[feature]['baseline_percentiles']
                box_data.append([
                    baseline_percentiles['low'],
                    baseline_percentiles['median'],
                    baseline_percentiles['high']
                ])
                labels.append(feature)
            
            # Plot baseline distribution
            bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
            
            # Add current performance points
            current_points = [results['current_mean'] for results in comparison_results.values()]
            ax3.plot(range(1, len(features) + 1), current_points, 'ro', label='Current Performance')
            
            ax3.set_title('Performance Distribution')
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            ax3.legend()
            
            plt.tight_layout()
            
            # Embed plot in GUI
            canvas = FigureCanvasTkAgg(fig, master=self.baseline_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            logging.error(f"Error plotting baseline comparison: {e}")
            raise

def main():
    try:
        root = tk.Tk()
        app = AnalysisGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()  