import pandas as pd
import tkinter as tk
from tkinter import ttk
import logging

def create_improved_descriptive_stats(self, tab):
    """
    An improved descriptive statistics visualization that shows different years side by side

    Args:
        self: The parent AnalysisGUI class instance
        tab: The tab to place the visualization in
    """
    # Clear previous content
    for widget in tab.winfo_children():
        widget.destroy()

    # Create a canvas with scrollbar
    canvas = tk.Canvas(tab, width=1500)
    scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)

    # Create a frame inside the canvas
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
            header_text = f"Descriptive Statistics - {len(self.datasets)} Years Available ({', '.join(sorted(self.datasets.keys()))})"
        else:
            year = next(iter(self.datasets.keys()))
            header_text = f"Descriptive Statistics - Year {year}"

        header_label = ttk.Label(header_frame, text=header_text, font=("Arial", 12, "bold"))
        header_label.pack(pady=5)

        # Create main content frame with notebook
        content_frame = ttk.Frame(plot_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create the main tabs
        yearly_comparison_tab = ttk.Frame(notebook)
        single_year_tab = ttk.Frame(notebook)

        notebook.add(yearly_comparison_tab, text="Year Comparison")
        notebook.add(single_year_tab, text="Detailed View")

        # YEARLY COMPARISON TAB
        # Create a sub-notebook for different stat types
        comparison_notebook = ttk.Notebook(yearly_comparison_tab)
        comparison_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs for different statistics
        mean_tab = ttk.Frame(comparison_notebook)
        median_tab = ttk.Frame(comparison_notebook)
        stddev_tab = ttk.Frame(comparison_notebook)
        range_tab = ttk.Frame(comparison_notebook)
        cv_tab = ttk.Frame(comparison_notebook)

        comparison_notebook.add(mean_tab, text="Mean Values")
        comparison_notebook.add(median_tab, text="Median Values")
        comparison_notebook.add(stddev_tab, text="Std Deviation")
        comparison_notebook.add(range_tab, text="Range")
        comparison_notebook.add(cv_tab, text="Coef. of Variation")

        # Collect all years and statistics
        years = sorted(self.datasets.keys())
        year_stats = {}
        all_features = set()

        # Process each year's data
        for year in years:
            # Get data for this year
            year_df = self.get_filtered_data(year)

            # Create numeric only dataframe
            numeric_df = year_df.copy()
            if 'department_name' in numeric_df.columns:
                numeric_df = numeric_df.drop('department_name', axis=1)

            # Convert columns to numeric
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                all_features.add(col)

            # Calculate statistics
            summary = numeric_df.describe().T
            summary['range'] = summary['max'] - summary['min']
            summary['cv'] = summary['std'] / summary['mean'] * 100

            year_stats[year] = summary

        # Calculate combined statistics if multiple years
        if len(years) > 1:
            combined_df = filtered_df.copy()
            if 'department_name' in combined_df.columns:
                combined_df = combined_df.drop('department_name', axis=1)

            # Convert columns to numeric
            for col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

            # Calculate statistics
            combined_summary = combined_df.describe().T
            combined_summary['range'] = combined_summary['max'] - combined_summary['min']
            combined_summary['cv'] = combined_summary['std'] / combined_summary['mean'] * 100

            year_stats['Combined'] = combined_summary
            display_years = years + ['Combined']
        else:
            display_years = years

        # Sort features alphabetically
        all_features = sorted(list(all_features))

        # Function to create comparison tables
        def create_comparison_table(parent, stat_type):
            # Create frame for table with fixed width
            table_frame = ttk.Frame(parent)
            table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Create treeview for table
            tree = ttk.Treeview(table_frame, show='headings')

            # Create columns for each year
            tree['columns'] = display_years

            # Configure columns
            for year in display_years:
                tree.heading(year, text=f"Year {year}")
                tree.column(year, width=100, anchor=tk.CENTER)

            # Add rows for each feature
            for feature in all_features:
                feature_name = feature.replace('_', ' ').title()
                values = []

                for year in display_years:
                    if feature in year_stats[year].index:
                        value = year_stats[year].loc[feature, stat_type]
                        values.append(f"{value:.2f}")
                    else:
                        values.append("N/A")

                tree.insert('', 'end', text=feature_name, values=values)

            # Add scrollbars
            vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

            # Layout
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)

            return tree

        # Create tables for each stat type
        mean_table = create_comparison_table(mean_tab, 'mean')
        median_table = create_comparison_table(median_tab, '50%')
        stddev_table = create_comparison_table(stddev_tab, 'std')
        range_table = create_comparison_table(range_tab, 'range')
        cv_table = create_comparison_table(cv_tab, 'cv')

        # Add explanation to comparison tab
        explanation_text = """
        Statistical Measures Explanation:
        • Mean: The average value of all ratings for this feature
        • Median: The middle value when all ratings are arranged in order
        • Std Dev: Standard deviation - measures how spread out the ratings are
        • Range: The difference between the highest and lowest rating
        • CV: Coefficient of Variation - relative variability as a percentage of the mean

        The comparison view allows you to compare statistics across different years.
        """
        explanation_label = ttk.Label(yearly_comparison_tab, text=explanation_text, justify=tk.LEFT, wraplength=900)
        explanation_label.pack(pady=10)

        # SINGLE YEAR DETAILED VIEW TAB
        # Create year selection frame
        year_selection_frame = ttk.Frame(single_year_tab)
        year_selection_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(year_selection_frame, text="Select Year:").pack(side=tk.LEFT, padx=5)

        # Add combined option if multiple years
        year_options = years.copy()
        if len(years) > 1:
            year_options = ["Combined"] + year_options

        year_var = tk.StringVar(value=year_options[0])
        year_dropdown = ttk.Combobox(year_selection_frame, textvariable=year_var,
                                     values=year_options, state="readonly", width=15)
        year_dropdown.pack(side=tk.LEFT, padx=5)

        # Create stats frame
        stats_frame = ttk.LabelFrame(single_year_tab, text="Detailed Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create text widget for detailed stats
        stats_text = tk.Text(stats_frame, wrap=tk.WORD, width=100, height=30)
        stats_scroll = ttk.Scrollbar(stats_frame, orient="vertical", command=stats_text.yview)
        stats_text.configure(yscrollcommand=stats_scroll.set)

        stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure text tags
        stats_text.tag_configure("heading", font=("Arial", 12, "bold"))
        stats_text.tag_configure("subheading", font=("Arial", 10, "bold"))
        stats_text.tag_configure("feature", font=("Arial", 10, "bold"))
        stats_text.tag_configure("value_good", foreground="green")
        stats_text.tag_configure("value_bad", foreground="red")
        stats_text.tag_configure("value_neutral", foreground="blue")

        # Function to update detailed view
        def update_detailed_stats(event=None):
            # Get selected year
            selected_year = year_var.get()

            # Clear text
            stats_text.config(state=tk.NORMAL)
            stats_text.delete(1.0, tk.END)

            if selected_year == "Combined":
                data = filtered_df
                title = "Combined Years"
            else:
                data = self.get_filtered_data(selected_year)
                title = f"Year {selected_year}"

            # Create numeric dataframe
            numeric_df = data.copy()
            if 'department_name' in numeric_df.columns:
                numeric_df = numeric_df.drop('department_name', axis=1)

            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

            # Get statistics
            stats = numeric_df.describe().T
            stats['range'] = stats['max'] - stats['min']
            stats['cv'] = stats['std'] / stats['mean'] * 100

            # Skewness and kurtosis
            for col in numeric_df.columns:
                stats.loc[col, 'skew'] = numeric_df[col].skew()
                stats.loc[col, 'kurtosis'] = numeric_df[col].kurtosis()

            # Add IQR
            stats['iqr'] = stats['75%'] - stats['25%']

            # Header
            stats_text.insert(tk.END, f"Detailed Statistics for {title}\n\n", "heading")

            # Dataset overview
            stats_text.insert(tk.END, "Dataset Overview:\n", "subheading")
            stats_text.insert(tk.END, f"• Total Events: {len(data)}\n")
            if 'department_name' in data.columns:
                dept_count = data['department_name'].nunique()
                stats_text.insert(tk.END, f"• Departments: {dept_count}\n")
            stats_text.insert(tk.END, f"• Features: {len(all_features)}\n\n")

            # Statistical measures explanation
            stats_text.insert(tk.END, "Statistical Measures Explanation:\n", "subheading")
            stats_text.insert(tk.END, "• Count: Number of non-null observations\n")
            stats_text.insert(tk.END, "• Mean: Average value of all ratings\n")
            stats_text.insert(tk.END, "• Std Dev: Standard deviation - measures data spread\n")
            stats_text.insert(tk.END, "• Min/Max: Minimum and maximum rating values\n")
            stats_text.insert(tk.END, "• 25%/50%/75%: Quartile values (50% is the median)\n")
            stats_text.insert(tk.END, "• Range: Difference between maximum and minimum values\n")
            stats_text.insert(tk.END, "• IQR: Interquartile Range - middle 50% of data\n")
            stats_text.insert(tk.END, "• CV: Coefficient of Variation - relative variability\n")
            stats_text.insert(tk.END, "• Skew: Measure of distribution asymmetry\n")
            stats_text.insert(tk.END, "• Kurtosis: Measure of 'tailedness' of distribution\n\n")

            # Feature details
            stats_text.insert(tk.END, "Feature Details:\n\n", "subheading")

            for feature, row in stats.iterrows():
                feature_name = feature.replace('_', ' ').title()
                stats_text.insert(tk.END, f"{feature_name}:\n", "feature")

                # Basic statistics
                stats_text.insert(tk.END, f"• Count: {row['count']:.0f}\n")

                # Mean (color coded)
                stats_text.insert(tk.END, f"• Mean: ")
                if row['mean'] >= 8.0:
                    stats_text.insert(tk.END, f"{row['mean']:.2f}\n", "value_good")
                elif row['mean'] <= 6.0:
                    stats_text.insert(tk.END, f"{row['mean']:.2f}\n", "value_bad")
                else:
                    stats_text.insert(tk.END, f"{row['mean']:.2f}\n", "value_neutral")

                # Other statistics
                stats_text.insert(tk.END, f"• Median: {row['50%']:.2f}\n")
                stats_text.insert(tk.END, f"• Std Dev: {row['std']:.2f}\n")
                stats_text.insert(tk.END, f"• Min: {row['min']:.2f}\n")
                stats_text.insert(tk.END, f"• 25%: {row['25%']:.2f}\n")
                stats_text.insert(tk.END, f"• 75%: {row['75%']:.2f}\n")
                stats_text.insert(tk.END, f"• Max: {row['max']:.2f}\n")
                stats_text.insert(tk.END, f"• Range: {row['range']:.2f}\n")
                stats_text.insert(tk.END, f"• IQR: {row['iqr']:.2f}\n")

                # CV with interpretation
                stats_text.insert(tk.END, f"• CV: ")
                if row['cv'] < 10:
                    stats_text.insert(tk.END, f"{row['cv']:.2f}% (Low Variability)\n", "value_good")
                elif row['cv'] > 30:
                    stats_text.insert(tk.END, f"{row['cv']:.2f}% (High Variability)\n", "value_bad")
                else:
                    stats_text.insert(tk.END, f"{row['cv']:.2f}% (Moderate Variability)\n", "value_neutral")

                # Skewness with interpretation
                stats_text.insert(tk.END, f"• Skew: ")
                if abs(row['skew']) < 0.5:
                    stats_text.insert(tk.END, f"{row['skew']:.2f} (Approximately Symmetric)\n", "value_good")
                elif row['skew'] > 0.5:
                    stats_text.insert(tk.END, f"{row['skew']:.2f} (Right Skewed - More Lower Values)\n", "value_neutral")
                else:
                    stats_text.insert(tk.END, f"{row['skew']:.2f} (Left Skewed - More Higher Values)\n", "value_neutral")

                # Kurtosis
                stats_text.insert(tk.END, f"• Kurtosis: {row['kurtosis']:.2f}\n\n")

            # Make read-only
            stats_text.config(state=tk.DISABLED)

        # Bind dropdown event
        year_dropdown.bind("<<ComboboxSelected>>", update_detailed_stats)

        # Initial call
        update_detailed_stats()

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
        logging.error(f"Error in improved descriptive stats: {str(e)}")
        label = ttk.Label(plot_frame, text=f"Error during descriptive analysis: {str(e)}")
        label.pack(pady=20)
