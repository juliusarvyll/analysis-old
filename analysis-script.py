import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel, QTabWidget, QComboBox, QSizePolicy, QSpacerItem, QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QTableWidget, QTableWidgetItem, QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
import networkx as nx
import sys
from collections import Counter
import re

# --- CONFIG ---
N_CLUSTERS = 3  # Number of clusters for KMeans
MIN_SUPPORT = 0.1  # Minimum support for Apriori
MIN_CONFIDENCE = 0.5  # Minimum confidence for association rules

# Set global font size for matplotlib
mpl.rcParams.update({'font.size': 22, 'axes.titlesize': 28, 'axes.labelsize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})

class RatingRangesDialog(QDialog):
    def __init__(self, ranges, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Set Rating Ranges')
        self.setMinimumWidth(400)
        layout = QFormLayout(self)
        self.range_edits = []
        for label, (low, high) in ranges.items():
            edit = QLineEdit(f"{low},{high if high is not None else ''}")
            layout.addRow(QLabel(label), edit)
            self.range_edits.append((label, edit))
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    def get_ranges(self):
        new_ranges = {}
        for label, edit in self.range_edits:
            text = edit.text().replace(' ', '')
            if ',' in text:
                low, high = text.split(',', 1)
                try:
                    low = float(low)
                except:
                    low = 0.0
                try:
                    high = float(high) if high != '' else None
                except:
                    high = None
                new_ranges[label] = (low, high)
        return new_ranges

class AnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Clustering & Association Rule Mining')
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        # Get screen width dynamically
        screen = QApplication.primaryScreen()
        screen_width = screen.size().width()
        self.setMaximumWidth(screen_width)

        # --- Modern Light Theme Stylesheet ---
        self.setStyleSheet('''
            QWidget {
                background: #f6f8fa;
                font-family: "Segoe UI", "Arial", sans-serif;
                color: #222;
            }
            QTabWidget::pane {
                border: 2px solid #d0d7de;
                border-radius: 18px;
                background: #fff;
                margin: 8px;
            }
            QTabBar::tab {
                background: #e9ecef;
                border: 1px solid #d0d7de;
                border-radius: 12px 12px 0 0;
                min-width: 180px;
                min-height: 60px;
                font-size: 32px;
                padding: 16px 32px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #fff;
                border-bottom: 2px solid #fff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #dde6f1;
            }
            QPushButton {
                background: #2563eb;
                color: #fff;
                border-radius: 12px;
                font-size: 32px;
                padding: 12px 36px;
                border: none;
            }
            QPushButton:hover {
                background: #174ea6;
            }
            QPushButton:pressed {
                background: #0b2547;
            }
            QComboBox {
                background: #fff;
                border: 1.5px solid #d0d7de;
                border-radius: 10px;
                font-size: 32px;
                padding: 10px 20px;
                min-width: 300px;
            }
            QComboBox QAbstractItemView {
                background: #fff;
                selection-background-color: #e9ecef;
                font-size: 32px;
            }
            QLabel {
                font-size: 32px;
            }
            QTextEdit {
                background: #f8fafc;
                border: 1.5px solid #d0d7de;
                border-radius: 10px;
                font-size: 28px;
                padding: 16px;
            }
            QTableWidget {
                background: #fff;
                border: 1.5px solid #d0d7de;
                border-radius: 10px;
                font-size: 28px;
                gridline-color: #d0d7de;
                selection-background-color: #c7d7f5;
                selection-color: #111;
                alternate-background-color: #f6f8fa;
            }
            QHeaderView::section {
                background: #e9ecef;
                font-weight: bold;
                font-size: 30px;
                border-radius: 8px;
                border: 1px solid #d0d7de;
                padding: 8px;
            }
        ''')

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        # Set max width for main layout container
        self.setMinimumWidth(1200)
        self.setMaximumWidth(screen_width)

        # --- Top bar: Logo, Spacer, Exit Button ---
        top_bar = QHBoxLayout()
        self.logo_label = QLabel('LOGO')  # Placeholder for logo
        self.logo_label.setStyleSheet('font-size: 48px; font-weight: bold; color: #3366cc; padding: 10px 30px;')
        self.logo_label.setFixedHeight(100)
        self.logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top_bar.addWidget(self.logo_label)

        top_bar.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.exit_button = QPushButton('Exit')
        self.exit_button.setStyleSheet('font-size: 32px; padding: 18px 48px; background: #e74c3c; color: white; border-radius: 16px;')
        self.exit_button.setFixedHeight(80)
        self.exit_button.setFixedWidth(180)
        self.exit_button.clicked.connect(self.close)
        top_bar.addWidget(self.exit_button)
        main_layout.addLayout(top_bar)

        # --- File selection and department selection ---
        file_dept_layout = QHBoxLayout()
        self.label = QLabel('Select one or more CSV files to analyze:')
        self.label.setStyleSheet('font-size: 32px; padding: 10px 20px;')
        file_dept_layout.addWidget(self.label)

        self.button = QPushButton('Open CSV(s)')
        self.button.setStyleSheet('font-size: 32px; padding: 12px 36px;')
        self.button.setFixedHeight(60)
        self.button.clicked.connect(self.open_csvs)
        file_dept_layout.addWidget(self.button)

        # Add More CSV(s) button
        self.add_button = QPushButton('Add More CSV(s)')
        self.add_button.setStyleSheet('font-size: 32px; padding: 12px 36px;')
        self.add_button.setFixedHeight(60)
        self.add_button.clicked.connect(self.add_more_csvs)
        file_dept_layout.addWidget(self.add_button)

        # Add Set Rating Ranges button
        self.rating_ranges_btn = QPushButton('Set Rating Ranges')
        self.rating_ranges_btn.setStyleSheet('font-size: 28px; padding: 10px 30px;')
        self.rating_ranges_btn.setFixedHeight(60)
        self.rating_ranges_btn.clicked.connect(self.set_rating_ranges)
        file_dept_layout.addWidget(self.rating_ranges_btn)

        # Add Save/Load Project buttons
        self.save_project_btn = QPushButton('Save Project')
        self.save_project_btn.setStyleSheet('font-size: 28px; padding: 10px 30px;')
        self.save_project_btn.setFixedHeight(60)
        self.save_project_btn.clicked.connect(self.save_project)
        file_dept_layout.addWidget(self.save_project_btn)
        self.load_project_btn = QPushButton('Load Project')
        self.load_project_btn.setStyleSheet('font-size: 28px; padding: 10px 30px;')
        self.load_project_btn.setFixedHeight(60)
        self.load_project_btn.clicked.connect(self.load_project)
        file_dept_layout.addWidget(self.load_project_btn)

        # Add Export to PDF button
        self.export_pdf_btn = QPushButton('Export to PDF')
        self.export_pdf_btn.setStyleSheet('font-size: 28px; padding: 10px 30px;')
        self.export_pdf_btn.setFixedHeight(60)
        self.export_pdf_btn.clicked.connect(self.export_to_pdf)
        file_dept_layout.addWidget(self.export_pdf_btn)

        # Add Dataset selection combo box
        dataset_label = QLabel('Select Dataset:')
        dataset_label.setStyleSheet('font-size: 32px; padding: 10px 20px;')
        file_dept_layout.addWidget(dataset_label)
        self.dataset_combo = QComboBox()
        self.dataset_combo.setStyleSheet('font-size: 32px; min-width: 300px; padding: 10px 20px;')
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_change)
        self.dataset_combo.setEnabled(False)
        self.dataset_combo.setFixedHeight(60)
        file_dept_layout.addWidget(self.dataset_combo)

        file_dept_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        dept_label = QLabel('Select Department:')
        dept_label.setStyleSheet('font-size: 32px; padding: 10px 20px;')
        file_dept_layout.addWidget(dept_label)
        self.dept_combo = QComboBox()
        self.dept_combo.setStyleSheet('font-size: 32px; min-width: 300px; padding: 10px 20px;')
        self.dept_combo.currentIndexChanged.connect(self.on_department_change)
        self.dept_combo.setEnabled(False)
        self.dept_combo.setFixedHeight(60)
        file_dept_layout.addWidget(self.dept_combo)

        main_layout.addLayout(file_dept_layout)

        # Tabs for Clustering and Association Rules
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet('font-size: 32px; min-height: 60px;')
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.tabs)

        # --- Clustering Tab ---
        self.cluster_tab = QWidget()
        self.cluster_layout = QVBoxLayout()
        self.cluster_layout.setContentsMargins(0, 0, 0, 0)
        self.cluster_layout.setSpacing(16)
        self.cluster_figure = Figure(dpi=150)
        self.cluster_canvas = FigureCanvas(self.cluster_figure)
        self.cluster_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cluster_layout.addWidget(self.cluster_canvas)
        self.cluster_tab.setLayout(self.cluster_layout)
        self.tabs.addTab(self.cluster_tab, "Clustering")

        # --- Association Rules Tab ---
        self.arm_tab = QWidget()
        self.arm_layout = QVBoxLayout()
        self.arm_layout.setContentsMargins(0, 0, 0, 0)
        self.arm_layout.setSpacing(16)
        self.arm_figure = Figure(dpi=150)
        self.arm_canvas = FigureCanvas(self.arm_figure)
        self.arm_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.arm_canvas.setMinimumHeight(500)  # Increased height for ARM plot
        self.arm_layout.addWidget(self.arm_canvas)
        self.arm_tab.setLayout(self.arm_layout)
        self.tabs.addTab(self.arm_tab, "Association Rules")
        # --- ARM Results Tab ---
        self.arm_results_tab = QWidget()
        self.arm_results_layout = QVBoxLayout()
        self.arm_results_layout.setContentsMargins(0, 0, 0, 0)
        self.arm_results_layout.setSpacing(16)
        self.arm_table = QTableWidget()
        self.arm_table.setColumnCount(5)
        self.arm_table.setHorizontalHeaderLabels(['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])
        self.arm_table.setStyleSheet('font-size: 24px;')
        self.arm_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.arm_table.setMinimumHeight(350)
        self.arm_table.setWordWrap(True)
        self.arm_table.setAlternatingRowColors(True)
        self.arm_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.arm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.arm_results_scroll = QScrollArea()
        self.arm_results_scroll.setWidgetResizable(True)
        self.arm_results_scroll.setWidget(self.arm_table)
        self.arm_results_layout.addWidget(self.arm_results_scroll)
        self.arm_results_tab.setLayout(self.arm_results_layout)
        self.tabs.addTab(self.arm_results_tab, "ARM Results")

        # --- Descriptive Analysis Tab ---
        self.desc_tab = QWidget()
        self.desc_layout = QVBoxLayout()
        self.desc_layout.setContentsMargins(0, 0, 0, 0)
        self.desc_layout.setSpacing(16)
        # Table for feature ratings
        self.desc_table = QTableWidget()
        self.desc_table.setColumnCount(3)
        self.desc_table.setHorizontalHeaderLabels(['Feature', 'Mean', 'Rating'])
        self.desc_table.setStyleSheet('font-size: 28px;')
        self.desc_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.desc_table.setMinimumHeight(220)
        self.desc_layout.addWidget(self.desc_table)
        # Text area for categorical value counts
        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        self.desc_text.setMinimumHeight(120)
        self.desc_text.setStyleSheet('font-size: 28px; padding: 16px;')
        self.desc_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.desc_layout.addWidget(self.desc_text)
        self.desc_tab.setLayout(self.desc_layout)
        self.tabs.addTab(self.desc_tab, "Descriptive Analysis")

        # --- Recommendations Tab ---
        self.recommend_tab = QWidget()
        self.recommend_layout = QVBoxLayout()
        self.recommend_layout.setContentsMargins(0, 0, 0, 0)
        self.recommend_layout.setSpacing(16)
        self.recommend_text = QTextEdit()
        self.recommend_text.setReadOnly(True)
        self.recommend_text.setMinimumHeight(220)
        self.recommend_text.setStyleSheet('font-size: 28px; padding: 16px;')
        self.recommend_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.recommend_layout.addWidget(self.recommend_text)
        self.recommend_tab.setLayout(self.recommend_layout)
        self.tabs.addTab(self.recommend_tab, "Recommendations")

        # --- Histograms Tab ---
        self.hist_tab = QWidget()
        self.hist_layout = QVBoxLayout()
        self.hist_layout.setContentsMargins(0, 0, 0, 0)
        self.hist_layout.setSpacing(16)
        self.hist_figure = Figure(dpi=150, constrained_layout=True)
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hist_layout.addWidget(self.hist_canvas)
        self.hist_tab.setLayout(self.hist_layout)
        self.tabs.addTab(self.hist_tab, "Histograms")

        # --- Trends Tab ---
        self.trends_tab = QWidget()
        self.trends_layout = QVBoxLayout()
        self.trends_layout.setContentsMargins(0, 0, 0, 0)
        self.trends_layout.setSpacing(16)
        self.trends_figure = Figure(dpi=150)
        self.trends_canvas = FigureCanvas(self.trends_figure)
        self.trends_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.trends_layout.addWidget(self.trends_canvas)
        # Add scrollable text area for Groq analysis
        self.trends_text = QTextEdit()
        self.trends_text.setReadOnly(True)
        self.trends_text.setStyleSheet('font-size: 28px; padding: 16px;')
        self.trends_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.trends_text.setMaximumHeight(350)
        self.trends_scroll = QScrollArea()
        self.trends_scroll.setWidgetResizable(True)
        self.trends_scroll.setWidget(self.trends_text)
        self.trends_layout.addWidget(self.trends_scroll)
        self.trends_tab.setLayout(self.trends_layout)
        self.tabs.addTab(self.trends_tab, "Trends")

        self.setLayout(main_layout)
        self.datasets = {}  # filename -> DataFrame
        self.current_dataset = 'All Datasets'
        self.df_all = None
        # Default rating ranges
        self.rating_ranges = {
            'needs_improvement': (float('-inf'), 0.74),
            'moderately_satisfactory': (0.75, 1.49),
            'satisfactory': (1.50, 2.24),
            'very_satisfactory': (2.25, float('inf'))
        }
        # For all QTextEdit widgets, ensure word wrap and max width
        self.desc_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.desc_text.setMaximumWidth(screen_width - 120)
        self.recommend_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.recommend_text.setMaximumWidth(screen_width - 120)
        self.trends_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.trends_text.setMaximumWidth(screen_width - 120)

    def open_csvs(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, 'Open CSV(s)', '', 'CSV Files (*.csv)')
        if file_paths:
            self.label.setText('Loaded files:\n' + '\n'.join(file_paths))
            # Read and store all CSVs
            self.datasets = {}
            for fp in file_paths:
                df = pd.read_csv(fp)
                self.datasets[fp] = df
            self.df_all = pd.concat(self.datasets.values(), ignore_index=True)
            # Populate dataset combo box
            self.dataset_combo.clear()
            self.dataset_combo.addItem('All Datasets')
            self.dataset_combo.addItems(list(self.datasets.keys()))
            self.dataset_combo.setEnabled(True)
            self.dataset_combo.setCurrentIndex(0)
            self.on_dataset_change()
            self.update_trends_tab()

    def add_more_csvs(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, 'Add More CSV(s)', '', 'CSV Files (*.csv)')
        if file_paths:
            for fp in file_paths:
                df = pd.read_csv(fp)
                self.datasets[fp] = df
            self.df_all = pd.concat(self.datasets.values(), ignore_index=True)
            # Update dataset combo box
            self.dataset_combo.clear()
            self.dataset_combo.addItem('All Datasets')
            self.dataset_combo.addItems(list(self.datasets.keys()))
            self.dataset_combo.setEnabled(True)
            self.dataset_combo.setCurrentIndex(0)
            self.on_dataset_change()
            self.update_trends_tab()

    def on_dataset_change(self):
        if not self.dataset_combo.isEnabled():
            return
        selected = self.dataset_combo.currentText()
        self.current_dataset = selected
        if selected == 'All Datasets':
            self.df_all = pd.concat(self.datasets.values(), ignore_index=True)
        else:
            self.df_all = self.datasets[selected].copy()
        # Repopulate department combo and rerun analysis
        if 'department_name' in self.df_all.columns:
            dept_counts = Counter(self.df_all['department_name'].dropna())
            sorted_depts = [dept for dept, _ in dept_counts.most_common()]
            self.dept_combo.clear()
            self.dept_combo.addItem('All Departments')
            self.dept_combo.addItems(sorted_depts)
            self.dept_combo.setEnabled(True)
            self.dept_combo.setCurrentIndex(0)
            self.analyze_by_department('All Departments')
        else:
            self.dept_combo.clear()
            self.dept_combo.setEnabled(False)
            self.analyze_by_department(None)

    def on_department_change(self):
        if self.df_all is not None and self.dept_combo.isEnabled():
            dept = self.dept_combo.currentText()
            self.analyze_by_department(dept)

    def set_rating_ranges(self):
        dlg = RatingRangesDialog(self.rating_ranges, self)
        if dlg.exec_():
            self.rating_ranges = dlg.get_ranges()
            # Re-run analysis for current department
            if self.df_all is not None:
                dept = self.dept_combo.currentText() if self.dept_combo.isEnabled() else None
                self.analyze_by_department(dept)

    def analyze_by_department(self, department):
        if self.df_all is None:
            return
        if department is not None and department != 'All Departments':
            df = self.df_all[self.df_all['department_name'] == department].copy()
        else:
            df = self.df_all.copy()
        try:
            # --- CLUSTERING (KMeans) ---
            self.cluster_figure.clear()
            ax1 = self.cluster_figure.add_subplot(111)
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] == 0:
                ax1.text(0.5, 0.5, 'No numeric columns for clustering',
                         ha='center', va='center', fontsize=24)
                ax1.set_axis_off()
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(numeric_df)
                kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                df['Cluster'] = clusters
                # PCA for 2D visualization
                if numeric_df.shape[1] >= 2:
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7, s=120)
                    ax1.set_xlabel('PCA 1')
                    ax1.set_ylabel('PCA 2')
                    ax1.set_title(f'KMeans Clustering (PCA) - {department}')
                else:
                    ax1.text(0.5, 0.5, 'Not enough numeric features for PCA plot',
                             ha='center', va='center', fontsize=24)
                    ax1.set_axis_off()
            self.cluster_figure.tight_layout()
            self.cluster_canvas.draw()

            # --- ASSOCIATION RULE MINING (Apriori) ---
            arm_output = []
            self.arm_figure.clear()
            ax2 = self.arm_figure.add_subplot(111)
            cat_df = df.select_dtypes(include=['object', 'category']).copy()
            # Fill NaNs with a placeholder
            cat_df = cat_df.fillna('missing')
            # Standardize text (lowercase, strip)
            for col in cat_df.columns:
                cat_df[col] = cat_df[col].astype(str).str.lower().str.strip()
            # Bin numeric columns and add as categorical
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                try:
                    cat_df[f'{col}_binned'] = pd.qcut(df[col], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                except Exception:
                    pass  # skip columns that can't be binned
            # Filter out rare categories (appearing in <2% of rows)
            min_count = max(2, int(0.02 * len(cat_df)))
            for col in cat_df.columns:
                value_counts = cat_df[col].value_counts()
                rare_values = value_counts[value_counts < min_count].index
                cat_df[col] = cat_df[col].replace(rare_values, 'other')
            # One-hot encode
            onehot_df = pd.get_dummies(cat_df)
            frequent_itemsets = apriori(onehot_df, min_support=MIN_SUPPORT, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
            # Fill ARM table (now in ARM Results tab)
            self.arm_table.setRowCount(0)
            if not rules.empty:
                for _, row in rules.iterrows():
                    r = self.arm_table.rowCount()
                    self.arm_table.insertRow(r)
                    self.arm_table.setItem(r, 0, QTableWidgetItem(', '.join(map(str, row['antecedents']))))
                    self.arm_table.setItem(r, 1, QTableWidgetItem(', '.join(map(str, row['consequents']))))
                    self.arm_table.setItem(r, 2, QTableWidgetItem(f"{row['support']:.3f}"))
                    self.arm_table.setItem(r, 3, QTableWidgetItem(f"{row['confidence']:.3f}"))
                    self.arm_table.setItem(r, 4, QTableWidgetItem(f"{row['lift']:.3f}"))
                self.arm_table.resizeColumnsToContents()
                self.arm_table.resizeRowsToContents()
                # Association rule network plot
                G = nx.DiGraph()
                for _, row in rules.iterrows():
                    for ant in row['antecedents']:
                        for cons in row['consequents']:
                            G.add_edge(ant, cons, lift=row['lift'], confidence=row['confidence'])
                pos = nx.spring_layout(G, k=0.7, seed=42)
                edge_lifts = [G[u][v]['lift'] for u, v in G.edges()]
                edge_widths = [4 + 4 * (l - min(edge_lifts)) / (max(edge_lifts) - min(edge_lifts) + 1e-6) if len(edge_lifts) > 1 else 4 for l in edge_lifts]
                nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=1800, node_color="#7fa7ff")
                nx.draw_networkx_labels(G, pos, ax=ax2, font_size=22)
                nx.draw_networkx_edges(G, pos, ax=ax2, arrows=True, arrowstyle='-|>', width=edge_widths, edge_color="#b0b0ff")
                ax2.set_title(f'Association Rule Network (edges: rules, width: lift) - {department}', fontsize=28)
                ax2.axis('off')
            else:
                self.arm_table.setRowCount(1)
                self.arm_table.setSpan(0, 0, 1, 5)
                self.arm_table.setItem(0, 0, QTableWidgetItem('No association rules found.'))
                # Draw empty network
                ax2.set_title('Association Rule Network', fontsize=28)
                ax2.text(0.5, 0.5, 'No association rules found', ha='center', va='center', fontsize=28)
                ax2.set_axis_off()
            self.arm_figure.tight_layout()
            self.arm_canvas.draw()

            # --- DESCRIPTIVE ANALYSIS ---
            # Fill table for feature ratings
            self.desc_table.setRowCount(0)
            avg_scores = numeric_df.mean() if not numeric_df.empty else pd.Series(dtype=float)
            for feature, mean in avg_scores.items():
                rating = None
                for label, (low, high) in self.rating_ranges.items():
                    if (low is None or mean >= low) and (high is None or mean <= high):
                        rating = label
                        break
                row = self.desc_table.rowCount()
                self.desc_table.insertRow(row)
                self.desc_table.setItem(row, 0, QTableWidgetItem(str(feature)))
                self.desc_table.setItem(row, 1, QTableWidgetItem(f"{mean:.2f}"))
                self.desc_table.setItem(row, 2, QTableWidgetItem(str(rating)))
            self.desc_table.resizeColumnsToContents()
            self.desc_table.resizeRowsToContents()
            # Categorical summary as text
            desc_output = []
            desc_output.append(f"Descriptive Analysis for: {department}\n\n")
            cat_df = df.select_dtypes(include=['object', 'category'])
            if not cat_df.empty:
                desc_output.append("Categorical Columns Value Counts:\n")
                for col in cat_df.columns:
                    desc_output.append(f"{col} value counts:\n{cat_df[col].value_counts().to_string()}\n")
            self.desc_text.setPlainText('\n'.join(desc_output))

            # --- HISTOGRAM TAB ---
            self.hist_figure.clear()
            max_hist = 6
            if not numeric_df.empty:
                ncols = 2
                nrows = 3
                show_cols = numeric_df.columns[:max_hist]
                axes = self.hist_figure.subplots(nrows, ncols, squeeze=False)
                axes = axes.flatten()
                for i, col in enumerate(show_cols):
                    ax = axes[i]
                    ax.hist(numeric_df[col].dropna(), bins=20, color='#3366cc', alpha=0.8)
                    ax.set_title(col, fontsize=22)
                    ax.set_xlabel('Value', fontsize=18)
                    ax.set_ylabel('Frequency', fontsize=18)
                # Hide unused subplots
                for j in range(i+1, len(axes)):
                    self.hist_figure.delaxes(axes[j])
                if len(numeric_df.columns) > max_hist:
                    self.hist_figure.suptitle(f"Showing first {max_hist} of {len(numeric_df.columns)} numeric columns", fontsize=24, color='red')
            else:
                self.hist_figure.text(0.5, 0.5, 'No numeric columns to plot.', ha='center', va='center', fontsize=28)
            self.hist_canvas.draw()

            # --- RECOMMENDATIONS TAB ---
            self.update_recommendations_tab()

        except Exception as e:
            self.cluster_figure.clear()
            self.cluster_canvas.draw()
            # Show error in ARM table
            self.arm_table.setRowCount(1)
            self.arm_table.setSpan(0, 0, 1, 5)
            self.arm_table.setItem(0, 0, QTableWidgetItem(f"Error: {e}"))
            self.arm_figure.clear()
            self.arm_canvas.draw()
            self.desc_text.setPlainText(f"Error: {e}")
            self.hist_figure.clear()
            self.hist_canvas.draw()
            # Optionally, clear the ARM plot and show error text on the plot
            ax2 = self.arm_figure.add_subplot(111)
            ax2.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', fontsize=24, color='red')
            ax2.set_axis_off()
            self.arm_figure.tight_layout()
            self.arm_canvas.draw()
            self.recommend_text.setPlainText(f"Error: {e}")

    def update_recommendations_tab(self):
        # Gather feature means, ratings, and department
        features = []
        for row in range(self.desc_table.rowCount()):
            feature = self.desc_table.item(row, 0).text()
            mean = self.desc_table.item(row, 1).text()
            rating = self.desc_table.item(row, 2).text()
            features.append(f"{feature}: mean={mean}, rating={rating}")
        department = self.dept_combo.currentText() if self.dept_combo.isEnabled() else 'All Departments'
        prompt = (
            f"Department: {department}\n"
            f"Feature summary (mean and rating):\n" + '\n'.join(features) +
            "\n\nBased on these, give recommendations for improvement, strengths to maintain, and any actionable insights."
        )
        ai_response = self.call_groq_ai(prompt)
        self.recommend_text.setPlainText(ai_response)

    def call_groq_ai(self, prompt):
        # --- REAL GROQ API CALL ---
        try:
            from groq import Groq
        except ImportError:
            return "groq package not installed. Please install with 'pip install groq'."
        client = Groq(api_key="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")
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )
            response = ""
            for chunk in completion:
                response += chunk.choices[0].delta.content or ""
            return response.strip()
        except Exception as e:
            return f"Groq API error: {e}"

    def update_trends_tab(self):
        if not self.datasets:
            self.trends_figure.clear()
            self.trends_canvas.draw()
            self.trends_text.clear()
            return
        # Extract years and sort
        year_map = {}
        for fname in self.datasets:
            match = re.search(r'(\d{4})', fname)
            if match:
                year = int(match.group(1))
            else:
                year = 0  # fallback if no year found
            year_map[fname] = year
        sorted_files = sorted(self.datasets.keys(), key=lambda x: year_map[x])
        years = [year_map[f] for f in sorted_files]
        # Remove .5's: ensure years are int
        years = [int(y) for y in years]
        # Aggregate means
        feature_means = {}
        for fname in sorted_files:
            df = self.datasets[fname]
            numeric_df = df.select_dtypes(include=['number'])
            for col in numeric_df.columns:
                if col not in feature_means:
                    feature_means[col] = []
                feature_means[col].append(numeric_df[col].mean())
        # Plot
        self.trends_figure.clear()
        ax = self.trends_figure.add_subplot(111)
        colors = [c for c in mpl.colormaps['tab10'].colors]
        for i, (feature, means) in enumerate(feature_means.items()):
            ax.plot(years, means, marker='o', label=feature, color=colors[i % len(colors)])
            # Highlight baseline (oldest year)
            ax.scatter(years[0], means[0], s=200, color=colors[i % len(colors)], edgecolor='black', zorder=5)
        ax.set_xlabel('Year')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Means Across Years (Baseline: Oldest Year)')
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        self.trends_figure.tight_layout()
        self.trends_canvas.draw()
        # --- Groq AI Trend Analysis ---
        # Prepare summary for Groq
        summary_lines = []
        for i, year in enumerate(years):
            summary_lines.append(f"Year {year}:")
            for feature, means in feature_means.items():
                summary_lines.append(f"  {feature}: {means[i]:.2f}")
        trend_prompt = (
            "Analyze the following trends in feature means across years. "
            "Identify which features are improving, declining, or stable, and provide actionable insights.\n" +
            '\n'.join(summary_lines)
        )
        ai_trend_analysis = self.call_groq_ai(trend_prompt)
        # Show analysis below the plot
        self.trends_text.setPlainText(ai_trend_analysis)

    def save_project(self):
        import pickle
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Project', '', 'Project Files (*.pkl)')
        if fname:
            # Save datasets dict as pickle
            with open(fname, 'wb') as f:
                pickle.dump(self.datasets, f)

    def load_project(self):
        import pickle
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Project', '', 'Project Files (*.pkl)')
        if fname:
            with open(fname, 'rb') as f:
                self.datasets = pickle.load(f)
            if self.datasets:
                self.df_all = pd.concat(self.datasets.values(), ignore_index=True)
                # Update dataset combo box
                self.dataset_combo.clear()
                self.dataset_combo.addItem('All Datasets')
                self.dataset_combo.addItems(list(self.datasets.keys()))
                self.dataset_combo.setEnabled(True)
                self.dataset_combo.setCurrentIndex(0)
                self.on_dataset_change()
                self.update_trends_tab()

    def export_to_pdf(self):
        from PyQt5.QtWidgets import QFileDialog
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.lib import utils
        import tempfile
        import os
        fname, _ = QFileDialog.getSaveFileName(self, 'Export to PDF', '', 'PDF Files (*.pdf)')
        if not fname:
            return
        c = canvas.Canvas(fname, pagesize=letter)
        width, height = letter
        y = height - 40
        c.setFont('Helvetica-Bold', 24)
        c.drawString(40, y, 'Clustering & Association Rule Mining Report')
        y -= 40
        c.setFont('Helvetica', 16)
        c.drawString(40, y, f"Department: {self.dept_combo.currentText()}")
        y -= 30
        c.drawString(40, y, f"Dataset: {self.dataset_combo.currentText()}")
        y -= 40
        # Helper to add images
        def add_fig(fig, title):
            nonlocal y
            c.setFont('Helvetica-Bold', 18)
            c.drawString(40, y, title)
            y -= 24
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                fig.savefig(tmpfile.name, bbox_inches='tight')
                img = utils.ImageReader(tmpfile.name)
                iw, ih = img.getSize()
                aspect = ih / iw
                maxw = width - 80
                maxh = 320
                iw, ih = min(iw, maxw), min(int(maxw * aspect), maxh)
                c.drawImage(tmpfile.name, 40, y - ih, width=iw, height=ih)
                y -= ih + 24
                os.unlink(tmpfile.name)
        # Helper to add text
        def add_text(text, title):
            nonlocal y
            c.setFont('Helvetica-Bold', 18)
            c.drawString(40, y, title)
            y -= 24
            c.setFont('Helvetica', 12)
            for line in text.split('\n'):
                c.drawString(40, y, line[:120])
                y -= 16
                if y < 80:
                    c.showPage()
                    y = height - 40
                    c.setFont('Helvetica', 12)
            y -= 16
        # Clustering Tab
        add_fig(self.cluster_figure, 'Clustering (PCA)')
        # Association Rules Tab
        add_fig(self.arm_figure, 'Association Rule Network')
        add_text(self.arm_text.toPlainText(), 'Association Rules Summary')
        # Descriptive Analysis Tab
        # Table as text
        desc_table_text = 'Feature\tMean\tRating\n'
        for row in range(self.desc_table.rowCount()):
            desc_table_text += '\t'.join(self.desc_table.item(row, col).text() for col in range(self.desc_table.columnCount())) + '\n'
        add_text(desc_table_text, 'Descriptive Analysis Table')
        add_text(self.desc_text.toPlainText(), 'Descriptive Analysis (Categorical)')
        # Histograms Tab
        add_fig(self.hist_figure, 'Histograms')
        # Recommendations Tab
        add_text(self.recommend_text.toPlainText(), 'Recommendations')
        # Trends Tab
        add_fig(self.trends_figure, 'Trends')
        add_text(self.trends_text.toPlainText(), 'Trends Analysis (Groq AI)')
        c.save()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnalysisApp()
    window.show()
    sys.exit(app.exec_())
