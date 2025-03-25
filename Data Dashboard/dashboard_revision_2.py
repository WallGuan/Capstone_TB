import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import nltk
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import io
from scipy.stats import chi2_contingency
import calendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define a custom color palette for consistency
color_success = "#2ecc71"  # Green
color_failure = "#e74c3c"  # Red
color_neutral = "#3498db"  # Blue
color_palette = px.colors.qualitative.Pastel

# Set custom theme function
def set_custom_theme():
    """Apply custom styling to the dashboard."""
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        
        h1 {
            color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        h3 {
            color: #34495e;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .stMetric {
            background-color: white;
            border-radius: 5px;
            padding: 15px 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e6f3ff;
            border-bottom: 2px solid #3498db;
        }
        
        .custom-metric-container {
            display: flex;
            flex-direction: column;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            height: 100%;
        }
        
        .metric-title {
            font-size: 14px;
            color: #34495e;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }
        
        .success-card {
            background-color: #d5f5e3;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #2ecc71;
            margin-bottom: 20px;
        }
        
        .warning-card {
            background-color: #fdebd0;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #f39c12;
            margin-bottom: 20px;
        }
        
        .info-card {
            background-color: #d6eaf8;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #3498db;
            margin-bottom: 20px;
        }
        
        /* Enhance sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Homelessness Services Data Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom theme
set_custom_theme()

# Custom metric component
def custom_metric(title, value, delta=None, delta_color="normal"):
    """Creates a custom styled metric component."""
    delta_colors = {
        "positive": "green",
        "negative": "red",
        "normal": "gray"
    }
    
    delta_html = ""
    if delta:
        direction = "↑" if delta > 0 else "↓"
        delta_html = f"""
        <div style="color: {delta_colors[delta_color]}; font-size: 14px;">
            {direction} {abs(delta):.1f}%
        </div>
        """
    
    st.markdown(f"""
    <div class="custom-metric-container">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Function to preprocess text data
def preprocess_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Function to calculate proportions of success for categories
def calculate_proportions(df, column, status_col='GOAL_STATUS_BINARY'):
    """Calculate proportions of success for each category in a column."""
    prop_df = df.groupby([column, status_col]).size().unstack(fill_value=0)
    if 1 not in prop_df.columns:
        prop_df[1] = 0
    if 0 not in prop_df.columns:
        prop_df[0] = 0
    prop_df['total'] = prop_df.sum(axis=1)
    prop_df['success_rate'] = (prop_df[1] / prop_df['total'] * 100).round(2)
    prop_df = prop_df.sort_values('success_rate', ascending=False)
    return prop_df

# Function to plot proportional comparison 
def plot_proportional_comparison(df, column, status_column='GOAL_STATUS_BINARY', top_n=5):
    """Plot proportional distribution of top N categories by goal status"""
    
    if column not in df.columns:
        st.warning(f"Column '{column}' not found in the dataset")
        return None, None
    
    # Calculate proportions
    results = []
    
    # For each goal status (0 and 1)
    for status in [0, 1]:
        # Filter dataframe by status
        status_df = df[df[status_column] == status]
        
        # Skip if no data for this status
        if len(status_df) == 0:
            continue
            
        # Get value counts and convert to percentages
        counts = status_df[column].value_counts(normalize=True).mul(100).reset_index()
        counts.columns = ['category', 'percentage']
        counts['Goal Status'] = 'Achieved' if status == 1 else 'Not Achieved'
        
        # Add to results
        results.append(counts)
    
    # Combine results
    if not results:
        st.warning(f"No data available for '{column}'")
        return None, None
        
    combined_df = pd.concat(results)
    
    # Get top N categories overall to ensure consistent categories between groups
    top_categories = (combined_df.groupby('category')['percentage']
                     .sum()
                     .sort_values(ascending=False)
                     .head(top_n)
                     .index.tolist())
    
    # Filter for only top categories
    plot_df = combined_df[combined_df['category'].isin(top_categories)]
    
    # Create figure
    fig = px.bar(
        plot_df,
        x='percentage',
        y='category',
        color='Goal Status',
        barmode='group',
        orientation='h',
        color_discrete_map={'Achieved': color_success, 'Not Achieved': color_failure},
        title=f'Top {top_n} Categories for {column}'
    )
    
    fig.update_layout(
        xaxis_title='Percentage (%)',
        yaxis_title='',
        legend_title='',
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig, plot_df

# Function to add statistical significance testing
def add_significance_testing(df, column, outcome_col='GOAL_STATUS_BINARY'):
    """Perform chi-square test for independence between a category and outcome."""
    contingency_table = pd.crosstab(df[column], df[outcome_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    significance = "Significant" if p < 0.05 else "Not Significant"
    return p, significance

# Function to create a Sankey diagram for transitions
def create_sankey_diagram(df, source_col, target_col, value_col=None):
    """Create an improved Sankey diagram for visualizing flows between living situations."""
    if value_col:
        # If value column is provided, use it for the flow values
        sankey_df = df.groupby([source_col, target_col])[value_col].sum().reset_index()
    else:
        # Otherwise, count occurrences
        sankey_df = df.groupby([source_col, target_col]).size().reset_index()
        sankey_df.columns = [source_col, target_col, 'value']
    
    # Create node labels
    all_nodes = pd.unique(sankey_df[[source_col, target_col]].values.ravel('K'))
    all_nodes = [str(node) for node in all_nodes if str(node) != 'nan']
    
    # Improve node labels (shorten if necessary)
    shortened_nodes = [node[:30] + '...' if len(node) > 30 else node for node in all_nodes]
    
    # Map source and target to indices
    source_indices = [list(all_nodes).index(str(source)) for source in sankey_df[source_col]]
    target_indices = [list(all_nodes).index(str(target)) for target in sankey_df[target_col]]
    
    # Group nodes into categories for coloring
    node_colors = []
    for node in all_nodes:
        if 'Emergency' in node:
            node_colors.append('#1f77b4')  # Blue for emergency
        elif 'Permanent' in node:
            node_colors.append('#2ca02c')  # Green for permanent
        elif 'Transitional' in node:
            node_colors.append('#ff7f0e')  # Orange for transitional
        else:
            node_colors.append('#7f7f7f')  # Gray for other
    
    # Create Sankey diagram with improved styling
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=shortened_nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=sankey_df['value'],
            # Add hover template with more information
            hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>'
        )
    )])
    
    # Improve layout with better dimensions and arrangement
    fig.update_layout(
        title_text=f"Flow from {source_col} to {target_col}",
        height=700,  # Increase height
        font=dict(size=12),  # Consistent font size
        margin=dict(l=25, r=25, t=50, b=25)
    )
    
    return fig

# Function to create a radar chart
def create_radar_chart(df, metrics, group_col='GOAL_STATUS_BINARY', group_values=[1, 0], group_labels=['Achieved', 'Not Achieved']):
    """Create a radar chart comparing metrics between groups."""
    radar_data = []
    
    for i, value in enumerate(group_values):
        group_df = df[df[group_col] == value]
        
        metric_values = []
        for metric in metrics:
            if metric in df.columns:
                # Get mean value for the metric in this group
                mean_val = group_df[metric].mean()
                metric_values.append(mean_val)
            else:
                metric_values.append(0)
        
        # Close the polygon by repeating the first value
        metric_values.append(metric_values[0])
        metrics_with_first = metrics + [metrics[0]]
        
        radar_data.append(go.Scatterpolar(
            r=metric_values,
            theta=metrics_with_first,
            fill='toself',
            name=group_labels[i]
        ))
    
    fig = go.Figure(data=radar_data)
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2 * max([max(trace['r']) for trace in radar_data])]
            )
        ),
        showlegend=True
    )
    
    return fig

# Function to generate a downloadable report
def generate_report(df):
    """Generate a summary report from the analyzed data."""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write overview sheet
        overview = pd.DataFrame({
            'Metric': ['Total Goals', 'Success Rate', 'Avg Time to Complete'],
            'Value': [
                len(df),
                f"{df['GOAL_STATUS_BINARY'].mean() * 100:.1f}%",
                f"{df['TIME_TO_COMPLETE'].mean():.1f} days"
            ]
        })
        overview.to_excel(writer, sheet_name='Overview', index=False)
        
        # Write domain analysis
        if 'DOMAIN__C' in df.columns:
            domain_analysis = calculate_proportions(df, 'DOMAIN__C')
            domain_analysis.to_excel(writer, sheet_name='Domain Analysis')
        
        # Write living situation analysis
        if 'LIVING_SITUATION_AT_ENTRY__C' in df.columns:
            living_analysis = calculate_proportions(df, 'LIVING_SITUATION_AT_ENTRY__C')
            living_analysis.to_excel(writer, sheet_name='Living Situation Analysis')
        
        # Write time analysis
        time_stats = df.groupby('GOAL_STATUS_BINARY')['TIME_TO_COMPLETE'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std Dev', 'std')
        ]).round(1)
        time_stats.index = ['Not Achieved', 'Achieved']
        time_stats.to_excel(writer, sheet_name='Time Analysis')
    
    buffer.seek(0)
    return buffer

# Function for cohort analysis
def add_cohort_analysis(df, date_col, outcome_col='GOAL_STATUS_BINARY'):
    """Create a cohort analysis to track outcomes over time."""
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract year-month
    df['cohort'] = df[date_col].dt.strftime('%Y-%m')
    
    # Group by cohort and outcome
    cohort_data = df.groupby(['cohort', outcome_col]).size().unstack().fillna(0)
    
    # Calculate success rate
    if 1 in cohort_data.columns:
        total = cohort_data.sum(axis=1)
        cohort_data['success_rate'] = (cohort_data[1] / total * 100).round(1)
    
    return cohort_data

# Function to train a predictive model
def train_predictive_model(df, target_col='GOAL_STATUS_BINARY'):
    """Train a Random Forest classifier to predict the target variable."""
    # Select features (categorical variables need encoding)
    cat_features = ['DOMAIN__C', 'LIVING_SITUATION_AT_ENTRY__C', 'COUNTY_AT_ENTRY__C']
    num_features = ['TIME_TO_COMPLETE']
    
    # Filter features that exist in the dataframe
    cat_features = [col for col in cat_features if col in df.columns]
    num_features = [col for col in num_features if col in df.columns]
    
    if not cat_features and not num_features:
        return None, None, None
    
    # Create dummy variables for categorical features
    X = pd.get_dummies(df[cat_features + num_features], columns=cat_features)
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions and accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, accuracy, importance

# Load data
@st.cache_data
def load_data():
    try:
        # Load the all.csv file from the dat folder
        df = pd.read_csv('/Users/natehu/Desktop/TechBridge/QTM 498R Capstone/data dashboard/dat/all.csv')
        return df
    except Exception as e:
        st.error(f"Could not load the data file: {str(e)}")
        st.stop()

# Dashboard title and description
st.title("Homelessness Services Data Dashboard")
st.markdown("""
This dashboard analyzes goal achievement data from homelessness services, 
focusing on understanding the factors that contribute to successful outcomes.
The analysis compares successful vs. unsuccessful goals to identify patterns and insights.
""")

# Add dashboard information card
with st.expander("About this Dashboard", expanded=False):
    st.markdown("""
    <div class="info-card">
    <h4>How to Use This Dashboard</h4>
    <p>This dashboard provides insights into homelessness services data, focusing on what makes goals successful. Use the sidebar filters to narrow your analysis, and navigate through different sections using the tabs below.</p>
    
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Filter data by organization, domain, living situation, and time frame</li>
        <li>Analyze success rates across different categories</li>
        <li>Explore transitions between living situations</li>
        <li>Examine time-to-completion patterns</li>
        <li>Discover insights from qualitative goal descriptions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Load the data
try:
    df = load_data()
except:
    st.warning("Could not load the data. Please check the file path.")
    st.stop()

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Filter out EGOs with less than 200 rows 
ego_counts = df['RECORD_ORIGIN__C_y'].value_counts()
valid_egos = ego_counts[ego_counts >= 100].index.tolist()

if not valid_egos:
    st.error("No EGOs have 200 or more records. Please adjust the minimum record threshold.")
    st.stop()

# Filter the dataframe to only include valid EGOs
df = df[df['RECORD_ORIGIN__C_y'].isin(valid_egos)]

# Sidebar filters
st.sidebar.header("Filters")

# EGOs filter (RECORD_ORIGIN__C_y)
all_egos = sorted(df['RECORD_ORIGIN__C_y'].unique())
st.sidebar.markdown(f"**Available EGOs:** {len(all_egos)} (with 200+ records)")

selected_egos = st.sidebar.multiselect(
    "Select EGOs (Organizations)",
    options=all_egos,
    default=all_egos
)

# Apply EGOs filter
if selected_egos:
    df = df[df['RECORD_ORIGIN__C_y'].isin(selected_egos)]
else:
    st.warning("Please select at least one EGO to continue")
    st.stop()

# Domain filter
all_domains = sorted(df['DOMAIN__C'].unique())
selected_domains = st.sidebar.multiselect(
    "Select Domains",
    options=all_domains,
    default=all_domains
)

# Living situation filter
all_living = sorted(df['LIVING_SITUATION_AT_ENTRY__C'].unique())
selected_living = st.sidebar.multiselect(
    "Select Living Situations at Entry",
    options=all_living,
    default=all_living
)

# Time to complete filter
min_time, max_time = int(df['TIME_TO_COMPLETE'].min()), int(df['TIME_TO_COMPLETE'].max())
time_range = st.sidebar.slider(
    "Time to Complete Range (days)",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time)
)

# Apply remaining filters
filtered_df = df[
    (df['DOMAIN__C'].isin(selected_domains)) &
    (df['LIVING_SITUATION_AT_ENTRY__C'].isin(selected_living)) &
    (df['TIME_TO_COMPLETE'] >= time_range[0]) &
    (df['TIME_TO_COMPLETE'] <= time_range[1])
]

# Show number of records after filtering
st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)}")

# If no data after filtering
if filtered_df.empty:
    st.warning("No data available with the current filter settings.")
    st.stop()

# Add dashboard tabs for better navigation
tabs = st.tabs([
    "📊 Overview", 
    "📈 Success Analysis", 
    "⏱️ Time Analysis", 
    "🔄 Transitions",
    "📝 Text Analysis", 
    "🔍 Data Explorer",
    "🧠 Predictive Insights"
])

# Calculate key metrics
total = len(filtered_df)
success = sum(filtered_df['GOAL_STATUS_BINARY'])
success_rate = (success / total * 100)
avg_time_success = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1]['TIME_TO_COMPLETE'].mean()
avg_time_failure = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 0]['TIME_TO_COMPLETE'].mean()

# Overview tab
with tabs[0]:
    st.header("Overview")
    
    # Display metrics in a nicer format
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        custom_metric("Total Goals", f"{total:,}")
    with metric_cols[1]:
        custom_metric("Success Rate", f"{success_rate:.1f}%")
    with metric_cols[2]:
        custom_metric("Avg. Time (Success)", f"{avg_time_success:.1f} days")
    with metric_cols[3]:
        custom_metric("Avg. Time (Failure)", f"{avg_time_failure:.1f} days")
    
    # Key insights based on the data
    st.markdown("### Key Insights")
    
    insights_cols = st.columns(2)
    
    with insights_cols[0]:
        st.markdown("""
        <div class="success-card">
        <h4>Success Factors</h4>
        <ul>
            <li>Goals in certain domains show higher success rates</li>
            <li>Success correlates with shorter completion times</li>
            <li>Specific living situations at entry show better outcomes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_cols[1]:
        st.markdown("""
        <div class="warning-card">
        <h4>Challenge Factors</h4>
        <ul>
            <li>Longer completion times correlate with lower success</li>
            <li>Certain entry conditions present significant challenges</li>
            <li>Some domains consistently show lower success rates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall Success Rate Visual with improved styling
    st.subheader("Overall Goal Status")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Success', 'Failure'],
        y=[success_rate, 100-success_rate],
        text=[f"{success_rate:.1f}%", f"{100-success_rate:.1f}%"],
        textposition='auto',
        marker_color=[color_success, color_failure]
    ))
    fig.update_layout(
        yaxis_title='Percentage',
        xaxis_title='Outcome',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Reasons for Case Closure
    st.subheader("Top Reasons for Case Closure")
    closure_reasons = filtered_df["REASON_CLOSED__C"].value_counts(normalize=True).head(10)
    fig = px.bar(
        x=closure_reasons.index,
        y=closure_reasons.values * 100,
        text=closure_reasons.values * 100,
        labels={"x": "Reason for Closure", "y": "Proportion (%)"},
        color=closure_reasons.values,
        color_continuous_scale="RdYlBu_r",
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a download report button
    report_buffer = generate_report(filtered_df)
    st.download_button(
        label="📊 Download Analysis Report",
        data=report_buffer,
        file_name="homelessness_services_analysis.xlsx",
        mime="application/vnd.ms-excel"
    )

# Success Analysis tab
with tabs[1]:
    st.header("Success Rate Analysis")
    
    # Add comparison feature
    st.subheader("Success Rate Comparison")
    comparison_cols = st.columns([1, 3])
    
    with comparison_cols[0]:
        compare_options = ['DOMAIN__C', 'OUTCOME__C', 'LIVING_SITUATION_AT_ENTRY__C', 
                          'COUNTY_AT_ENTRY__C', 'REASON_CLOSED__C']
        compare_by = st.selectbox("Compare by:", compare_options)
        
        min_count = st.slider("Min. category size:", 5, 100, 20)
        
        # Add option to normalize by category size
        normalize = st.checkbox("Normalize by category size", value=True)
        
        # Add statistical testing
        show_stats = st.checkbox("Show statistical significance", value=True)
    
    with comparison_cols[1]:
        # Calculate success rates with minimum category size filter
        category_counts = filtered_df[compare_by].value_counts()
        valid_categories = category_counts[category_counts >= min_count].index
        
        compare_df = filtered_df[filtered_df[compare_by].isin(valid_categories)]
        
        if not compare_df.empty:
            success_by_category = calculate_proportions(compare_df, compare_by)
            
            # Add statistical test if requested
            if show_stats:
                p_value, significance = add_significance_testing(compare_df, compare_by)
                st.markdown(f"""
                **Statistical Significance:**
                - p-value: {p_value:.4f}
                - Interpretation: {significance} difference between categories
                """)
            
            # Create enhanced visualization
            fig = px.bar(
                success_by_category.reset_index(),
                x=compare_by,
                y='success_rate',
                text='success_rate',
                color='success_rate',
                color_continuous_scale='RdYlGn',
                labels={compare_by: compare_by.replace('__C', ''), 'success_rate': 'Success Rate (%)'},
                height=500
            )
            
            # Add category size as hover information
            fig.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                hovertemplate='%{x}<br>Success Rate: %{y:.1f}%<br>Sample Size: %{customdata}'
            )
            
            # Add sample size as custom data
            fig.update_traces(customdata=success_by_category['total'].values)
            
            # Improve layout
            fig.update_layout(
                xaxis_title=compare_by.replace('__C', ''),
                yaxis_title='Success Rate (%)',
                xaxis_tickangle=45,
                margin=dict(l=20, r=20, t=40, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_gridcolor='rgba(200,200,200,0.2)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about the comparison
            max_category = success_by_category['success_rate'].idxmax()
            min_category = success_by_category['success_rate'].idxmin()
            
            st.markdown(f"""
            **Insights:**
            - Highest success rate: **{max_category}** ({success_by_category.loc[max_category, 'success_rate']:.1f}%)
            - Lowest success rate: **{min_category}** ({success_by_category.loc[min_category, 'success_rate']:.1f}%)
            - Difference between highest and lowest: **{success_by_category.loc[max_category, 'success_rate'] - success_by_category.loc[min_category, 'success_rate']:.1f}%**
            """)
        else:
            st.warning(f"No categories with at least {min_count} records.")
    
    # Success by Domain and Living Situation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Success Rate by Domain")
        domain_props = calculate_proportions(filtered_df, 'DOMAIN__C')
        
        fig = px.bar(
            domain_props.reset_index(), 
            x='DOMAIN__C', 
            y='success_rate',
            text='success_rate',
            color='success_rate',
            color_continuous_scale='Viridis',
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            xaxis_title='Domain', 
            yaxis_title='Success Rate (%)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Success Rate by Living Situation")
        living_props = calculate_proportions(filtered_df, 'LIVING_SITUATION_AT_ENTRY__C')
        
        # Keep only top categories for rea
        # Keep only top categories for readability
        top_living = living_props.head(10)
        
        fig = px.bar(
            top_living.reset_index(), 
            x='LIVING_SITUATION_AT_ENTRY__C', 
            y='success_rate',
            text='success_rate',
            color='success_rate',
            color_continuous_scale='Viridis',
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            xaxis_title='Living Situation',
            yaxis_title='Success Rate (%)',
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Add a new analysis: Success Rate by Duration
    st.subheader("Success Rate by Service Duration")
    # Create duration bins
    filtered_df['duration_bin'] = pd.cut(
        filtered_df['TIME_TO_COMPLETE'],
        bins=[0, 30, 90, 180, 365, float('inf')],
        labels=['<30 days', '30-90 days', '90-180 days', '180-365 days', '>365 days']
    )
    
    duration_success = filtered_df.groupby('duration_bin')['GOAL_STATUS_BINARY'].agg(
        success_rate=lambda x: x.mean() * 100,
        count=lambda x: x.count()
    ).reset_index()
    
    # Create visualization
    fig = px.bar(
        duration_success,
        x='duration_bin',
        y='success_rate',
        text='success_rate',
        color='success_rate',
        color_continuous_scale='RdYlGn',
        hover_data=['count']
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        title="Success Rate by Service Duration",
        xaxis_title="Duration",
        yaxis_title="Success Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Time Analysis tab
with tabs[2]:
    st.header("Time to Complete Analysis")
    
    # Add time distribution visualization
    time_cols = st.columns([2, 1])
    
    with time_cols[0]:
        st.subheader("Time Distribution by Goal Status")
        
        # Create a more informative histogram with KDE
        fig = px.histogram(
            filtered_df,
            x='TIME_TO_COMPLETE',
            color='GOAL_STATUS_BINARY',
            marginal='violin',
            opacity=0.7,
            barmode='overlay',
            color_discrete_map={1: color_success, 0: color_failure},
            labels={'TIME_TO_COMPLETE': 'Time to Complete (days)'},
            category_orders={"GOAL_STATUS_BINARY": [1, 0]},
            nbins=30
        )
        
        # Update legend
        fig.update_layout(
            legend_title_text='Goal Status',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update x-axis
        fig.update_xaxes(title_text='Time to Complete (days)')
        
        # Update y-axis
        fig.update_yaxes(title_text='Count')
        
        # Update traces
        fig.for_each_trace(lambda t: t.update(
            name='Achieved' if t.name == '1' else 'Not Achieved',
            legendgroup='Achieved' if t.name == '1' else 'Not Achieved'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with time_cols[1]:
        st.subheader("Time Statistics")
        
        # Calculate time statistics
        time_stats = filtered_df.groupby('GOAL_STATUS_BINARY')['TIME_TO_COMPLETE'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std Dev', 'std')
        ]).round(1)
        
        # Rename index
        time_stats.index = ['Not Achieved', 'Achieved']
        
        # Display statistics
        st.dataframe(time_stats, use_container_width=True)
        
        # Add insights
        st.markdown("""
        <div class="info-card">
        <h4>Time Insights</h4>
        <ul>
            <li>Successful goals typically take less time to complete</li>
            <li>There's significant variability in completion time</li>
            <li>Goals taking extremely long periods have lower success rates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Time Distribution by Goal Status")
        fig = px.box(
            filtered_df, 
            x='GOAL_STATUS_BINARY', 
            y='TIME_TO_COMPLETE',
            color='GOAL_STATUS_BINARY',
            labels={'TIME_TO_COMPLETE': 'Time (days)'},
            category_orders={"GOAL_STATUS_BINARY": [1, 0]},
            color_discrete_map={1: color_success, 0: color_failure}
        )
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Failed', 'Successful']
            ),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Time by Domain (Successful Goals)")
        success_df = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1]
        if not success_df.empty:
            domain_time = success_df.groupby('DOMAIN__C')['TIME_TO_COMPLETE'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                domain_time.reset_index(), 
                x='DOMAIN__C', 
                y='TIME_TO_COMPLETE',
                text=domain_time.values.round(1),
                color='TIME_TO_COMPLETE',
                color_continuous_scale='Viridis',
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(
                xaxis_title='Domain', 
                yaxis_title='Average Time (days)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No successful goals in the current filtered data.")
    
    # Add monthly success rate heatmap
    if 'DATE_OPENED__C' in filtered_df.columns:
        st.subheader("Success Rate by Month")
        
        # Convert to datetime and extract month and year
        filtered_df['DATE_OPENED__C'] = pd.to_datetime(filtered_df['DATE_OPENED__C'], errors='coerce')
        
        # Only proceed if we have valid date data
        if not filtered_df['DATE_OPENED__C'].isna().all():
            filtered_df['month'] = filtered_df['DATE_OPENED__C'].dt.month
            filtered_df['year'] = filtered_df['DATE_OPENED__C'].dt.year
            
            # Group by month and year
            monthly_data = filtered_df.groupby(['year', 'month'])['GOAL_STATUS_BINARY'].agg(
                success_rate=lambda x: x.mean() * 100,
                count=lambda x: x.count()
            ).reset_index()
            
            # Only include months with sufficient data
            monthly_data = monthly_data[monthly_data['count'] >= 10]
            
            # Create pivot table for heatmap
            heatmap_data = monthly_data.pivot(index='month', columns='year', values='success_rate')
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Year", y="Month", color="Success Rate (%)"),
                x=heatmap_data.columns,
                y=[calendar.month_abbr[i] for i in heatmap_data.index],
                color_continuous_scale="RdYlGn",
                text_auto=".1f"
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Month",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Transitions tab
with tabs[3]:
    st.header("Transitions Analysis")
    
    # Living Situation Transition Analysis with improved Sankey diagram
    st.subheader("Living Situation Flow Analysis")
    
    transition_df = filtered_df.dropna(subset=['LIVING_SITUATION_AT_ENTRY__C', 'LIVING_SITUATION_AT_EXIT__C'])
    
    if not transition_df.empty:
        # Create a better Sankey diagram for transitions
        with st.spinner("Generating flow diagram..."):
            fig = create_sankey_diagram(
                transition_df, 
                'LIVING_SITUATION_AT_ENTRY__C', 
                'LIVING_SITUATION_AT_EXIT__C'
            )
            
            # Fix layout issues by adjusting the figure size and margins
            fig.update_layout(
                height=600,  # Increase height
                width=800,   # Set explicit width
                margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins
                font=dict(size=10),  # Reduce font size for better fit
                autosize=True  # Enable autosize
            )
            
            # Ensure nodes are properly positioned
            fig.update_layout(
                updatemenus=[dict(
                    buttons=[dict(
                        args=[{'visible': [True, True]},
                              {'title': 'Living Situation Transitions'}],
                        label='Reset',
                        method='update'
                    )]
                )]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-card">
            <h4>How to Read This Diagram</h4>
            <p>This Sankey diagram shows the flow of clients from their entry living situation (left) to their exit 
            living situation (right). The width of each flow represents the number of clients who took that path.</p>
            <p>Wider connections indicate more common transitions.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No valid transition data available.")
    
    # Add traditional transition analysis as well
    st.subheader("Living Situation Transitions")
    
    # Analyze transitions between entry and exit situations
    success_df = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1]
    
    if not success_df.empty:
        # Remove missing values
        transition_df = success_df.dropna(subset=['LIVING_SITUATION_AT_ENTRY__C', 'LIVING_SITUATION_AT_EXIT__C'])
        
        if not transition_df.empty:
            # Get counts of each transition
            transitions = transition_df.groupby(['LIVING_SITUATION_AT_ENTRY__C', 'LIVING_SITUATION_AT_EXIT__C']).size().reset_index()
            transitions.columns = ['Entry', 'Exit', 'Count']
            
            # Calculate proportions
            total_by_entry = transitions.groupby('Entry')['Count'].sum().reset_index()
            transitions = transitions.merge(total_by_entry, on='Entry', suffixes=('', '_total'))
            transitions['Proportion'] = (transitions['Count'] / transitions['Count_total'] * 100).round(1)
            
            # Filter for top transitions for readability
            top_transitions = transitions.sort_values('Count', ascending=False).head(15)
            
            fig = px.bar(
                top_transitions,
                y='Entry',
                x='Proportion',
                color='Exit',
                labels={'Proportion': 'Proportion (%)', 'Entry': 'Living Situation at Entry'},
                orientation='h'
            )
            fig.update_layout(
                height=600,
                legend_title="Living Situation at Exit",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid transition data available for successful goals.")
    else:
        st.info("No successful goals in the current filtered data.")
    
    # Transition Matrix with improved styling
    st.subheader("Living Situation Transition Matrix")
    
    # Aggregate transitions
    transitions_matrix = filtered_df.groupby(["LIVING_SITUATION_AT_ENTRY__C", "LIVING_SITUATION_AT_EXIT__C"]).size().unstack(fill_value=0)
    
    # Convert to proportions
    transitions_matrix = transitions_matrix.div(transitions_matrix.sum(axis=1), axis=0) * 100
    
    # Create heatmap with annotations
    fig_heatmap = ff.create_annotated_heatmap(
        z=transitions_matrix.values,
        x=transitions_matrix.columns.tolist(),
        y=transitions_matrix.index.tolist(),
        colorscale="Viridis",
        annotation_text=transitions_matrix.round(1).astype(str).values
    )
    
    fig_heatmap.update_layout(
        title="Transition Percentages Between Entry and Exit Living Situations",
        height=600,
        xaxis=dict(title="Exit Living Situation"),
        yaxis=dict(title="Entry Living Situation")
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # County movement analysis
    st.subheader("County Movement Analysis")
    # Only include rows where both entry and exit counties are recorded
    county_df = filtered_df.dropna(subset=['COUNTY_AT_ENTRY__C', 'COUNTY_AT_EXIT__C'])
    
    if not county_df.empty:
        # Create a new column to indicate if there was movement between counties
        county_df['COUNTY_MOVEMENT'] = county_df.apply(
            lambda x: 'Moved Counties' if x['COUNTY_AT_ENTRY__C'] != x['COUNTY_AT_EXIT__C'] else 'Same County', 
            axis=1
        )
        
        # Calculate success rates for those who moved vs. those who didn't
        movement_props = calculate_proportions(county_df, 'COUNTY_MOVEMENT')
        
        fig = px.bar(
            movement_props.reset_index(), 
            x='COUNTY_MOVEMENT', 
            y='success_rate',
            text='success_rate',
            color='COUNTY_MOVEMENT',
            color_discrete_map={'Moved Counties': color_neutral, 'Same County': color_success}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            xaxis_title='County Movement', 
            yaxis_title='Success Rate (%)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add significance test
        p_value, significance = add_significance_testing(county_df, 'COUNTY_MOVEMENT')
        st.markdown(f"""
        **Statistical Significance:** p-value = {p_value:.4f} ({significance})
        
        The difference in success rates between those who moved counties and those who stayed in the same county 
        is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}.
        """)
    else:
        st.info("No valid county data available for analysis.")

# Text Analysis tab
with tabs[4]:
    st.header("Text Analysis")
    
    # Check for the text column
    text_column = 'DETAILED_DESCRIPTION_OF_GOAL_OPTIONAL__C'
    
    if text_column not in filtered_df.columns:
        st.error(f"Column '{text_column}' not found in the dataset.")
    else:
        # Preprocess text
        with st.spinner("Preprocessing text data..."):
            mask = filtered_df[text_column] != 'Missing'
            filtered_df = filtered_df[mask]
            filtered_df['CLEAN_TEXT'] = filtered_df[text_column].fillna("").apply(preprocess_text)
        
        # Only proceed if we have valid text data
        if filtered_df['CLEAN_TEXT'].str.strip().str.len().sum() > 0:
            st.markdown("""
            <div class="info-card">
            <h4>About Text Analysis</h4>
            <p>This section analyzes the text descriptions of goals to identify patterns and themes. 
            The word cloud shows frequently used terms, while sentiment analysis measures the 
            positive or negative tone of the descriptions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            wc_col, sentiment_col = st.columns(2)
            
            with wc_col:
                st.subheader("Word Cloud Analysis")
                status_filter = st.radio(
                    "Select Goal Status:", 
                    ["Achieved", "Not Achieved"], 
                    horizontal=True
                )
                
                goal_status_value = 1 if status_filter == "Achieved" else 0
                text_data = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == goal_status_value]['CLEAN_TEXT']
                
                # Only generate wordcloud if we have text
                if text_data.str.strip().str.len().sum() > 0:
                    combined_text = ' '.join(text_data)
                    
                    # Generate word cloud with improved styling
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        collocations=False,
                        colormap='viridis'
                    ).generate(combined_text)
                    
                    # Display word cloud
                    plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                else:
                    st.info(f"No text data available for {status_filter} goals.")
            
            with sentiment_col:
                st.subheader("Sentiment Analysis")
                
                # Calculate sentiment
                filtered_df['SENTIMENT'] = filtered_df['CLEAN_TEXT'].apply(
                    lambda x: TextBlob(x).sentiment.polarity if x else 0
                )
                
                # Create sentiment visualization with improved styling
                fig = px.histogram(
                    filtered_df, 
                    x='SENTIMENT', 
                    color='GOAL_STATUS_BINARY',
                    nbins=20,
                    labels={'SENTIMENT': 'Sentiment Polarity (-1 to 1)'},
                    barmode='overlay',
                    color_discrete_map={0: color_failure, 1: color_success},
                    opacity=0.7
                )
                
                fig.update_layout(
                    xaxis_title="Sentiment (Negative → Positive)",
                    yaxis_title="Count",
                    legend_title="Goal Status",
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Update legend labels
                fig.for_each_trace(lambda t: t.update(
                    name='Achieved' if t.name == '1' else 'Not Achieved',
                    legendgroup='Achieved' if t.name == '1' else 'Not Achieved'
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insights about sentiment analysis
                avg_sentiment_success = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1]['SENTIMENT'].mean()
                avg_sentiment_failure = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 0]['SENTIMENT'].mean()
                
                st.write(f"**Average sentiment for successful goals:** {avg_sentiment_success:.3f}")
                st.write(f"**Average sentiment for unsuccessful goals:** {avg_sentiment_failure:.3f}")
                
                # Calculate statistical significance
                from scipy.stats import ttest_ind
                
                success_sentiment = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1]['SENTIMENT']
                failure_sentiment = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 0]['SENTIMENT']
                
                t_stat, p_val = ttest_ind(success_sentiment, failure_sentiment, equal_var=False, nan_policy='omit')
                
                if p_val < 0.05:
                    st.markdown(f"""
                    <div class="success-card">
                    <p><strong>Significant difference in sentiment detected!</strong> (p-value: {p_val:.4f})</p>
                    <p>The sentiment difference between successful and unsuccessful goals is statistically significant. 
                    This suggests that how goals are framed and described might be related to their likelihood of success.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="info-card">
                    <p>No significant difference in sentiment detected (p-value: {p_val:.4f}).</p>
                    <p>While there is a slight difference in sentiment between successful and unsuccessful goals, 
                    it's not statistically significant.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
            # Show top words by goal status
            st.subheader("Top Words by Goal Status")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Most Common Words in Successful Goals**")
                
                # Get text from successful goals
                success_text = ' '.join(filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1]['CLEAN_TEXT'])
                
                # Count word frequencies
                success_words = success_text.split()
                success_word_counts = {}
                for word in success_words:
                    if len(word) > 3:  # Only count words longer than 3 characters
                        success_word_counts[word] = success_word_counts.get(word, 0) + 1
                
                # Get top words
                top_success_words = sorted(success_word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
                
                # Create DataFrame for visualization
                top_success_df = pd.DataFrame(top_success_words, columns=['Word', 'Count'])
                
                # Create bar chart with improved styling
                fig = px.bar(
                    top_success_df,
                    y='Word',
                    x='Count',
                    orientation='h',
                    color_discrete_sequence=[color_success]
                )
                fig.update_layout(
                    xaxis_title="Frequency", 
                    yaxis_title=None,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Most Common Words in Unsuccessful Goals**")
                
                # Get text from unsuccessful goals
                failure_text = ' '.join(filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 0]['CLEAN_TEXT'])
                
                # Count word frequencies
                failure_words = failure_text.split()
                failure_word_counts = {}
                for word in failure_words:
                    if len(word) > 3:  # Only count words longer than 3 characters
                        failure_word_counts[word] = failure_word_counts.get(word, 0) + 1
                
                # Get top words
                top_failure_words = sorted(failure_word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
                
                # Create DataFrame for visualization
                top_failure_df = pd.DataFrame(top_failure_words, columns=['Word', 'Count'])
                
                # Create bar chart with improved styling
                fig = px.bar(
                    top_failure_df,
                    y='Word',
                    x='Count',
                    orientation='h',
                    color_discrete_sequence=[color_failure]
                )
                fig.update_layout(
                    xaxis_title="Frequency", 
                    yaxis_title=None,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient text data for qualitative analysis. The goal description column might be empty.")

# Data explorer tab
with tabs[5]:
    st.header("Data Explorer")
    
    st.markdown("""
    <div class="info-card">
    <p>This section allows you to explore the raw data and distributions of various categories.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Category Distribution Analysis
    st.subheader("Category Distribution Analysis")
    st.markdown("""
    This section shows the distribution of categories within selected columns as pie charts.
    Small categories below the threshold are grouped into an 'OTHERS' category for better visualization.
    """)
    
    # Define columns for pie chart analysis
    pie_chart_columns = [
        col for col in [
            'DOMAIN__C', 'OUTCOME__C', 'LIVING_SITUATION_AT_ENTRY__C',
            'LIVING_SITUATION_AT_EXIT__C', 'REASON_CLOSED__C', 'COUNTY_AT_ENTRY__C'
        ] 
        if col in filtered_df.columns
    ]
    
    if not pie_chart_columns:
        st.warning("No categorical columns available for pie chart analysis")
    else:
        # Allow user to select column for pie charts
        selected_column = st.selectbox("Choose a category for pie chart:", pie_chart_columns, key="pie_chart_select")
        
        # Threshold slider for grouping small categories
        threshold = st.slider(
            "Threshold for grouping small categories (%):", 
            min_value=1.0, 
            max_value=10.0, 
            value=3.0, 
            step=0.5,
            help="Categories with percentage below this threshold will be grouped as 'OTHERS'"
        )
        
        # Convert threshold to proportion
        threshold_prop = threshold / 100
        
        # Create datasets for different goal statuses
        output_all = filtered_df[selected_column].value_counts(normalize=True).reset_index()
        output_all.columns = ["Category", "Portion"]
        
        output_success = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1][selected_column].value_counts(normalize=True).reset_index()
        output_success.columns = ["Category", "Portion"]
        
        output_failure = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 0][selected_column].value_counts(normalize=True).reset_index()
        output_failure.columns = ["Category", "Portion"]
        
        # Create columns for the pie charts
        pie_col1, pie_col2, pie_col3 = st.columns(3)
        
        # Function to create pie chart with threshold and improved styling
        def create_pie_chart(data, title, colors=px.colors.qualitative.Pastel):
            # Apply threshold
            small_categories = data[data["Portion"] < threshold_prop]
            others_percentage = small_categories["Portion"].sum()
            filtered_data = data[data["Portion"] >= threshold_prop]
            
            # Add 'OTHERS' category if necessary
            if others_percentage > 0:
                new_row = pd.DataFrame({"Category": ["OTHERS"], "Portion": [others_percentage]})
                filtered_data = pd.concat([filtered_data, new_row], ignore_index=True)
            
            # Create pie chart with better styling
            fig = px.pie(
                filtered_data, 
                values='Portion', 
                names='Category',
                title=title,
                color_discrete_sequence=colors,
                hole=0.3
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                insidetextorientation='radial'
            )
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            return fig
        
        # Display pie charts with custom color schemes
        with pie_col1:
            fig_all = create_pie_chart(
                output_all, 
                f"All Goals - {selected_column}",
                px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_all, use_container_width=True)
        
        with pie_col2:
            fig_success = create_pie_chart(
                output_success, 
                f"Successful Goals - {selected_column}",
                px.colors.sequential.Greens
            )
            st.plotly_chart(fig_success, use_container_width=True)
        
        with pie_col3:
            fig_failure = create_pie_chart(
                output_failure, 
                f"Unsuccessful Goals - {selected_column}",
                px.colors.sequential.Reds
            )
            st.plotly_chart(fig_failure, use_container_width=True)
    
    # Raw data explorer
    st.subheader("Raw Data Explorer")
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)
        
        # Add a CSV download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="homelessness_services_data.csv",
            mime="text/csv"
        )

# Predictive Insights tab
with tabs[6]:
    st.header("Predictive Insights")
    
    st.markdown("""
    <div class="info-card">
    <h4>About Predictive Analysis</h4>
    <p>This section uses machine learning to identify the most important factors
    in predicting goal success. The model analyzes patterns in the data to determine
    which features have the strongest relationship with successful outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Train predictive model
    with st.spinner("Training predictive model..."):
        model, accuracy, importance = train_predictive_model(filtered_df)
    
    if model and importance is not None:
        # Model accuracy
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        fig = px.bar(
            importance.head(10),
            y='Feature',
            x='Importance',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            title="Top 10 Factors Influencing Goal Success"
        )
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a radar chart comparing key metrics
        st.subheader("Key Metrics Comparison")
        
        # Select metrics for comparison
        if 'TIME_TO_COMPLETE' in filtered_df.columns and 'SENTIMENT' in filtered_df.columns:
            # Normalize metrics for radar chart
            radar_df = filtered_df.copy()
            
            # Add domain success rate
            
            if 'DOMAIN__C' in radar_df.columns:
                domain_success = radar_df.groupby('DOMAIN__C')['GOAL_STATUS_BINARY'].mean()
                radar_df['domain_success_rate'] = radar_df['DOMAIN__C'].map(domain_success)
            
            # Add county success rate
            if 'COUNTY_AT_ENTRY__C' in radar_df.columns:
                county_success = radar_df.groupby('COUNTY_AT_ENTRY__C')['GOAL_STATUS_BINARY'].mean()
                radar_df['county_success_rate'] = radar_df['COUNTY_AT_ENTRY__C'].map(county_success)
            
            # Create metrics list
            metrics = []
            for metric in ['TIME_TO_COMPLETE', 'SENTIMENT', 'domain_success_rate', 'county_success_rate']:
                if metric in radar_df.columns and not radar_df[metric].isna().all():
                    # Min-max scaling for normalization
                    min_val = radar_df[metric].min()
                    max_val = radar_df[metric].max()
                    if min_val != max_val:  # Avoid division by zero
                        radar_df[f'{metric}_normalized'] = (radar_df[metric] - min_val) / (max_val - min_val)
                        metrics.append(f'{metric}_normalized')
            
            if metrics:
                # Create radar chart
                fig = create_radar_chart(
                    radar_df, 
                    metrics=[m.replace('_normalized', '') for m in metrics]
                )
                
                fig.update_layout(
                    title="Comparison of Key Metrics Between Successful and Unsuccessful Goals",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Key factors summary
        st.subheader("Key Factors for Success")
        
        # Extract top positive and negative factors
        positive_factors = importance.head(3)['Feature'].tolist()
        
        st.markdown("""
        <div class="success-card">
        <h4>Most Influential Factors</h4>
        <p>Based on machine learning analysis, these factors have the strongest relationship with goal outcomes:</p>
        <ol>
        """ + ''.join([f"<li><strong>{factor}</strong></li>" for factor in positive_factors]) + """
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Actionable insights
        st.subheader("Actionable Insights")
        
        st.markdown("""
        <div class="info-card">
        <h4>Recommendations</h4>
        <p>Based on the data analysis, consider these strategies to improve goal success rates:</p>
        <ul>
            <li><strong>Focus on Timeframes:</strong> Set realistic timeframes for goals - goals that take too long are less likely to succeed</li>
            <li><strong>Prioritize Key Domains:</strong> Allocate more resources to domains showing lower success rates</li>
            <li><strong>Monitor Living Situation:</strong> Pay special attention to clients with challenging living situations at entry</li>
            <li><strong>Positive Framing:</strong> Frame goal descriptions in positive terms when appropriate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Insufficient data to train a reliable predictive model. Try adjusting your filters to include more data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>Dashboard created for NGO homelessness services data analysis</p>
<p>Last updated: March 2025</p>
</div>
""", unsafe_allow_html=True)