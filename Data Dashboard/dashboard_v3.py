import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import nltk
import matplotlib.pyplot as plt
import re
import io
from scipy.stats import chi2_contingency
import calendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")
# Define a custom color palette for consistency
color_success = "#2ecc71"  # Green
color_failure = "#e74c3c"  # Red
color_neutral = "#3498db"  # Blue
color_palette = px.colors.qualitative.Pastel

# mapping for severity levels - this will be utilized in our enhanced analysis
housing_severity_scale = {
    "Missing": 0,
    "Place not meant for habitation": 1,
    "Emergency shelter, including hotel or motel paid for with emergency shelter voucher": 2,
    "Hotel or motel paid for without emergency shelter voucher": 3,
    "Staying or living in a friend's room, apartment or house": 4,
    "Staying or living with family, temporary tenure": 5,
    "Substance abuse treatment facility or detox center": 5,
    "Residential project or halfway house with no homeless criteria": 6,
    "Staying or living in a family member's room, apartment or house": 6,
    "Other": 6,
    "Rental by client, with RRH or equivalent subsidy": 7,
    "Rental by client, with other ongoing housing subsidy": 8,
    "Rental by client, with other ongoing housing subsidy (including RRH)": 8,
    "Permanent housing (other than RRH) for formerly homeless persons": 9,
    "Rental by client, no ongoing housing subsidy": 10,
    "Staying or living with family, permanent tenure": 11,
    "Owned by client, with ongoing housing subsidy": 12,
    "Owned by client, no ongoing housing subsidy": 13
}


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
        
        .failure-card {
            background-color: #fadbd8;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #e74c3c;
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
# def preprocess_text(text):
#     """Clean and preprocess text data."""
#     if not isinstance(text, str):
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove special characters and numbers
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     words = text.split()
#     words = [word for word in words if word not in stop_words and len(word) > 2]
    
#     return ' '.join(words)

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

# Function to create a Sankey diagram for transitions - FIXED to handle missing values appropriately
def create_sankey_diagram(df, source_col, target_col, value_col=None):
    """Create an improved Sankey diagram for visualizing flows between living situations."""
    # Filter out missing values in both source and target columns
    valid_df = df.dropna(subset=[source_col, target_col])
    
    # Filter out "Missing" values if they exist
    valid_df = valid_df[(valid_df[source_col] != "Missing") & (valid_df[target_col] != "Missing")]
    
    if valid_df.empty:
        return None
    
    if value_col:
        # If value column is provided, use it for the flow values
        sankey_df = valid_df.groupby([source_col, target_col])[value_col].sum().reset_index()
    else:
        # Otherwise, count occurrences
        sankey_df = valid_df.groupby([source_col, target_col]).size().reset_index()
        sankey_df.columns = [source_col, target_col, 'value']
    
    # Create node labels
    all_nodes = pd.unique(sankey_df[[source_col, target_col]].values.ravel('K'))
    all_nodes = [str(node) for node in all_nodes if str(node) != 'nan']
    
    # Improve node labels (shorten if necessary)
    shortened_nodes = [node[:30] + '...' if len(node) > 30 else node for node in all_nodes]
    
    # Map source and target to indices
    source_indices = [list(all_nodes).index(str(source)) for source in sankey_df[source_col]]
    target_indices = [list(all_nodes).index(str(target)) for target in sankey_df[target_col]]
    
    # Group nodes into categories for coloring based on housing severity scale
    node_colors = []
    for node in all_nodes:
        # Get severity score, defaulting to middle value if not found
        severity = housing_severity_scale.get(node, 6)
        
        # Create a color scale based on severity (red for severe, green for stable)
        if severity <= 3:  # High severity (emergency)
            node_colors.append('#e74c3c')  # Red
        elif severity <= 6:  # Medium severity (temporary)
            node_colors.append('#f39c12')  # Orange
        elif severity <= 9:  # Lower severity (subsidized)
            node_colors.append('#3498db')  # Blue
        else:  # Low severity (stable)
            node_colors.append('#2ecc71')  # Green
    
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

# NEW FUNCTION: Add housing stability analysis based on severity scale
def add_housing_stability_analysis(df):
    """Add housing stability score based on the severity scale and analyze outcomes."""
    # Create a copy to avoid modifying the original dataframe
    analysis_df = df.copy()
    
    
    # Map living situations to severity scores
    analysis_df['entry_severity'] = analysis_df['LIVING_SITUATION_AT_ENTRY__C'].map(housing_severity_scale)
    analysis_df['exit_severity'] = analysis_df['LIVING_SITUATION_AT_EXIT__C'].map(housing_severity_scale)
    
    # Calculate stability improvement (higher score means more stable housing)
    analysis_df['stability_change'] = analysis_df['exit_severity'] - analysis_df['entry_severity']
    
    # Create stability categories
    stability_bins = [-float('inf'), -2, -0.001, 0.001, 2, float('inf')]
    stability_labels = ['Significant Decline', 'Slight Decline', 'No Change', 'Slight Improvement', 'Significant Improvement']
    
    analysis_df['stability_category'] = pd.cut(
        analysis_df['stability_change'],
        bins=stability_bins,
        labels=stability_labels
    )
    
    # Create initial housing stability categories based on entry severity
    stability_entry_bins = [0, 3, 6, 9, 13]
    stability_entry_labels = ['Crisis', 'Temporary', 'Transitional', 'Stable']
    
    analysis_df['entry_stability'] = pd.cut(
        analysis_df['entry_severity'],
        bins=stability_entry_bins,
        labels=stability_entry_labels
    )
    
    return analysis_df

# NEW FUNCTION: Demographics analysis
def add_demographics_analysis(df, demographic_cols):
    """Add demographics analysis by age, gender, race/ethnicity, etc."""
    demographics = {}
    
    for col in demographic_cols:
        if col in df.columns:
            # Get counts and proportions
            counts = df[col].value_counts().reset_index()
            counts.columns = ['Category', 'Count']
            counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
            
            # Get success rates by demographic category
            demo_success = calculate_proportions(df, col)
            counts = counts.merge(
                demo_success['success_rate'].reset_index(),
                left_on='Category',
                right_on=col,
                how='left'
            )
            counts = counts.drop(columns=[col])
            
            demographics[col] = counts
    
    return demographics

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
        
        # NEW: Add housing stability analysis to report
        if 'LIVING_SITUATION_AT_ENTRY__C' in df.columns and 'LIVING_SITUATION_AT_EXIT__C' in df.columns:
            stability_df = add_housing_stability_analysis(df)
            stability_analysis = calculate_proportions(stability_df, 'stability_category')
            stability_analysis.to_excel(writer, sheet_name='Housing Stability Analysis')
        
        # NEW: Add demographics analysis to report if available
        demo_cols = ['AGE_GROUP', 'GENDER', 'RACE', 'ETHNICITY']
        available_demo_cols = [col for col in demo_cols if col in df.columns]
        
        if available_demo_cols:
            demo_analysis = add_demographics_analysis(df, available_demo_cols)
            for col, data in demo_analysis.items():
                data.to_excel(writer, sheet_name=f'{col} Analysis', index=False)
    
    buffer.seek(0)
    return buffer

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

# Filter out EGOs with less than 100 rows (changed from 200 to capture more data)
ego_counts = df['RECORD_ORIGIN__C_y'].value_counts()
valid_egos = ego_counts[ego_counts >= 100].index.tolist()

if not valid_egos:
    st.error("No EGOs have 100 or more records. Please adjust the minimum record threshold.")
    st.stop()

# Filter the dataframe to only include valid EGOs
df = df[df['RECORD_ORIGIN__C_y'].isin(valid_egos)]

# Sidebar filters
st.sidebar.header("Filters")

# EGOs filter (RECORD_ORIGIN__C_y)
all_egos = sorted(df['RECORD_ORIGIN__C_y'].unique())
st.sidebar.markdown(f"**Available NGOs:** {len(all_egos)} (with 100+ records)")

selected_egos = st.sidebar.multiselect(
    "Select NGOs (Organizations)",
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

# NEW: Enhance filtered data with additional analysis
# Add housing stability metrics
stability_df = add_housing_stability_analysis(filtered_df)

# Add dashboard tabs for better navigation - REVISED with new dedicated sections
# tabs = st.tabs([
#     "📊 Overview", 
#     "📈 Success Factors", 
#     "❌ Failure Analysis",  # NEW tab for unsuccessful analysis
#     "🏠 Housing Stability",  # NEW tab for housing stability analysis
#     "👥 Demographics",      # NEW tab for demographics analysis
#     "⏱️ Time Analysis", 
#     "🔄 Transitions",
#     "🔍 Data Explorer"
# ])
tabs = st.tabs([
    "📊 Overview",              # general summary 🧮
    "🚀 Success Factors",       # upward/rocket for positive drivers
    "⚠️ Failure Analysis",      # warning sign for issues/failures
    "🏘️ Housing Stability",     # cluster of homes for community/housing
    "🧑‍🤝‍🧑 Demographics",       # diverse people icon for demographic spread
    "⏳ Time Analysis",         # hourglass for passage of time
    # "🔁 Transitions",           # looped arrow for state transitions
    "🗂️ Data Explorer"          # folders for data browsing
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
    
    # NEW: Add a summary of housing stability improvement
    st.subheader("Housing Stability Overview")
    
    # Calculate average change in housing stability
    avg_stability_change = stability_df['stability_change'].mean()
    stability_improved = (stability_df['stability_change'] > 0).mean() * 100
    
    stability_cols = st.columns(2)
    
    with stability_cols[0]:
        custom_metric(
            "Average Housing Stability Change", 
            f"+ {avg_stability_change:.2f} points",
            # delta=avg_stability_change,
            # delta_color="positive" if avg_stability_change > 0 else "negative"
        )
        
    with stability_cols[1]:
        custom_metric(
            "Clients with Improved Housing", 
            f"{stability_improved:.1f}%"
        )
    
    # Add a download report button
    report_buffer = generate_report(stability_df)
    st.download_button(
        label="📊 Download Analysis Report",
        data=report_buffer,
        file_name="homelessness_services_analysis.xlsx",
        mime="application/vnd.ms-excel"
    )

    # Success Factors tab (renamed from Success Analysis)
with tabs[1]:
    st.header("Success Factors Analysis")
    
    # Add comparison feature
    st.subheader("Success Rate by Category")
    comparison_cols = st.columns([1, 3])
    
    with comparison_cols[0]:
        compare_options = ['DOMAIN__C', 'OUTCOME__C', 'LIVING_SITUATION_AT_ENTRY__C', 
                           'REASON_CLOSED__C']
        compare_by = st.selectbox("Compare by:", compare_options, key="success_compare")
        
        min_count = st.slider("Min. category size:", 5, 500, 20, key="success_min_count")
        
        # Add option to normalize by category size
        # normalize = st.checkbox("Normalize by category size", value=True, key="success_normalize")
        
        # # Add statistical testing
        # show_stats = st.checkbox("Show statistical significance", value=True, key="success_stats")
    
    with comparison_cols[1]:
        # Calculate success rates with minimum category size filter
        category_counts = filtered_df[compare_by].value_counts()
        valid_categories = category_counts[category_counts >= min_count].index
        
        compare_df = filtered_df[filtered_df[compare_by].isin(valid_categories)]
        
        if not compare_df.empty:
            success_by_category = calculate_proportions(compare_df, compare_by)
            
            # Add statistical test if requested
            # if show_stats:
            #     p_value, significance = add_significance_testing(compare_df, compare_by)
            #     st.markdown(f"""
            #     **Statistical Significance:**
            #     - p-value: {p_value:.4f}
            #     - Interpretation: {significance} difference between categories
            #     """)
            
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
    
    
    # Key Success Factors - Modified to be more focused on housing outcomes
    st.subheader("Key Success Factors for Housing Stability")

    # First, identify clients in Housing domain with successful goals
    housing_success_df = filtered_df[(filtered_df['GOAL_STATUS_BINARY'] == 1) & 
                                (filtered_df['DOMAIN__C'] == 'Housing')]

    # Create a new dataframe for clients who had improved housing stability
    if 'LIVING_SITUATION_AT_ENTRY__C' in filtered_df.columns and 'LIVING_SITUATION_AT_EXIT__C' in filtered_df.columns:
        # Add severity scores
        housing_success_df['entry_severity'] = housing_success_df['LIVING_SITUATION_AT_ENTRY__C'].map(housing_severity_scale)
        housing_success_df['exit_severity'] = housing_success_df['LIVING_SITUATION_AT_EXIT__C'].map(housing_severity_scale)
        
        # Calculate improvement
        housing_success_df['stability_improved'] = housing_success_df['exit_severity'] > housing_success_df['entry_severity']
        
        # Get improved housing situations
        improved_df = housing_success_df[housing_success_df['stability_improved'] == True]
        
        # Get the most successful exit living situations (destinations)
        if not improved_df.empty:
            successful_destinations = improved_df['LIVING_SITUATION_AT_EXIT__C'].value_counts().reset_index()
            successful_destinations.columns = ['Living Situation', 'Count']
            successful_destinations['Percentage'] = (successful_destinations['Count'] / len(improved_df) * 100).round(1)
            
            # Only display if we have data
            if not successful_destinations.empty:
                st.markdown("#### Most Successful Housing Destinations")
                st.markdown("*For housing goals where stability level increased:*")
                
                # Display top 5 successful destinations
                for idx, row in successful_destinations.head(5).iterrows():
                    # Get the severity level for context
                    severity = housing_severity_scale.get(row['Living Situation'], 'Unknown')
                    
                    st.markdown(f"""
                    <div class="success-card">
                        <h5>{row['Living Situation']}</h5>
                        <p>Stability Level: {severity}<br>
                        Count: {row['Count']} clients ({row['Percentage']}% of improved housing)<br>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No data available on housing improvements.")
        else:
            st.info("No data available for clients with improved housing stability.")
    else:
        st.info("Living situation data not available for analysis.")
    
    # Add a section on combined factors
    st.subheader("Combined Success Factors (Domain + Living Situation at Entry)")
    
    
    if 'DOMAIN__C' in filtered_df.columns and 'LIVING_SITUATION_AT_ENTRY__C' in filtered_df.columns:
        
        filtered_df_1 = filtered_df.copy()  # Create a copy to avoid modifying the original dataframe
        filtered_df_1 = filtered_df_1[filtered_df_1['DOMAIN__C'] == 'Housing']  # Filter for housing domain only
        # Create a combined column
        filtered_df_1['domain_living'] = filtered_df_1['DOMAIN__C'] + ' + ' + filtered_df_1['LIVING_SITUATION_AT_ENTRY__C']
        
        # Add slider for minimum successful cases
        min_success_count = st.slider(
            "Minimum number of successful cases:", 
            min_value=0, 
            max_value=50, 
            value=5,
            help="Only show combinations with at least this many successful goals"
        )
        
        # Calculate success rates for combined factors
        combined_props = calculate_proportions(filtered_df_1, 'domain_living')
        
        # Filter for combinations with enough data and successful cases
        combined_props = combined_props[
            (combined_props['total'] >= 10) & 
            (combined_props[1] >= min_success_count)
        ].head(10)
        
        if not combined_props.empty:
            # Create a dataframe for plotting that includes success counts
            plot_df = combined_props.reset_index()
            plot_df['successful_cases'] = plot_df[1]  # Get the number of successful cases
            
            fig = px.bar(
                plot_df,
                x='success_rate',
                y='domain_living',
                orientation='h',
                text='success_rate',
                color='success_rate',
                color_continuous_scale='RdYlGn',
                labels={
                    'domain_living': 'Domain + Living Situation',
                    'success_rate': 'Success Rate (%)'
                },
                # Add hover data to show success cases and total
                hover_data={
                    'successful_cases': True,
                    'total': True,
                    'success_rate': ':.1f'
                }
            )
            
            # Customize hover template
            fig.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Success Rate: %{x:.1f}%<br>Successful Cases: %{customdata[0]}<br>Total Cases: %{customdata[1]}'
            )
            
            fig.update_layout(
                yaxis_title='',
                xaxis_title='Success Rate (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to analyze combined success factors with the current filters.")

# NEW: Failure Analysis tab
with tabs[2]:
    st.header("Failure Analysis")
    
    # Introduction to failure analysis
    st.markdown("""
    <div class="failure-card">
    <p>Focusing on failed client goal</p>
    
    </div>
    """, unsafe_allow_html=True)
    
    # Get failure data
    failure_df = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 0]
    
    # Calculate key failure metrics
    failure_rate = 100 - success_rate
    failure_count = len(failure_df)
    
    # Show key metrics
    metric_cols = st.columns(3)
    
    with metric_cols[0]:
        custom_metric("Failure Rate", f"{failure_rate:.1f}%")
    
    with metric_cols[1]:
        custom_metric("Failed Goals", f"{failure_count:,}")
    
    with metric_cols[2]:
        custom_metric("Avg. Time Before Failure", f"{avg_time_failure:.1f} days")
    
    # # Top reasons for failure
    # st.subheader("Top Domains with High Failure Rates")
    
    # domain_failure = calculate_proportions(filtered_df, 'DOMAIN__C')
    # domain_failure['failure_rate'] = 100 - domain_failure['success_rate']
    # domain_failure = domain_failure.sort_values('failure_rate', ascending=False)
    
    # # Plot domains with highest failure rates
    # fig = px.bar(
    #     domain_failure.head(10).reset_index(),
    #     x='DOMAIN__C',
    #     y='failure_rate',
    #     text='failure_rate',
    #     color='failure_rate',
    #     color_continuous_scale='Reds',
    #     labels={'DOMAIN__C': 'Domain', 'failure_rate': 'Failure Rate (%)'}
    # )
    # fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    # fig.update_layout(
    #     xaxis_title='Domain',
    #     yaxis_title='Failure Rate (%)',
    #     xaxis_tickangle=45,
    #     plot_bgcolor='rgba(0,0,0,0)'
    # )
    # st.plotly_chart(fig, use_container_width=True)
    # Top reasons for failure
    st.subheader("Top Domains with High Failure Rates")

    domain_failure = calculate_proportions(filtered_df, 'DOMAIN__C')
    domain_failure['failure_rate'] = 100 - domain_failure['success_rate']
    domain_failure = domain_failure.sort_values('failure_rate', ascending=False)

    # Create plot dataframe with sample size
    plot_df = domain_failure.head(10).reset_index()
    plot_df['label'] = plot_df['DOMAIN__C'] + '<br>n=' + plot_df['total'].astype(str)

    # Plot domains with highest failure rates
    fig = px.bar(
        plot_df,
        x='DOMAIN__C',
        y='failure_rate',
        text='failure_rate',
        color='failure_rate',
        color_continuous_scale='Reds',
        labels={'DOMAIN__C': 'Domain', 'failure_rate': 'Failure Rate (%)'},
        hover_data=['total']  # Add total to hover information
    )

    # Customize hover template to show sample size
    fig.update_traces(
        texttemplate='%{text:.1f}%', 
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Failure Rate: %{y:.1f}%<br>Sample Size: %{customdata[0]}'
    )

    fig.update_layout(
        xaxis_title='Domain',
        yaxis_title='Failure Rate (%)',
        xaxis_tickangle=45,
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )

    # Add sample size as text below domain names
    fig.update_xaxes(
        tickmode='array',
        tickvals=plot_df['DOMAIN__C'],
        ticktext=[f"{domain}<br><span style='font-size:10px'>n={total}</span>" for domain, total in zip(plot_df['DOMAIN__C'], plot_df['total'])]
    )

    st.plotly_chart(fig, use_container_width=True)
    # Living situations with highest failure rates
    st.subheader("Living Situations with Highest Failure Rates (Housing Domain Only)")
    
    livingf_df = filtered_df[filtered_df['DOMAIN__C'] == 'Housing']
    living_failure = calculate_proportions(livingf_df, 'LIVING_SITUATION_AT_ENTRY__C')
    
    living_failure['failure_rate'] = 100 - living_failure['success_rate']
    living_failure = living_failure.sort_values('failure_rate', ascending=False)
    
    # Add minimum count filter for more meaningful analysis
    living_failure = living_failure[living_failure['total'] >= 10]
    
    # Plot living situations with highest failure rates
    fig = px.bar(
        living_failure.head(10).reset_index(),
        x='LIVING_SITUATION_AT_ENTRY__C',
        y='failure_rate',
        text='failure_rate',
        color='failure_rate',
        color_continuous_scale='Reds',
        labels={'LIVING_SITUATION_AT_ENTRY__C': 'Living Situation', 'failure_rate': 'Failure Rate (%)'},
        hover_data=['total']  # Add total to hover information
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', hovertemplate='<b>%{x}</b><br>Failure Rate: %{y:.1f}%<br>Sample Size: %{customdata[0]}')
    fig.update_layout(
        xaxis_title='Living Situation',
        yaxis_title='Failure Rate (%)',
        xaxis_tickangle=45,
        plot_bgcolor='rgba(0,0,0,0)',
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Reason for closure distribution among failed goals
    st.subheader("Primary Reasons for Goal Failure")
    
    if 'REASON_CLOSED__C' in failure_df.columns:
        failure_reasons = failure_df['REASON_CLOSED__C'].value_counts(normalize=True).head(10)
        
        fig = px.pie(
            values=failure_reasons.values * 100,
            names=failure_reasons.index,
            title="Reasons for Goal Failure",
            color_discrete_sequence=px.colors.sequential.Reds_r,
            hole=0.4
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time dimension for failures
    st.subheader("When Do Goals Typically Fail?")
    
    # Create time bins for failure analysis
    failure_df['time_bin'] = pd.cut(
        failure_df['TIME_TO_COMPLETE'],
        bins=[0, 30, 60, 90, 180, 365, float('inf')],
        labels=['<30 days', '30-60 days', '60-90 days', '90-180 days', '6-12 months', '>12 months']
    )
    
    # Count failures by time bin
    time_failure_counts = failure_df['time_bin'].value_counts().sort_index()
    
    # Plot time distribution of failures
    fig = px.bar(
        x=time_failure_counts.index,
        y=time_failure_counts.values,
        text=time_failure_counts.values,
        labels={'x': 'Time to Failure', 'y': 'Number of Failed Goals'},
        color=time_failure_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Time Period',
        yaxis_title='Number of Failed Goals',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    # # Recommendations based on failure analysis
    # st.subheader("Recommendations to Reduce Failures")
    
    # st.markdown("""
    # Based on the failure analysis, consider the following interventions:
    
    # 1. **Early Intervention**: Provide additional support during the first 30-60 days, when many goals fail
    # 2. **Targeted Support**: Focus additional resources on domains with high failure rates
    # 3. **Risk Assessment**: Develop a risk scoring system based on domain and living situation combinations
    # 4. **Follow-up Protocol**: Implement structured follow-up for clients in high-risk categories
    # """)

# NEW: Housing Stability tab
with tabs[3]:
    st.header("Housing Stability Analysis")
    
    st.markdown("""
    <div class="info-card">
    <h4>Housing Stability Framework</h4>
    <p>This analysis uses a severity scale (1-13) to measure housing stability, where lower numbers indicate
    less stable housing situations and higher numbers represent more stable, permanent housing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the housing severity scale
    st.subheader("Housing Stability Scale")
    
    # Create a dataframe from the housing severity scale
    scale_df = pd.DataFrame({
        'Living Situation': list(housing_severity_scale.keys()),
        'Stability Score': list(housing_severity_scale.values())
    })
    scale_df = scale_df.sort_values('Stability Score')
    
    # Create a color scale based on stability
    scale_df['Color'] = scale_df['Stability Score'].apply(
        lambda x: 'rgba(231, 76, 60, 0.8)' if x <= 3 else  # Red for crisis
                 'rgba(241, 196, 15, 0.8)' if x <= 6 else  # Yellow for temporary
                 'rgba(52, 152, 219, 0.8)' if x <= 9 else  # Blue for transitional
                 'rgba(46, 204, 113, 0.8)'                 # Green for stable
    )
    
    fig = px.bar(
        scale_df,
        x='Stability Score',
        y='Living Situation',
        orientation='h',
        color='Stability Score',
        color_continuous_scale='RdYlGn',
        text='Stability Score'
    )
    fig.update_traces(textposition='inside')
    fig.update_layout(
        height=600,
        yaxis_title='',
        xaxis_title='Housing Stability Score (Higher = More Stable)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Housing stability at entry vs. success rate
    st.subheader("Housing Stability at Entry (With Goal for Housing) vs. Goal Success")
    
    # Calculate success rates by entry stability category
    entry_stability_success = calculate_proportions(stability_df[stability_df['DOMAIN__C'] == 'Housing'], 'entry_stability')
    
    fig = px.bar(
        entry_stability_success.reset_index(),
        x='entry_stability',
        y='success_rate',
        text='success_rate',
        color='success_rate',
        color_continuous_scale='RdYlGn',
        labels={'entry_stability': 'Housing Stability at Entry', 'success_rate': 'Success Rate (%)'}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        xaxis_title='Housing Stability Category',
        yaxis_title='Success Rate (%)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # # Housing stability change analysis
    # st.subheader("Housing Stability Change Analysis")
    
    # # Filter for cases with both entry and exit data
    # stability_change_df = stability_df.dropna(subset=['entry_severity', 'exit_severity'])
    
    # # Calculate overall stability change
    # avg_change = stability_change_df['stability_change'].mean()
    # positive_change = (stability_change_df['stability_change'] > 0).mean() * 100
    
    # change_cols = st.columns(2)
    
    # with change_cols[0]:
    #     custom_metric(
    #         "Average Stability Change", 
    #         f"{avg_change:.2f} points",
    #         delta=avg_change,
    #         delta_color="positive" if avg_change > 0 else "negative"
    #     )
    
    # with change_cols[1]:
    #     custom_metric(
    #         "Clients with Improved Housing", 
    #         f"{positive_change:.1f}%"
    #     )
    
    # Stability change by category
    # st.subheader("Housing Stability Change by Category")
    stability_change_df = stability_df.dropna(subset=['entry_severity', 'exit_severity'])
    
    # # Calculate distribution of stability changes
    stability_category_counts = stability_change_df['stability_category'].value_counts(normalize=True) * 100
    
    # fig = px.pie(
    #     values=stability_category_counts.values,
    #     names=stability_category_counts.index,
    #     title="Distribution of Housing Stability Changes",
    #     color_discrete_sequence=px.colors.diverging.RdYlGn,
    #     hole=0.4
    # )
    # fig.update_traces(
    #     textposition='inside',
    #     textinfo='percent+label',
    #     insidetextorientation='radial'
    # )
    # st.plotly_chart(fig, use_container_width=True)
    
    # # Entry and exit severity comparison
    # st.subheader("Entry vs. Exit Housing Stability")
    
    # # Create violin plot to compare distributions
    # fig = go.Figure()
    
    # fig.add_trace(go.Violin(
    #     x=['Entry'] * len(stability_change_df),
    #     y=stability_change_df['entry_severity'],
    #     box_visible=True,
    #     line_color='#e74c3c',
    #     meanline_visible=True,
    #     name='Entry'
    # ))
    
    # fig.add_trace(go.Violin(
    #     x=['Exit'] * len(stability_change_df),
    #     y=stability_change_df['exit_severity'],
    #     box_visible=True,
    #     line_color='#2ecc71',
    #     meanline_visible=True,
    #     name='Exit'
    # ))
    
    # fig.update_layout(
    #     xaxis_title='',
    #     yaxis_title='Housing Stability Score',
    #     violinmode='group',
    #     plot_bgcolor='rgba(0,0,0,0)'
    # )
    
    # st.plotly_chart(fig, use_container_width=True)
    
    # # Housing stability change by domain
    # # st.subheader("Housing Stability Change by Domain")
    
    # if 'DOMAIN__C' in stability_change_df.columns:
    #     domain_stability = stability_change_df.groupby('DOMAIN__C')['stability_change'].mean().sort_values(ascending=False)
        
    #     fig = px.bar(
    #         domain_stability.reset_index(),
    #         x='DOMAIN__C',
    #         y='stability_change',
    #         text=domain_stability.values.round(2),
    #         color='stability_change',
    #         color_continuous_scale='RdYlGn',
    #         labels={'DOMAIN__C': 'Domain', 'stability_change': 'Avg. Stability Change'}
    #     )
    #     fig.update_traces(textposition='outside')
    #     fig.update_layout(
    #         xaxis_title='Domain',
    #         yaxis_title='Average Stability Change',
    #         xaxis_tickangle=45,
    #         plot_bgcolor='rgba(0,0,0,0)',
    #         height=600
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
    
    # Housing pathways visualization
    st.subheader("Common Housing Stability Pathways")
    
    # Create a simplified version for visualization
    stability_change_df['entry_category'] = pd.cut(
        stability_change_df['entry_severity'],
        bins=[0, 3, 6, 9, 13],
        labels=['Crisis', 'Temporary', 'Transitional', 'Stable']
    )
    
    stability_change_df['exit_category'] = pd.cut(
        stability_change_df['exit_severity'],
        bins=[0, 3, 6, 9, 13],
        labels=['Crisis', 'Temporary', 'Transitional', 'Stable']
    )
    
    # Create Sankey diagram for stability category flows
    fig = create_sankey_diagram(
        stability_change_df, 
        'entry_category', 
        'exit_category'
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to visualize housing stability pathways.")

# NEW: Demographics tab
with tabs[4]:
    st.header("Demographics Analysis")
    
    # Identify demographic columns
    demo_cols = [col for col in ['AGE_GROUP', 'GENDER', 'RACE', 'ETHNICITY','COMBINED_RACE'] if col in filtered_df.columns]
    
    if not demo_cols:
        st.warning("No demographic data columns found in the dataset.")
    else:
        st.markdown("""
        <div class="info-card">
        <h4>Demographics Overview</h4>
        <p>This section analyzes how various demographic factors correlate with goal achievement
        and housing stability outcomes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create demographics dictionary
        demographics = add_demographics_analysis(filtered_df, demo_cols)
        
        # Demographics overview
        st.subheader("Client Demographics Overview")
        
        # Create columns based on number of available demographics
        num_cols = min(len(demo_cols), 2)  # Maximum 2 columns
        demo_overview_cols = st.columns(num_cols)
        
        # Distribute pie charts across columns
        for i, col in enumerate(demo_cols):
            with demo_overview_cols[i % num_cols]:
                if col in demographics:
                    # Create pie chart for demographic distribution
                    fig = px.pie(
                        demographics[col],
                        values='Percentage',
                        names='Category',
                        title=f"{col.replace('_', ' ').title()} Distribution",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        insidetextorientation='radial'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Success rates by demographics
        st.subheader("Success Rates by Demographics (All Objectives)")
        
        # Create tabs for each demographic category
        if demo_cols:
            demo_tabs = st.tabs([col.replace('_', ' ').title() for col in demo_cols])
            
            for i, col in enumerate(demo_cols):
                with demo_tabs[i]:
                    if col in demographics:
                        # Sort by success rate
                        success_data = demographics[col].sort_values('success_rate', ascending=False)
                        
                        # Create bar chart of success rates
                        fig = px.bar(
                            success_data,
                            x='Category',
                            y='success_rate',
                            text='success_rate',
                            color='success_rate',
                            color_continuous_scale='RdYlGn',
                            labels={'success_rate': 'Success Rate (%)', 'Category': col.replace('_', ' ').title()},
                            
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(
                            xaxis_title=col.replace('_', ' ').title(),
                            yaxis_title='Success Rate (%)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=700
                        )
                        st.plotly_chart(fig, use_container_width=True)
                      
        
        # Housing stability by demographics
        st.subheader("Housing Stability by Demographics")
        
        # Create columns for each demographic category (up to 2)
        num_cols = min(len(demo_cols), 2)
        stability_cols = st.columns(num_cols)
        
        for i, col in enumerate(demo_cols):
            with stability_cols[i % num_cols]:
                if col in stability_df.columns:
                    # Calculate average stability change by demographic category
                    stability_df_housing = stability_df.copy()
                    stability_df_housing = stability_df_housing[stability_df_housing['DOMAIN__C'] == 'Housing']
                    stability_by_demo = stability_df_housing.groupby(col)['stability_change'].mean().sort_values(ascending=False)
                    
                    
                    if not stability_by_demo.empty:
                        fig = px.bar(
                            stability_by_demo.reset_index(),
                            x=col,
                            y='stability_change',
                            text=stability_by_demo.values.round(2),
                            color='stability_change',
                            color_continuous_scale='RdYlGn',
                            labels={col: col.replace('_', ' ').title(), 'stability_change': 'Avg. Stability Change'}
                        )
                        fig.update_traces(textposition='outside')
                        fig.update_layout(
                            title=f"Housing Stability Change by {col.replace('_', ' ').title()}",
                            xaxis_title=col.replace('_', ' ').title(),
                            yaxis_title='Average Stability Change',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=700
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Demographic intersections
        if len(demo_cols) >= 2:
            st.subheader("Demographic Intersections")
            
            # Select two demographics for intersection analysis
            intersection_cols = st.columns(2)
            
            with intersection_cols[0]:
                demo1 = st.selectbox("Select first demographic:", demo_cols, key="demo1")
            
            with intersection_cols[1]:
                demo2 = st.selectbox("Select second demographic:", [col for col in demo_cols if col != demo1], key="demo2")
            
            if demo1 != demo2 and demo1 in filtered_df.columns and demo2 in filtered_df.columns:
                # Create intersection column
                filtered_df['intersection'] = filtered_df[demo1] + " + " + filtered_df[demo2]
                
                # Calculate success rates for intersections
                intersection_props = calculate_proportions(filtered_df, 'intersection')
                
                # Filter for intersections with enough data
                intersection_props = intersection_props[intersection_props['total'] >= 10].sort_values('success_rate', ascending=False)
                
                if not intersection_props.empty:
                    # Show top and bottom intersections
                    top_n = min(5, len(intersection_props))
                    
                    outcome_cols = st.columns(2)
                    
                    with outcome_cols[0]:
                        st.markdown(f"#### Highest Success: {demo1} × {demo2}")
                        
                        for idx, (intersect, row) in enumerate(intersection_props.head(top_n).iterrows()):
                            st.markdown(f"""
                            <div class="success-card">
                                <h5>{intersect}</h5>
                                <p>Success Rate: {row['success_rate']:.1f}%<br>
                                Sample Size: {row['total']} clients</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with outcome_cols[1]:
                        st.markdown(f"#### Lowest Success: {demo1} × {demo2}")
                        
                        for idx, (intersect, row) in enumerate(intersection_props.tail(top_n).iterrows()):
                            st.markdown(f"""
                            <div class="failure-card">
                                <h5>{intersect}</h5>
                                <p>Success Rate: {row['success_rate']:.1f}%<br>
                                Sample Size: {row['total']} clients</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info(f"Not enough data for meaningful intersection analysis of {demo1} and {demo2}.")

# Time Analysis tab
with tabs[5]:
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
    
    # Boxplot comparison
    st.subheader("Time Distribution Comparison")
    
    # col1, col2 = st.columns(2)
    # col2 = st.columns(1)
    # with col1:
    #     fig = px.box(
    #         filtered_df, 
    #         x='GOAL_STATUS_BINARY', 
    #         y='TIME_TO_COMPLETE',
    #         color='GOAL_STATUS_BINARY',
    #         labels={'TIME_TO_COMPLETE': 'Time (days)'},
    #         category_orders={"GOAL_STATUS_BINARY": [1, 0]},
    #         color_discrete_map={1: color_success, 0: color_failure}
    #     )
    #     fig.update_layout(
    #         xaxis=dict(
    #             tickmode='array',
    #             tickvals=[0, 1],
    #             ticktext=['Failed', 'Successful']
    #         ),
    #         plot_bgcolor='rgba(0,0,0,0)'
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
    
    
    st.subheader("Average Time by Domain")
    domain_time = filtered_df.groupby(['DOMAIN__C', 'GOAL_STATUS_BINARY'])['TIME_TO_COMPLETE'].mean().unstack()
    
    if not domain_time.empty:
        # Rename columns for clarity
        if 1 in domain_time.columns:
            domain_time.rename(columns={1: 'Successful'}, inplace=True)
        if 0 in domain_time.columns:
            domain_time.rename(columns={0: 'Unsuccessful'}, inplace=True)
        
        fig = px.bar(
            domain_time.reset_index(), 
            x='DOMAIN__C',
            y=['Successful', 'Unsuccessful'] if ('Successful' in domain_time.columns and 'Unsuccessful' in domain_time.columns) else domain_time.columns,
            barmode='group',
            labels={'DOMAIN__C': 'Domain', 'value': 'Average Time (days)', 'variable': 'Goal Status'},
            color_discrete_map={'Successful': color_success, 'Unsuccessful': color_failure}
        )
        fig.update_layout(
            xaxis_title='Domain',
            yaxis_title='Average Time (days)',
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to analyze time by domain.")
    
    
    
    # Success rate by duration
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
        xaxis_title="Service Duration",
        yaxis_title="Success Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly success rate heatmap
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

# Transitions tab (improved to handle missing values)
# with tabs[6]:
#     st.header("Housing Transitions Analysis")
    
    
    
#     # Living Situation Transition Analysis with improved Sankey diagram
#     st.subheader("Housing Situation Flow Analysis")
    
#     # Filter out missing values for transition analysis
#     transition_df = filtered_df.dropna(subset=['LIVING_SITUATION_AT_ENTRY__C', 'LIVING_SITUATION_AT_EXIT__C'])
    
#     # Further filter out "Missing" values
#     transition_df = transition_df[
#         (transition_df['LIVING_SITUATION_AT_ENTRY__C'] != "Missing") & 
#         (transition_df['LIVING_SITUATION_AT_EXIT__C'] != "Missing")
#     ]
    
#     if not transition_df.empty:
#         # Create a better Sankey diagram for transitions
#         with st.spinner("Generating flow diagram..."):
#             fig = create_sankey_diagram(
#                 transition_df, 
#                 'LIVING_SITUATION_AT_ENTRY__C', 
#                 'LIVING_SITUATION_AT_EXIT__C'
#             )
            
#             if fig:
#                 # Fix layout issues by adjusting the figure size and margins
#                 fig.update_layout(
#                     height=600,  # Increase height
#                     width=800,   # Set explicit width
#                     margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins
#                     font=dict(size=10),  # Reduce font size for better fit
#                     autosize=True  # Enable autosize
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
#                 # <h4>How to Read This Diagram</h4>
#                 # <p>This Sankey diagram shows the flow of clients from their entry living situation (left) to their exit 
#                 # living situation (right). The width of each flow represents the number of clients who took that path.</p>
#                 st.markdown("""
#                 <div class="info-card">
                
#                 <p>Wider connections indicate more common transitions. Colors indicate housing stability level:</p>
#                 <ul>
#                     <li><span style="color:#e74c3c">■</span> <b>Red</b>: Crisis/Emergency (Stability Score 1-3)</li>
#                     <li><span style="color:#f39c12">■</span> <b>Orange</b>: Temporary (Stability Score 4-6)</li>
#                     <li><span style="color:#3498db">■</span> <b>Blue</b>: Transitional (Stability Score 7-9)</li>
#                     <li><span style="color:#2ecc71">■</span> <b>Green</b>: Stable (Stability Score 10-13)</li>
#                 </ul>
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.info("Not enough valid transition data to generate the flow diagram.")
#     else:
#         st.info("No valid transition data available.")
    
#     # # Add traditional transition analysis as well
#     # st.subheader("Most Common Housing Transitions")
    
#     # # Analyze transitions between entry and exit situations
#     # success_df = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 1]
    
#     # if not success_df.empty:
#     #     # Remove missing values
#     #     transition_df = success_df.dropna(subset=['LIVING_SITUATION_AT_ENTRY__C', 'LIVING_SITUATION_AT_EXIT__C'])
#     #     transition_df = transition_df[
#     #         (transition_df['LIVING_SITUATION_AT_ENTRY__C'] != "Missing") & 
#     #         (transition_df['LIVING_SITUATION_AT_EXIT__C'] != "Missing")
#     #     ]
        
#     #     if not transition_df.empty:
#     #         # Get counts of each transition
#     #         transitions = transition_df.groupby(['LIVING_SITUATION_AT_ENTRY__C', 'LIVING_SITUATION_AT_EXIT__C']).size().reset_index()
#     #         transitions.columns = ['Entry', 'Exit', 'Count']
            
#     #         # Calculate proportions
#     #         total_by_entry = transitions.groupby('Entry')['Count'].sum().reset_index()
#     #         transitions = transitions.merge(total_by_entry, on='Entry', suffixes=('', '_total'))
#     #         transitions['Proportion'] = (transitions['Count'] / transitions['Count_total'] * 100).round(1)
            
#     #         # Filter for top transitions for readability
#     #         top_transitions = transitions.sort_values('Count', ascending=False).head(15)
            
#     #         fig = px.bar(
#     #             top_transitions,
#     #             y='Entry',
#     #             x='Proportion',
#     #             color='Exit',
#     #             labels={'Proportion': 'Proportion (%)', 'Entry': 'Living Situation at Entry'},
#     #             orientation='h'
#     #         )
#     #         fig.update_layout(
#     #             height=600,
#     #             legend_title="Living Situation at Exit",
#     #             plot_bgcolor='rgba(0,0,0,0)'
#     #         )
#     #         st.plotly_chart(fig, use_container_width=True)
            
#     #         # Add information about top transitions
#     #         st.markdown("#### Most Common Transitions (Entry → Exit)")
            
#     #         top_5_transitions = transitions.sort_values('Count', ascending=False).head(5)
#     #         for idx, row in top_5_transitions.iterrows():
#     #             # Determine if this is a positive transition (higher stability)
#     #             entry_score = housing_severity_scale.get(row['Entry'], 0)
#     #             exit_score = housing_severity_scale.get(row['Exit'], 0)
#     #             stability_change = exit_score - entry_score
                
#     #             card_class = "success-card" if stability_change > 0 else "warning-card" if stability_change < 0 else "info-card"
                
#     #             st.markdown(f"""
#     #             <div class="{card_class}">
#     #                 <h5>{row['Entry']} → {row['Exit']}</h5>
#     #                 <p>Count: {row['Count']} clients ({row['Proportion']:.1f}% of clients from this entry situation)<br>
#     #                 Stability Change: {stability_change:.1f} points</p>
#     #             </div>
#     #             """, unsafe_allow_html=True)
#     #     else:
#     #         st.info("No valid transition data available for successful goals.")
#     # else:
#     #     st.info("No successful goals in the current filtered data.")
    
#     # Transition Matrix with improved styling
#     st.subheader("Housing Stability Transition Matrix")
    
#     # Create a simplified version for the matrix to avoid clutter
#     if not transition_df.empty:
#         # Add housing stability categories
#         transition_df['entry_category'] = pd.cut(
#             transition_df['LIVING_SITUATION_AT_ENTRY__C'].map(housing_severity_scale),
#             bins=[0, 3, 6, 9, 13],
#             labels=['Crisis', 'Temporary', 'Transitional', 'Stable']
#         )
        
#         transition_df['exit_category'] = pd.cut(
#             transition_df['LIVING_SITUATION_AT_EXIT__C'].map(housing_severity_scale),
#             bins=[0, 3, 6, 9, 13],
#             labels=['Crisis', 'Temporary', 'Transitional', 'Stable']
#         )
        
#         # Aggregate transitions by category
#         category_transitions = pd.crosstab(
#             transition_df['entry_category'], 
#             transition_df['exit_category'],
#             normalize='index'
#         ) * 100
        
#         # Create heatmap with annotations
#         fig_heatmap = ff.create_annotated_heatmap(
#             z=category_transitions.values,
#             x=category_transitions.columns.tolist(),
#             y=category_transitions.index.tolist(),
#             colorscale="RdYlGn",
#             annotation_text=category_transitions.round(1).astype(str).values,
#             showscale=True
#         )
        
#         fig_heatmap.update_layout(
#             title="Transition Percentages Between Housing Stability Categories",
#             height=400,
#             xaxis=dict(title="Exit Housing Category"),
#             yaxis=dict(title="Entry Housing Category")
#         )
        
#         st.plotly_chart(fig_heatmap, use_container_width=True)
    
# Data explorer tab
with tabs[6]:
    st.header("Data Explorer")
    
    # Category Distribution Analysis
    st.subheader("Category Distribution Analysis")
    st.markdown("""
    This section shows the distribution of categories within selected columns as pie charts.
    Small categories below the threshold are grouped into an 'OTHERS' category for better visualization.
    """)
    
    # Define columns for pie chart analysis
    pie_chart_columns = [
        col for col in [
            'DOMAIN__C',  'LIVING_SITUATION_AT_ENTRY__C', #'OUTCOME__C',
            'LIVING_SITUATION_AT_EXIT__C', 'REASON_CLOSED__C'
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
    
    # Housing severity analysis by categories
    # st.subheader("Housing Severity Analysis")
    
    # # Allow user to select column for analysis
    # severity_column = st.selectbox(
    #     "Select category to analyze by housing severity:", 
    #     pie_chart_columns, 
    #     key="severity_select"
    # )
    
    # if severity_column in filtered_df.columns:
    #     # Add housing severity scores to the filtered data
    #     severity_analysis_df = filtered_df.copy()
    #     severity_analysis_df['entry_severity'] = severity_analysis_df['LIVING_SITUATION_AT_ENTRY__C'].map(housing_severity_scale)
        
    #     # Calculate average severity by selected category
    #     severity_by_category = severity_analysis_df.groupby(severity_column)['entry_severity'].mean().sort_values()
        
    #     fig = px.bar(
    #         severity_by_category.reset_index(),
    #         x=severity_column,
    #         y='entry_severity',
    #         text=severity_by_category.values.round(2),
    #         color='entry_severity',
    #         color_continuous_scale='RdYlGn',
    #         labels={
    #             severity_column: severity_column.replace('__C', ''),
    #             'entry_severity': 'Avg. Housing Stability Score'
    #         }
    #     )
    #     fig.update_traces(textposition='outside')
    #     fig.update_layout(
    #         xaxis_title=severity_column.replace('__C', ''),
    #         yaxis_title='Average Housing Stability Score',
    #         xaxis_tickangle=45,
    #         plot_bgcolor='rgba(0,0,0,0)',
    #         height=700
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
    
    # # Raw data explorer
    # st.subheader("Raw Data Explorer")
    # if st.checkbox("Show Raw Data"):
    #     st.dataframe(filtered_df)
        
    #     # Add a CSV download button
    #     csv = filtered_df.to_csv(index=False)
    #     st.download_button(
    #         label="Download Data as CSV",
    #         data=csv,
    #         file_name="homelessness_services_data.csv",
    #         mime="text/csv"
    #     )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>Dashboard created for NGO homelessness services data analysis</p>
<p>Last updated: March 2025</p>
</div>
""", unsafe_allow_html=True)