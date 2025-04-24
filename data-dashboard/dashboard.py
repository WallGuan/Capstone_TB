import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import calendar
from scipy.stats import chi2_contingency
import warnings
import io
warnings.filterwarnings("ignore")

# Define color palette for consistency
color_success = "#90F7B7"  # Light mint green
color_failure = "#E83DDC"  # Magenta
color_neutral = "#32BDC2"  # Teal
color_dark = "#3300CC"     # Deep navy/indigo
color_accent = "#39D9CB"   # Bright teal
color_secondary = "#4ADCB2"  # Green-teal
color_highlight = "#6A22CF"  # Purple

# Custom color palettes for different types of charts
color_gradient_success = ["#90F7B7", "#4ADCB2", "#32BDC2"]  # Green gradient
color_gradient_failure = ["#6A22CF", "#9B4DDD", "#E83DDC"]  # Purple gradient
color_palette = [color_highlight, color_accent, color_secondary, color_success, color_failure, color_neutral]

# Create custom color scales for continuous variables
custom_rdylgn = [[0, "#E83DDC"], [0.5, "#32BDC2"], [1, "#90F7B7"]]  # Purple to teal to mint
custom_blue_purple = [[0, "#32BDC2"], [1, "#6A22CF"]]  # Teal to Purple
custom_reds = [[0, "#F8F9FA"], [1, "#E83DDC"]]  # White to magenta
custom_greens = [[0, "#F8F9FA"], [1, "#32BDC2"]]  # White to teal

# Page configuration
st.set_page_config(
    page_title="HomeBridger Data Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'Dashboard for analyzing HomeBridger services data'
    }
)

# Configure Streamlit theme
st.markdown("""
    <style>
        /* Override Streamlit's default theme */
        :root {
            --primary-color: #3300CC;
            --background-color: #f8f9fa;
            --secondary-background-color: #ffffff;
            --text-color: #3300CC;
            --font: 'Segoe UI', sans-serif;
        }
        
        /* Override Streamlit elements */
        .stSelectbox > div > div > div {
            background-color: #3300CC !important;
            color: white !important;
        }
        
        .stMultiSelect > div > div > div > div > div {
            background-color: #3300CC !important;
            color: white !important;
        }
        
        /* Style multiselect tags */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #3300CC !important;
            border: none !important;
            border-radius: 4px !important;
            color: white !important;
            font-size: 14px !important;
        }
        
        /* Style the X button in multiselect tags */
        .stMultiSelect [data-baseweb="tag"] span[role="button"] {
            color: white !important;
        }
        
        .stMultiSelect [data-baseweb="tag"] span[role="button"]:hover {
            color: #E83DDC !important;
        }
        
        /* Style the dropdown items */
        .stMultiSelect [role="listbox"] div {
            color: #3300CC !important;
        }
        
        /* Style selected items in dropdown */
        .stMultiSelect [role="listbox"] [aria-selected="true"] {
            background-color: rgba(26, 20, 100, 0.1) !important;
        }
        
        /* Sidebar text and title colors */
        .css-1v0mbdj.e115fcil1,
        .css-163ttbj.e1fqkh3o11 {
            color: #3300CC !important;
        }
        
        /* Sidebar selected items */
        .css-1v0mbdj.e115fcil1 [data-baseweb="tag"] {
            background-color: #3300CC !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Housing severity scale mapping
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
        .main { background-color: #f8f9fa; }
        h1, h2, h3 { 
            color: #3300CC;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1 { border-bottom: 2px solid #39D9CB; padding-bottom: 10px; }
        .stMetric {
            background-color: white;
            border-radius: 5px;
            padding: 15px 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding: 10px 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f7f5;
            border-bottom: 2px solid #39D9CB;
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
            color: #3300CC;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #3300CC !important;
        }
        .success-card {
            background-color: #e6f7f2;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #90F7B7;
            margin-bottom: 20px;
        }
        .warning-card {
            background-color: #f3e9f7;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #9B4DDD;
            margin-bottom: 20px;
        }
        .info-card {
            background-color: #e6f7f5;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #32BDC2;
            margin-bottom: 20px;
        }
        .failure-card {
            background-color: #f7e6f5;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #E83DDC;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
        
        /* Style for the multiselect pills in sidebar */
        .stMultiSelect div div div div div:has(span[data-baseweb="tag"]) span[data-baseweb="tag"] {
            background-color: #3300CC !important;
        }
        
        .stMultiSelect div div div div div:has(span[data-baseweb="tag"]) span[data-baseweb="tag"] span:first-child {
            color: white !important;
        }
        
        .stMultiSelect div div div div div:has(span[data-baseweb="tag"]) span[data-baseweb="tag"] span:nth-child(2) svg {
            fill: white !important;
        }
        
        /* Style for the sidebar header */
        .sidebar .sidebar-content .block-container h1, 
        .sidebar .sidebar-content .block-container h2, 
        .sidebar .sidebar-content .block-container h3, 
        .sidebar .sidebar-content .block-container h4 {
            color: #3300CC !important;
            font-weight: 600;
        }
        
        /* Strengthen multiselect styling */
        div[data-baseweb="select"] span[data-baseweb="tag"] {
            background-color: #3300CC !important;
            border: none !important;
            color: white !important;
        }
        
        div[data-baseweb="select"] span[data-baseweb="tag"] span {
            color: white !important;
        }
        
        div[data-baseweb="select"] span[data-baseweb="tag"] button {
            color: white !important;
        }
        
        /* Override any Streamlit defaults */
        .stMultiSelect div[role="button"] {
            background-color: #3300CC !important;
            color: white !important;
        }
        
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #3300CC !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply custom theme
set_custom_theme()

# Custom metric component
def custom_metric(title, value, delta=None, delta_color="normal"):
    """Creates a custom styled metric component."""
    delta_colors = {
        "positive": "#90F7B7",
        "negative": "#E83DDC",
        "normal": "#3300CC"
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
        <div class="metric-value" style="color: #3300CC;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

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
    valid_df = df.dropna(subset=[source_col, target_col])
    valid_df = valid_df[(valid_df[source_col] != "Missing") & (valid_df[target_col] != "Missing")]
    
    if valid_df.empty:
        return None
    
    if value_col:
        sankey_df = valid_df.groupby([source_col, target_col])[value_col].sum().reset_index()
    else:
        sankey_df = valid_df.groupby([source_col, target_col]).size().reset_index()
        sankey_df.columns = [source_col, target_col, 'value']
    
    all_nodes = pd.unique(sankey_df[[source_col, target_col]].values.ravel('K'))
    all_nodes = [str(node) for node in all_nodes if str(node) != 'nan']
    shortened_nodes = [node[:30] + '...' if len(node) > 30 else node for node in all_nodes]
    
    source_indices = [list(all_nodes).index(str(source)) for source in sankey_df[source_col]]
    target_indices = [list(all_nodes).index(str(target)) for target in sankey_df[target_col]]
    
    node_colors = []
    for node in all_nodes:
        severity = housing_severity_scale.get(node, 6)
        if severity <= 3:
            node_colors.append('#E83DDC')  # Magenta for crisis
        elif severity <= 6:
            node_colors.append('#9B4DDD')  # Mid-purple for temporary
        elif severity <= 9:
            node_colors.append('#32BDC2')  # Teal for transitional
        else:
            node_colors.append('#90F7B7')  # Mint green for stable
    
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
            hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title_text=f"Flow from {source_col} to {target_col}",
        height=700,
        font=dict(size=12, color=color_dark),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# NEW FUNCTION: Add housing stability analysis based on severity scale
def add_housing_stability_analysis(df):
    """Add housing stability score based on the severity scale and analyze outcomes."""
    analysis_df = df.copy()
    
    analysis_df['entry_severity'] = analysis_df['LIVING_SITUATION_AT_ENTRY__C'].map(housing_severity_scale)
    analysis_df['exit_severity'] = analysis_df['LIVING_SITUATION_AT_EXIT__C'].map(housing_severity_scale)
    analysis_df['stability_change'] = analysis_df['exit_severity'] - analysis_df['entry_severity']
    
    stability_bins = [-float('inf'), -2, -0.001, 0.001, 2, float('inf')]
    stability_labels = ['Significant Decline', 'Slight Decline', 'No Change', 'Slight Improvement', 'Significant Improvement']
    
    analysis_df['stability_category'] = pd.cut(
        analysis_df['stability_change'],
        bins=stability_bins,
        labels=stability_labels
    )
    
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
            counts = df[col].value_counts().reset_index()
            counts.columns = ['Category', 'Count']
            counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
            
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
                mean_val = group_df[metric].mean()
                metric_values.append(mean_val)
            else:
                metric_values.append(0)
        
        metric_values.append(metric_values[0])
        metrics_with_first = metrics + [metrics[0]]
        
        radar_data.append(go.Scatterpolar(
            r=metric_values,
            theta=metrics_with_first,
            fill='toself',
            name=group_labels[i],
            line=dict(color=color_success if value == 1 else color_failure),
            fillcolor=f"rgba({int(color_success[1:3], 16)}, {int(color_success[3:5], 16)}, {int(color_success[5:7], 16)}, 0.3)" if value == 1 else f"rgba({int(color_failure[1:3], 16)}, {int(color_failure[3:5], 16)}, {int(color_failure[5:7], 16)}, 0.3)"
        ))
    
    fig = go.Figure(data=radar_data)
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2 * max([max(trace['r']) for trace in radar_data])],
                linecolor=color_dark,
                gridcolor=f"rgba({int(color_dark[1:3], 16)}, {int(color_dark[3:5], 16)}, {int(color_dark[5:7], 16)}, 0.2)"
            ),
            angularaxis=dict(
                linecolor=color_dark,
                gridcolor=f"rgba({int(color_dark[1:3], 16)}, {int(color_dark[3:5], 16)}, {int(color_dark[5:7], 16)}, 0.1)"
            )
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=color_dark)
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
        
        # Add housing stability analysis
        if 'LIVING_SITUATION_AT_ENTRY__C' in df.columns and 'LIVING_SITUATION_AT_EXIT__C' in df.columns:
            stability_df = add_housing_stability_analysis(df)
            stability_analysis = calculate_proportions(stability_df, 'stability_category')
            stability_analysis.to_excel(writer, sheet_name='Housing Stability Analysis')
        
        # Add demographics analysis
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
    """Load and cache the data."""
    try:
        df = pd.read_csv('/Users/natehu/Desktop/TechBridge/QTM 498R Capstone/data dashboard/dat/all.csv')
        return df
    except Exception as e:
        st.error(f"Could not load the data file: {str(e)}")
        st.stop()

# Dashboard title and description
st.title("HomeBridger Data Dashboard")
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

# Filter out EGOs with less than 100 rows
ego_counts = df['RECORD_ORIGIN__C_y'].value_counts()
valid_egos = ego_counts[ego_counts >= 100].index.tolist()

if not valid_egos:
    st.error("No EGOs have 100 or more records. Please adjust the minimum record threshold.")
    st.stop()

# Filter the dataframe to only include valid EGOs
df = df[df['RECORD_ORIGIN__C_y'].isin(valid_egos)]

# Sidebar filters
st.sidebar.header("Filters")

# EGOs filter
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

# Add housing stability metrics
stability_df = add_housing_stability_analysis(filtered_df)

# Create dashboard tabs
tabs = st.tabs([
    "📊 Overview",              # general summary
    "🚀 Success Factors",       # upward/rocket for positive drivers
    "⚠️ Failure Analysis",      # warning sign for issues/failures
    "🏘️ Housing Stability",     # cluster of homes for community/housing
    "🧑‍🤝‍🧑 Demographics",       # diverse people icon for demographic spread
    "⏳ Time Analysis",         # hourglass for passage of time
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
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=color_dark)
    )
    st.plotly_chart(fig, use_container_width=True, key="overview_success_rate")
    
    # Top Reasons for Case Closure
    st.subheader("Top Reasons for Case Closure")
    closure_reasons = filtered_df["REASON_CLOSED__C"].value_counts(normalize=True).head(10)
    fig = px.bar(
        x=closure_reasons.index,
        y=closure_reasons.values * 100,
        text=closure_reasons.values * 100,
        labels={"x": "Reason for Closure", "y": "Proportion (%)"},
        color=closure_reasons.values,
        color_continuous_scale=custom_rdylgn,
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=color_dark)
    )
    st.plotly_chart(fig, use_container_width=True, key="overview_closure_reasons")
    
    
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
        

    
    with comparison_cols[1]:
        # Calculate success rates with minimum category size filter
        category_counts = filtered_df[compare_by].value_counts()
        valid_categories = category_counts[category_counts >= min_count].index
        
        compare_df = filtered_df[filtered_df[compare_by].isin(valid_categories)]
        
        if not compare_df.empty:
            success_by_category = calculate_proportions(compare_df, compare_by)
            
        
            # Create enhanced visualization
            fig = px.bar(
                success_by_category.reset_index(),
                x=compare_by,
                y='success_rate',
                text='success_rate',
                color='success_rate',
                color_continuous_scale=custom_rdylgn,
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
            
            st.plotly_chart(fig, use_container_width=True, key=f"success_rate_{compare_by}")
            
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
    st.subheader("Living Situation at Entry Success Rates")
    
    
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
                color_continuous_scale=custom_rdylgn,
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
            st.plotly_chart(fig, use_container_width=True, key=f"combined_success_factors")
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
        color_continuous_scale=custom_reds,
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

    st.plotly_chart(fig, use_container_width=True, key="domain_failure_rates")
    # Living situations with highest failure rates
    st.subheader("Living Situations at Entry with Highest Failure Rates (Housing Domain Only)")
    
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
        color_continuous_scale=custom_reds,
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
    
    st.plotly_chart(fig, use_container_width=True, key="living_situation_failure_rates")
    
    # Reason for closure distribution among failed goals
    st.subheader("Primary Reasons for Goal Failure")
    
    if 'REASON_CLOSED__C' in failure_df.columns:
        failure_reasons = failure_df['REASON_CLOSED__C'].value_counts(normalize=True).head(10)
        
        fig = px.pie(
            values=failure_reasons.values * 100,
            names=failure_reasons.index,
            title="Reasons for Goal Failure",
            color_discrete_sequence=color_gradient_failure,
            hole=0.4
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial'
        )
        st.plotly_chart(fig, use_container_width=True, key="failure_reasons_pie")
    
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
                color_continuous_scale=custom_rdylgn,
                text_auto=".1f"
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Month",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="time_monthly_success_heatmap")


# Housing Stability tab
with tabs[3]:
    st.header("Housing Stability Analysis")

    house_df = filtered_df.copy()
    
    original_count = len(house_df)
    
    # both entry and exit living situation are not  "missing"
    house_df = house_df[house_df['LIVING_SITUATION_AT_ENTRY__C'] != "Missing"]
    house_df = house_df[house_df['LIVING_SITUATION_AT_EXIT__C'] != "Missing"]
    # Add housing stability scores
    stability_df = add_housing_stability_analysis(house_df)
    
    # Key Metrics Section
    st.subheader("Key Housing Stability Metrics")
    
    # Calculate metrics for meaningful indicators
    avg_stability_change = stability_df['stability_change'].mean()
    stability_improved_pct = (stability_df['stability_change'] > 0).mean() * 100
    stability_unchanged_pct = (stability_df['stability_change'] == 0).mean() * 100
    stability_declined_pct = (stability_df['stability_change'] < 0).mean() * 100
    
    # Calculate data completeness - corrected calculation 
    entry_missing_count = (stability_df['LIVING_SITUATION_AT_ENTRY__C'] == "Missing").sum()
    exit_missing_count = (stability_df['LIVING_SITUATION_AT_EXIT__C'] == "Missing").sum()
    total_records = len(stability_df)
    
    entry_missing_pct = (entry_missing_count / total_records) * 100 if total_records > 0 else 0
    exit_missing_pct = (exit_missing_count / total_records) * 100 if total_records > 0 else 0
    overall_completeness = (1 - (original_count - total_records) / original_count) * 100
    
    # Metric rows
    metric_cols1 = st.columns(3)
    with metric_cols1[0]:
        custom_metric("Average Stability Change", f"+ {avg_stability_change:.2f} points")
    with metric_cols1[1]:
        custom_metric("Clients with Improved Housing", f"{stability_improved_pct:.1f}%")
    with metric_cols1[2]:
        custom_metric("Data Completeness", f"{overall_completeness:.1f}%")
    
    # Cross-Organization Comparison
    st.subheader("Cross-Organization Housing Stability Comparison")
    
    # Calculate average stability change by organization
    org_stability = stability_df.groupby('RECORD_ORIGIN__C_y').agg(
        avg_change=('stability_change', 'mean'),
        improved_pct=('stability_change', lambda x: (x > 0).mean() * 100),
        count=('RECORD_ORIGIN__C_y', 'count')
    ).reset_index()
    
    # Filter for organizations with sufficient data
    org_stability = org_stability[org_stability['count'] >= 20]
    
    if not org_stability.empty:
        # Create scatter plot comparing stability metrics
        fig = px.scatter(
            org_stability,
            x='avg_change',
            y='improved_pct',
            size='count',
            color='avg_change',
            text='RECORD_ORIGIN__C_y',
            color_continuous_scale=custom_rdylgn,
            labels={
                'avg_change': 'Average Stability Change',
                'improved_pct': 'Clients with Improved Stability (%)',
                'count': 'Number of Records'
            },
            height=500
        )
        fig.update_traces(
            textposition='top center',
            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
            marker_sizemin=10,
            marker_sizeref=2.*max(org_stability['count'])/(50.**2)
        )
        fig.update_layout(
            title="Organization Performance Comparison",
            xaxis_title="Average Stability Change",
            yaxis_title="Clients with Improved Stability (%)",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add reference lines
        fig.add_hline(
            y=org_stability['improved_pct'].mean(), 
            line_dash="dash", 
            line_color="grey", 
            annotation_text="Avg Improvement %"
        )
        fig.add_vline(
            x=org_stability['avg_change'].mean(), 
            line_dash="dash", 
            line_color="grey", 
            annotation_text="Avg Change"
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True, key="org_stability_comparison")
        
        # Add insights on cross-organization comparison
        st.markdown("""
        <div class="info-card">
        <h5>Cross-Organization Insights</h5>
        <p>Organizations in the top-right quadrant are outperforming peers in improving client housing stability.
        Those in the bottom-left quadrant may benefit from shared best practices through the data platform.</p>
        <p>Size of bubbles indicates volume of clients served - larger organizations have greater impact potential.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Insufficient data to compare organizations.")
    
    # Add Housing Severity Scale visualization at the end
    st.subheader("Housing Stability Severity Scale")
    
    # Create a sorted list of the housing severity scale
    severity_items = sorted(housing_severity_scale.items(), key=lambda x: x[1])
    
    # Create a dataframe for the scale visualization
    scale_df = pd.DataFrame({
        'Housing Situation': [item[0] for item in severity_items],
        'Severity Score': [item[1] for item in severity_items],
        'Category': pd.cut(
            [item[1] for item in severity_items],
            bins=[0, 3, 6, 9, 13],
            labels=['Crisis', 'Temporary', 'Transitional', 'Stable']
        )
    })
    
    # Create a color map from purple to pink gradient
    purple_pink_scale = [
        [0, "#6A22CF"],     # Deep purple
        [0.33, "#8B3DCE"],  # Medium purple
        [0.66, "#C73DDD"],  # Light purple
        [1, "#E83DDC"]      # Pink
    ]
    
    # Create the visualization
    fig = px.bar(
        scale_df,
        x='Severity Score',
        y='Housing Situation',
        color='Severity Score',
        color_continuous_scale=purple_pink_scale,
        text='Severity Score',
        labels={
            'Severity Score': 'Housing Stability Score (higher is better)',
            'Housing Situation': 'Living Situation'
        },
        height=700,
        orientation='h'
    )
    
    # Add category bands to visually distinguish groups
    category_colors = {
        'Crisis': 'rgba(232, 61, 220, 0.15)',      # Pink with transparency
        'Temporary': 'rgba(199, 61, 221, 0.15)',   # Light purple with transparency  
        'Transitional': 'rgba(139, 61, 206, 0.15)', # Medium purple with transparency
        'Stable': 'rgba(106, 34, 207, 0.15)'       # Deep purple with transparency
    }
    
    # Add category bands
    for category in ['Crisis', 'Temporary', 'Transitional', 'Stable']:
        category_items = scale_df[scale_df['Category'] == category]
        if not category_items.empty:
            min_y_position = category_items.index.min() - 0.5
            max_y_position = category_items.index.max() + 0.5
            
            fig.add_shape(
                type="rect",
                x0=0,
                x1=13,
                y0=min_y_position,
                y1=max_y_position,
                fillcolor=category_colors[category],
                line=dict(width=0),
                layer="below"
            )
            
            # Add category label
            fig.add_annotation(
                x=12,
                y=(min_y_position + max_y_position) / 2,
                text=category,
                showarrow=False,
                font=dict(size=14, color="#3300CC")
            )
    
    # Update layout
    fig.update_layout(
        title="Housing Stability Severity Scale (Higher is More Stable)",
        xaxis=dict(
            title="Stability Score",
            range=[0, 13],
            dtick=1
        ),
        yaxis=dict(
            title=""
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True, key="housing_severity_scale")
    

with tabs[4]: # Demographics tab - UNCOMMENTED
    st.header("Demographics Analysis")

    race_col = None
    if 'COMBINED_RACE' in filtered_df.columns:
        race_col = 'COMBINED_RACE'
    elif 'RACE' in filtered_df.columns:
        race_col = 'RACE'

    if race_col:
        demo_df = filtered_df[[race_col, 'GOAL_STATUS_BINARY']].copy()
        demo_df[race_col] = demo_df[race_col].fillna('Missing')

        # --- Race Splitting Logic ---
        delimiters = r'/|;' # Regex for splitting
        
        # Create a series suitable for splitting, handle potential non-string types
        races_series = demo_df[race_col].astype(str).str.split(delimiters, expand=False)
        
        # Explode the list into separate rows, keeping the original index
        races_exploded = races_series.explode()
        
        # Trim whitespace and handle empty results
        races_exploded = races_exploded.str.strip()
        races_exploded = races_exploded.replace('', 'Unknown/Missing')
        
        # Combine with original index and goal status
        race_analysis_df = demo_df[['GOAL_STATUS_BINARY']].join(races_exploded.rename('race'))
        
        # Remove rows where race is 'Missing' or 'Unknown/Missing' for clearer analysis
        valid_races_df = race_analysis_df[
            ~race_analysis_df['race'].isin(['Missing', 'Unknown/Missing', None]) & 
            (race_analysis_df['race'] != '')
        ].copy()

        if not valid_races_df.empty:
            # --- 1. Success Rate by Race ---
            st.subheader("Success Rate by Race/Ethnicity")
            st.markdown("Analyzes goal success rates across different racial/ethnic identities, based on self-reported data. Handles multiple selections per client.")

            min_count_race = st.slider(
                "Minimum sample size for race analysis:", 
                min_value=1, 
                max_value=max(10, int(len(valid_races_df)/20)), # Dynamic max
                value=max(1, min(10, int(len(valid_races_df)/50))), # Dynamic default
                key="min_count_race",
                help="Minimum number of client entries needed to include a race category in the success rate analysis."
            )

            # Calculate success rates
            race_success_rates = valid_races_df.groupby('race')['GOAL_STATUS_BINARY'].agg(
                count='size',
                success_rate=lambda x: x.mean() * 100
            ).reset_index()
            
            # Filter by minimum count
            race_success_rates_filtered = race_success_rates[race_success_rates['count'] >= min_count_race]
            
            # Sort for horizontal bar chart (ascending looks better)
            race_success_rates_filtered = race_success_rates_filtered.sort_values('success_rate', ascending=True)

            if not race_success_rates_filtered.empty:
                fig_race_success = px.bar(
                    race_success_rates_filtered,
                    x='success_rate',
                    y='race',
                    orientation='h',
                    text='success_rate',
                    color='success_rate',
                    color_continuous_scale=custom_rdylgn, # Use defined color scale
                    labels={'race': 'Race/Ethnicity', 'success_rate': 'Success Rate (%)'},
                    hover_data={'count': True, 'success_rate': ':.1f'}, # Show count on hover
                    title="Success Rate by Race/Ethnicity"
                )
                fig_race_success.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Success Rate: %{x:.1f}%<br>Sample Size: %{customdata[0]}'
                )
                fig_race_success.update_layout(
                    yaxis_title='', # Remove y-axis title for cleaner look
                    xaxis_title='Success Rate (%)',
                    plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                    height=max(400, len(race_success_rates_filtered) * 30) # Adjust height dynamically
                )
                st.plotly_chart(fig_race_success, use_container_width=True)
            else:
                st.info(f"No race/ethnicity categories met the minimum sample size of {min_count_race}.")

            # --- 2. Overall Racial Composition ---
            st.subheader("Overall Racial Composition")
            st.markdown("Shows the distribution of racial/ethnic identities across all analyzed client entries.")
            
            # Calculate distribution based on the exploded (split) data
            race_distribution = valid_races_df['race'].value_counts(normalize=True).reset_index()
            race_distribution.columns = ['race', 'percentage']
            race_distribution['percentage'] *= 100

            # Threshold slider for grouping small categories
            threshold_comp = st.slider(
                "Threshold for grouping small races (%):",
                min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                key="threshold_comp",
                help="Races representing less than this percentage will be grouped into 'OTHERS'."
            )
            
            # Apply threshold grouping
            small_races = race_distribution[race_distribution["percentage"] < threshold_comp]
            others_percentage = small_races["percentage"].sum()
            filtered_distribution = race_distribution[race_distribution["percentage"] >= threshold_comp]

            if others_percentage > 0:
                # Ensure 'OTHERS' isn't already present before adding
                if not filtered_distribution['race'].str.contains('OTHERS').any():
                    new_row = pd.DataFrame({"race": ["OTHERS"], "percentage": [others_percentage]})
                    filtered_distribution = pd.concat([filtered_distribution, new_row], ignore_index=True)
                else:
                    # Add to existing 'OTHERS' if it exists (edge case)
                    filtered_distribution.loc[filtered_distribution['race'] == 'OTHERS', 'percentage'] += others_percentage

            if not filtered_distribution.empty:
                fig_race_comp = px.pie(
                    filtered_distribution,
                    values='percentage',
                    names='race',
                    title='Overall Racial Composition of Clients',
                    color_discrete_sequence=px.colors.sequential.Plotly3, # Use a sequential magenta scale
                    hole=0.3 # Make it a donut chart
                )
                fig_race_comp.update_traces(
                    textposition='outside', # Place labels outside
                    textinfo='percent+label',
                    # insidetextorientation='radial' # Less effective with outside labels
                )
                fig_race_comp.update_layout(
                    margin=dict(t=50, b=100, l=20, r=20), # Adjust margins
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False # Labels on slices are usually sufficient
                )
                st.plotly_chart(fig_race_comp, use_container_width=True)
            else:
                st.info("Could not generate the racial composition chart.")
            
            # --- 3. NGO Client Demographics ---
            st.subheader("NGO Client Demographics Breakdown")
            st.markdown("Shows the racial/ethnic composition of clients served by each selected NGO.")
            
            # Join valid races back to the original filtered_df to get NGO info
            # Need to ensure index alignment
            ngo_race_df = filtered_df.join(valid_races_df.set_index(valid_races_df.index)[['race']])
            ngo_race_df = ngo_race_df.dropna(subset=['race', 'RECORD_ORIGIN__C_y']) # Drop rows missing essential info
            
            # Slider for minimum clients per NGO for this analysis
            min_clients_ngo = st.slider(
                "Minimum total clients per NGO:", 
                min_value=10, 
                max_value=max(50, int(len(filtered_df) / len(filtered_df['RECORD_ORIGIN__C_y'].unique()) * 2)), # Dynamic max
                value=max(10, min(20, int(len(filtered_df) / len(filtered_df['RECORD_ORIGIN__C_y'].unique())))), # Dynamic default
                key="min_clients_ngo",
                help="Minimum number of total client entries required to include an NGO in this breakdown."
            )
            
            # Calculate client counts per NGO
            ngo_counts = ngo_race_df['RECORD_ORIGIN__C_y'].value_counts()
            valid_ngos = ngo_counts[ngo_counts >= min_clients_ngo].index
            
            # Filter the dataframe for valid NGOs
            ngo_race_filtered_df = ngo_race_df[ngo_race_df['RECORD_ORIGIN__C_y'].isin(valid_ngos)]

            if not ngo_race_filtered_df.empty:
                # Calculate race distribution per NGO
                ngo_race_dist = ngo_race_filtered_df.groupby(['RECORD_ORIGIN__C_y', 'race']).size().unstack(fill_value=0)
                ngo_race_dist_pct = ngo_race_dist.apply(lambda x: x / x.sum() * 100, axis=1)
                
                # Prepare data for plotting
                plot_data_ngo = ngo_race_dist_pct.reset_index().melt(
                    id_vars='RECORD_ORIGIN__C_y', 
                    var_name='race', 
                    value_name='percentage'
                )
                
                # Add total count for hover info
                plot_data_ngo = plot_data_ngo.merge(ngo_counts.rename('total_clients'), left_on='RECORD_ORIGIN__C_y', right_index=True)

                # Sort NGOs by total clients for better visualization (optional)
                plot_data_ngo = plot_data_ngo.sort_values(by='total_clients', ascending=False)
                
                # Create stacked bar chart
                fig_ngo_dist = px.bar(
                    plot_data_ngo,
                    x='RECORD_ORIGIN__C_y',
                    y='percentage',
                    color='race',
                    title='Racial/Ethnic Composition by NGO',
                    labels={'RECORD_ORIGIN__C_y': 'NGO (Organization)', 'percentage': 'Percentage of Clients (%)', 'race': 'Race/Ethnicity'},
                    color_discrete_sequence=px.colors.qualitative.Pastel, # Use a distinct color palette
                    hover_data={'total_clients': True, 'percentage': ':.1f'}
                )
                
                fig_ngo_dist.update_layout(
                    xaxis_tickangle=45,
                    yaxis_title='Percentage of Clients (%)',
                    xaxis_title='NGO (Organization)',
                    barmode='stack',
                    legend_title='Race/Ethnicity',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=600
                )
                
                fig_ngo_dist.update_traces(
                     hovertemplate='<b>NGO:</b> %{x}<br><b>Race:</b> %{fullData.name}<br><b>Percentage:</b> %{y:.1f}%<br><b>Total Clients for NGO:</b> %{customdata[0]}<extra></extra>'
                )
                
                st.plotly_chart(fig_ngo_dist, use_container_width=True)
            else:
                st.info(f"No NGOs met the minimum requirement of {min_clients_ngo} clients for this breakdown.")

        else:
            st.warning("No valid race/ethnicity data found after processing and cleaning.")
    else:
        st.warning("No 'COMBINED_RACE' or 'RACE' column found in the data. Cannot perform race-based analysis.")

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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=color_dark)
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
        
        st.plotly_chart(fig, use_container_width=True, key="time_distribution_histogram")
    
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
        st.plotly_chart(fig, use_container_width=True, key="avg_time_by_domain")
    else:
        st.info("Not enough data to analyze time by domain.")
    
    # When do goals typically fail?
    st.subheader("When Do Goals Typically Fail?")
    
    # Create time bins for failure analysis
    failure_df = filtered_df[filtered_df['GOAL_STATUS_BINARY'] == 0]
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
        color_continuous_scale=custom_reds
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Time Period',
        yaxis_title='Number of Failed Goals',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, key="time_failure_distribution")
    
    
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
        color_continuous_scale=custom_rdylgn,
        hover_data=['count']
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        xaxis_title="Service Duration",
        yaxis_title="Success Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="time_tab_success_rate_by_duration")
    
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
                y=heatmap_data.index,
                color_continuous_scale=custom_rdylgn,
                text_auto=".1f"
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Month",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="time_monthly_success_heatmap")

# Data Explorer tab
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
        def create_pie_chart(data, title, colors=color_palette):
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
            fig.update_layout(
                margin=dict(t=40, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Display pie charts with custom color schemes
        with pie_col1:
            fig_all = create_pie_chart(
                output_all, 
                f"All Goals - {selected_column}",
                color_palette
            )
            st.plotly_chart(fig_all, use_container_width=True, key=f"pie_all_{selected_column}")
        
        with pie_col2:
            fig_success = create_pie_chart(
                output_success, 
                f"Successful Goals - {selected_column}",
                color_gradient_success
            )
            st.plotly_chart(fig_success, use_container_width=True, key=f"pie_success_{selected_column}")
        
        with pie_col3:
            fig_failure = create_pie_chart(
                output_failure, 
                f"Unsuccessful Goals - {selected_column}",
                color_gradient_failure
            )
            st.plotly_chart(fig_failure, use_container_width=True, key=f"pie_failure_{selected_column}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {color_dark};">
<p>Dashboard created for TechBridge HomeBridger services data analysis</p>
<p>Last updated: March 2025</p>
</div>
""", unsafe_allow_html=True)
