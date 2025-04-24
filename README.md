# Emory QTM 498R Capstone Project: TechBridge HomeBridger Analytics

## Description

This Emory University capstone project, in collaboration with TechBridge, analyzed collaborative service data from the HomeBridger platform to enhance nonprofit effectiveness. The goal was to provide actionable insights for optimizing service delivery and improving client outcomes.

**Key Findings:**

*   **Service Success Variation:** Significant differences in service success rates were observed across domains (ranging from 55.5% to 100%). Adult Education and Financial services emerged as areas needing focused improvement.
*   **Housing Stability:** On average, housing stability scores improved by +2.13 points for 55.5% of clients, highlighting performance variations among participating organizations.
*   **Demographic Disparities:** Black/African American clients, despite being the majority, experienced lower success rates (75.9%), indicating potential equity gaps.
*   **Timing Impact:** Goals addressed within the first 30 days showed the highest success rates (85.5%). A critical window for increased failure rates was identified between 90-180 days.
*   **Financial Needs:** Forecasting projected monthly financial assistance requirements between $70,000 and $100,000.
*   **Client Segmentation:** Cluster analysis identified four distinct family profiles based on needs, suggesting the value of tailored service approaches.

**Recommendations:**

Our findings led to recommendations for TechBridge and partner NGOs focused on:
*   Enhancing data quality and collection practices.
*   Optimizing the sequencing and timing of service delivery.
*   Addressing identified equity gaps.
*   Implementing predictive analytics for proactive support.

These evidence-based insights aim to strengthen the collaborative network's capacity for impactful, coordinated care and sustainable client outcome improvements.

## Features
- Interactive dashboards for tracking service delivery performance.
- Predictive analytics for identifying optimal service pathways.
- Financial forecasting tools for resource allocation.
- Demographic analysis to uncover service equity gaps.
- Housing stability tracking to measure collective impact.

## Installation

Follow these steps to set up the project locally:

```bash
git clone https://github.com/WallGuan/Capstone_TB.git
cd Capstone_TB
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

1.  Place the raw data files into the `data-dashboard/data/` directory.
2.  Run the `data_preprocessing.ipynb` notebook located in the `data-dashboard/` directory to process the data.
3.  The preprocessed data will be saved (ensure the notebook saves it to a designated location, e.g., `data-dashboard/processed_data/`). *Note: The original README mentioned `data-dashboard/dat` which might be a typo or require clarification in the notebook.*

### Running the Dashboard

1.  Navigate to the data dashboard directory:
    ```bash
    cd data-dashboard
    ```
2.  Run the Streamlit application:
    ```bash
    streamlit run dashboard.py
    ```

### Publishing the Dashboard (Simple Method)

To temporarily share the dashboard online:

```bash
streamlit run dashboard.py --server.port 8501
ssh -T -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -R 80:localhost:8501 localhost.run
```
*Note: This requires the command to be run each time you want to make the dashboard accessible.*

### Running the Service Optimization Analysis

1.  Ensure the necessary raw data is present in the `service-optimization/` directory.
2.  Run the `simplified_service_list_final.ipynb` notebook located within the `service-optimization/` directory to perform the analysis.

## Project Structure

```
Capstone_TB/
├── data-dashboard/
│   ├── dat/                 # Contains data used by the dashboard
│   ├── dashboard.py
│   └── data_preprocessing.ipynb
├── service-optimization/
│   ├── living_situation_analysis.R
│   ├── service_path.ipynb
│   ├── simplified_service_list_final.ipynb
│   └── simplified_service_path_final.ipynb
├── .gitignore
├── entity_relationship_diagram.png
├── README.md
├── requirements.txt
├── techbridge_presentation_slides.pdf
└── techbridge_white_paper.pdf
```


