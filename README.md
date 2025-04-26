# Emory University QTM 498R Capstone Project: TechBridge HomeBridger

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org) [![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org) [![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org) [![Seaborn](https://img.shields.io/badge/Seaborn-88d1de?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org) [![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

## Description

TechBridge is an Atlanta-based tech nonprofit organization that provides enterprise-grade software for nonprofit organizations in the areas of hunger relief, homelessness, social justice, and workforce development at below-market cost. This capstone project analyzed internal service data from the HomeBridger platform to explore the value of data sharing in maximizing the collective impact and reach of nonprofit organizations. The goal was to provide actionable insights for optimizing service delivery and improving client outcomes. For more detailed information, please refer to our **[presentation slides](techbridge_presentation_slides.pdf)** and **[white paper](techbridge_white_paper.pdf)**.

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
*   Optimizing the sequence and timing of service delivery.
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
*--server.port xxxx is optional, it is used to specify the port number of the dashboard.*

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
---
This project was developed by Anna Choi, Mary Guan, Wendy Guerrero, Nate Hu, Nick Richards, Katie Shao, and Laura Wang with the guidance and support of Professor Jiwon Kim, TechBridge, and the Emory University QTM Department.

