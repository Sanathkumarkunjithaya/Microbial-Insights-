# Microbial Insights: Leveraging Soil Health For Predictive Crop Analysis (Arecanut)

A comprehensive machine learning project designed to predict crop yield for Arecanut based on a variety of soil health indicators, microbial activity, nutrient levels, and environmental factors. This project focuses on analyzing and leveraging soil microbial data to enhance agricultural productivity specifically for Arecanut crops.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Overview
**Microbial Insights** aims to assist farmers and researchers by providing an automated system to analyze soil health indicators and predict yield accurately for Arecanut. The system is built upon various machine learning models and utilizes ensemble learning for better accuracy. Soil microbial health plays a significant role in determining the nutrient availability and overall health of the Arecanut crop, making it a vital component of this analysis.

## Features
- **Predictive Analysis**: Predicts Arecanut yield based on soil health, microbial activity, and nutrient composition.
- **Soil Health Insights**: Offers insights into the soil’s microbial ecosystem and its impact on Arecanut yield.
- **Ensemble Model**: Combines the strengths of Random Forest and XGBoost for better prediction accuracy.
- **User-Friendly Interface**: Includes a [Streamlit](https://streamlit.io/) app for easy input and result visualization.

## File Structure
```plaintext
Microbial-Insights/
├── README.md         # Project documentation
├── app/              # Application files and assets
│   ├── app.py        # Main application script
│   ├── app1.py       # Secondary(Good visualization and graphics) application script
│   ├── arecimg.jpg   # Visualization image used in the app
├── notebooks/        # Jupyter notebooks for model development and analysis
│   ├── rf_model.ipynb    # Random Forest model development
│   ├── stens.ipynb       # Ensemble model (Random Forest + XGBoost)
│   ├── xgb_model.ipynb   # XGBoost model development
├── scripts/          # Python scripts for reusable code
│   └── stens.py       # Script for ensemble model implementation
├── models/           # Placeholder for saved model files
└── __pycache__/      # Python cache files (not required for version control)
```

## Dataset
The dataset used in this project includes features such as:
- Soil pH, Nitrogen, Phosphorus, Potassium levels
- Organic Matter, Temperature
- Beneficial and Harmful Microbes (CFU/g)
- Microbial Biomass, Soil Enzyme Activity
- Disease and Nutrient Deficiency indicators specific to Arecanut
- Weather conditions and Arecanut Yield (kg/palm)

For privacy and legal reasons, the dataset may not be provided here. However, you may use your dataset with similar structure.

## Model Architecture
This project leverages an ensemble model of **Random Forest** and **XGBoost** to predict Arecanut yield. Both models are trained on preprocessed data and tuned for accuracy. The final prediction is obtained by averaging the predictions of both models. 

### Key Model Components
- **Random Forest**: Handles non-linear relationships and provides robust predictions for complex data.
- **XGBoost**: Known for its gradient boosting approach, providing high performance in predicting Arecanut yield based on soil health metrics.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Required Python packages (listed in `requirements.txt`)

### Clone the Repository
```bash
git clone https://github.com/Sanathkumarkunjithaya/Microbial-Insights-.git
  ```
## Usage

### Run the Application
- **Navigate to the app/ directory:** cd app
- **Run the application** streamlit run app.py (or streamlit run app1.py)

### Explore the Notebooks
- **Open Jupyter Notebook:** `jupyter notebook`
- **Navigate to the notebooks/ directory and explore:**
  - **rf_model.ipynb** for Random Forest
  - **xgb_model.ipynb** for XGBoost
  - **stens.ipynb** for the ensemble model

## Results

The model achieved the following performance metrics:

- **Mean Squared Error (MSE):** 0.10
- **Mean Absolute Error (MAE):** 0.26
- **R² Score:** 0.95
  
These metrics demonstrate the accuracy and reliability of the ensemble model.

