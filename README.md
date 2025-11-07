

# Comparative Volatility Modeling: GARCH(1,1) vs Stochastic Volatility (SV)

**Erods Institute Quant Finance Bootcamp – Fall 2025 Final Project**

## Overview
This project compares two volatility modeling approaches—**GARCH(1,1)** and **Stochastic Volatility (SV)** with a particle filter—across three asset classes and multiple market shock periods. The goal is to evaluate their forecasting accuracy, robustness, and behavior under structural breaks.

## Assets Analyzed
- **FX:** USD/INR  
- **Bond:** BNDX (International Bond ETF)  
- **Commodity:** Crude Oil  

## Shock Windows
- COVID-19  
- Inflation/Energy Crisis  
- Oil Price Spike  

## Methodology
1. **Data Preparation:**  
   - Historical prices are fetched, cleaned, and transformed into returns.  
   - Two training windows:  
     - Pre-COVID (6 years)  
     - COVID (3 years)  

2. **Modeling:**  
   - Fit **GARCH(1,1)**  and **SV** (Normal) models.  
   - Estimate parameters and appropraite distribtion per model via **negative log-likelihood**.  
   - Recalibrate parameters mid-COVID to capture regime change.  

3. **Evaluation:**  
   - Rolling-window volatility plots.  
   - Likelihood metrics and forecast performance comparison.  

## Project Structure

- notebooks/
    - 01_data_fetch.ipynb       
      -  Fetch and save raw/processed data
    - 02_exploration.ipynb     
      -  Explore/plot data (price, returns volatility)
    - 03_split_data.ipynb      
      - Call function and perform rain/test split 
    - 04_modeling.ipynb        
      - Fit GARCH(1,1) and SV model and compute vol
    - 05_analysis.ipynb         
      - Evaluate model performance, plot results
    - 06_metrics.ipynb          
      - compare models using and plotting some metrics
    -  data/              
      - Store CSV files
        - splits/
      - train and test CSVs
    - plot/ 
       - store plots
    - models/
       - model train  CSVs for both pre-covid and covid  CSVs
       - optimal parameters/fit from NLL computaiton
    - results/
        - garch/
           - test CSV
        - sv/
           - test CSV
        - merged_test/
           - merged test CSV


- modules/
    - data_utils.py          
       - Data handling functions   
    - sv _utils.py     
       - SV model + fitting
    - garch_utils.py   
       - GARCH model + fitting         
    - plot_utils.py   
       - Visualization utilities




