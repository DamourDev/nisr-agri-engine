# NISR 2026 Big Data Hackathon — Track 1: Agricultural Productivity

An end-to-end data fusion and machine learning pipeline combining baseline National Institute of Statistics of Rwanda (NISR) agricultural and living conditions surveys, Open-Meteo historical climate records, Google Earth Engine (Sentinel-2) satellite imagery, and eSoko market prices to predict crop yields and evaluate district-level climate vulnerability across Rwanda.

---

## Project Overview

This project addresses national agricultural productivity and food security challenges in Rwanda by constructing an integrated feature matrix and predictive models across **6 target districts** (`BURERA`, `GATSIBO`, `MUSANZE`, `HUYE`, `RUSIZI`, `GASABO`) and **6 priority crops** (`Irish Potato`, `Maize`, `Beans`, `Cassava`, `Rice`, `Amaranth`).

### Key Capabilities
- **Feature Fusion Matrix:** Combines socioeconomic indicators, post-harvest losses, agricultural extension reach, farmgate prices, weather deficits, and satellite vegetation vigor (NDVI) across 36 district-crop pairs.
- **Predictive Machine Learning:** Trained regression ensemble models predicting crop yields ($kg/ha$) with a champion **Gradient Boosting Regressor** achieving a Leave-One-Out Cross-Validation (LOOCV) $R^2 = 0.8116$.
- **Climate Vulnerability & Irrigation Priority Index:** Multi-criteria composite scoring system ranking districts to guide targeted irrigation infrastructure and agricultural investments.

---

## Repository Structure

```text
NISR-Hackathon/
├── .streamlit/
│   └── config.toml                           # Streamlit Dark Theme & Dashboard Layout
├── .gitignore                                # Excludes venv, checkpoints, and caches
├── README.md                                 # Project Documentation & Executive Summary
├── requirements.txt                          # Python Package Dependencies
├── app.py                                    # Phase 5: Interactive Streamlit Web Application
├── data/
│   ├── sas_2025_crop_production.xlsx        # NISR Seasonal Agricultural Survey 2025
│   ├── nisr_agri_household_survey_2024.xlsx  # NISR Agricultural Household Survey 2024
│   ├── nisr_eicv7_main_indicators_2024.xlsx  # NISR EICV7 Poverty & Living Conditions
│   ├── nisr_eicv7_poverty_profile_2024.xlsx  # NISR EICV7 Poverty Profile
│   ├── tangabid_market_prices.csv            # eSoko Farmgate Commodity Prices
│   ├── rwanda_district_weather_2024.csv      # Open-Meteo 2024 Daily Climate Data
│   ├── rwanda_district_ndvi_2024.csv         # GEE Sentinel-2 Seasonal NDVI Indices
│   ├── df_master_6districts.csv             # Phase 2: Consolidated Survey Baseline (36 rows)
│   ├── df_master_phase3.csv                 # Phase 3: Full Feature Fusion Matrix
│   ├── df_master_phase4_predictions.csv     # Phase 4: Model Predictions & Risk Rankings
│   └── *.png                                # 11 Diagnostic & Vulnerability Visualizations
├── models/
│   ├── yield_predictor_best.pkl              # Champion Gradient Boosting Pipeline (R² = 0.81)
│   └── yield_predictor_rf.pkl                # Random Forest Pipeline
└── notebooks/
    ├── 01_baseline_data_merge.ipynb          # Phase 2: Survey Data Ingestion & Cleaning
    ├── 02_weather_and_gee_pipelines.ipynb    # Phase 3: Weather API & Sentinel-2 GEE Pipelines
    └── 03_ml_yield_models.ipynb              # Phase 4: Feature Engineering & ML Training
```

---

## Pipeline Architecture & Phases

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA INGESTION & FUSION                                 │
├───────────────────┬────────────────────┬──────────────────────┬─────────────────────────┤
│   NISR Surveys    │    eSoko Prices    │    Open-Meteo API    │   Google Earth Engine   │
│ (SAS, AHS, EICV7) │   (Farmgate CSV)   │  (2024 Daily Weather)│ (Sentinel-2 NDVI Rasters│
└─────────┬─────────┴──────────┬─────────┴──────────┬───────────┴────────────┬────────────┘
          │                    │                    │                        │
          └────────────────────┼────────────────────┴────────────────────────┘
                               ▼
                 【 FEATURE FUSION MATRIX 】
            36 District-Crop Combinations (Full Matrix)
                               │
                               ▼
              【 PREDICTIVE MACHINE LEARNING 】
         Champion: Gradient Boosting Regressor (LOOCV R² = 0.8116)
                               │
                               ▼
        【 CLIMATE VULNERABILITY & IRRIGATION PRIORITY INDEX 】
```

### Phase Breakdown

1. **Phase 2 — Baseline Survey Data Extraction & Processing**
   - Dynamically parses complex multi-header Excel tables from NISR SAS 2025, AHS 2024, and EICV7.
   - Standardizes district keys and maps crop synonyms into a 36-row master matrix (6 districts × 6 crops).
   - Integrates eSoko farmgate commodity prices (`Mean_Farmgate_Price_RWF`).
   - Exports `data/df_master_6districts.csv`.

2. **Phase 3 — Climate & Satellite NDVI Data Pipeline**
   - **Open-Meteo API:** Programmatically pulls 2024 daily precipitation, mean temperature, and reference evapotranspiration ($ET_0$) across all target district coordinates.
   - **Google Earth Engine (Sentinel-2):** Extracts seasonal mean NDVI vegetation health scores (`NDVI_Mean_SeasonA`, `NDVI_Mean_SeasonB`).
   - **Feature Engineering:** Derives `Annual_Precipitation_mm`, `Water_Deficit_mm`, and `Dry_Days_Count`.
   - Exports `data/df_master_phase3.csv`.

3. **Phase 4 — Feature Engineering & Machine Learning**
   - Preprocesses numeric features using `StandardScaler` and categoricals with One-Hot Encoding.
   - Benchmarks Linear/Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor, and XGBoost using **Leave-One-Out Cross-Validation (LOOCV)**.
   - Evaluates feature importances quantifying the impact of water deficits, satellite canopy health, poverty rates, and extension service reach on crop yields.
   - Computes district-level **Climate Vulnerability Scores** and **Irrigation Priority Rankings**.
   - Exports serialized models to `./models/` and predictions to `data/df_master_phase4_predictions.csv`.

4. **Phase 5 — Interactive Streamlit Dashboard & Policy Decision Support (`app.py`)**
   - Built a comprehensive 5-tab web dashboard powered by Streamlit and Plotly.
   - **Executive Overview:** Multi-district KPI cards and comparative yield vs. socioeconomic fragility scatter charts.
   - **Interactive ML Yield Gap Predictor:** Live model inference simulator enabling users to dynamically adjust rainfall, fertilizer adoption, extension reach, and satellite NDVI to test real-time yield responses.
   - **Climate Vulnerability Matrix:** District risk cards and multi-dimensional bubble matrix mapping moisture deficits against poverty headcount rates.
   - **Policy Support (NST2 & PSTA 5):** Automated strategic action matrix and direct CSV data export portal.
   - **Diagnostic Gallery:** Interactive viewer for all 11 generated high-resolution visualizations.

---

## Key Results & Machine Learning Performance

### 1. Yield Predictor Benchmark (LOOCV)

Because the dataset consists of 36 aggregated district-crop observations, Leave-One-Out Cross-Validation was used to avoid fold leakage and ensure rigorous generalization assessment:

| Model Pipeline | Cross-Validated $R^2$ | RMSE ($kg/ha$) | MAE ($kg/ha$) | Evaluation Rank |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **0.8116** | **1,920.43** | **1,222.41** | **Champion** |
| **XGBoost Regressor (Tuned)** | 0.7965 | 1,995.43 | 1,458.92 | Runner-Up |
| **Ridge Regression (Tuned, $\alpha=10.0$)** | 0.7307 | 2,295.84 | 1,740.82 | 3rd |
| **Random Forest Regressor** | 0.7088 | 2,387.42 | 1,748.24 | 4th |
| **Linear Regression (OLS)** | 0.6900 | 2,463.24 | 1,832.74 | 5th |

### 2. District Climate Vulnerability & Irrigation Priority Ranking

The composite index balances moisture shortage (40%), poverty headcount rate (30%), and irrigation infrastructure gap (30%):

| Priority Rank | District | Vulnerability Score | Moisture Shortage Index | Irrigation % | Poverty Headcount % | Priority Classification |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **#1** | **GATSIBO** | **0.5717** | 1.249 (916.8 mm deficit) | 44.7% | 18.4% | **URGENT** |
| **#2** | **MUSANZE** | **0.5379** | 0.650 | 8.3% | 21.0% | **URGENT** |
| **#3** | **GASABO** | **0.4587** | 0.793 | 28.6% | 11.1% | **HIGH** |
| **#4** | **RUSIZI** | **0.4090** | 0.006 | 50.0% | 44.2% | **HIGH** |
| **#5** | **BURERA** | **0.3880** | 0.056 | 0.0% | 21.8% | **MODERATE** |
| **#6** | **HUYE** | **0.3543** | 0.250 | 50.9% | 24.2% | **MODERATE** |

---

## Quickstart & Setup Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Installation & Environment Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/nisr-hackathon-2026.git
cd nisr-hackathon-2026

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 3. Google Earth Engine Authentication (Optional)
To pull live satellite data directly from GEE:

```bash
earthengine authenticate
```
*Note: If GEE is uninitialized, the pipeline automatically falls back to an empirical proxy model calibrated for Rwandan ecological zones.*

### 4. Running the Pipeline Notebooks
Execute the notebooks in sequence:

```bash
jupyter notebook notebooks/01_baseline_data_merge.ipynb
jupyter notebook notebooks/02_weather_and_gee_pipelines.ipynb
jupyter notebook notebooks/03_ml_yield_models.ipynb
```

### 5. Launching the Interactive Web Dashboard
Run the Streamlit application to start the interactive decision support portal:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the dashboard.

---

## License & Acknowledgments

Developed for the **National Institute of Statistics of Rwanda (NISR) 2026 Big Data Hackathon** (Track 1: Agricultural Productivity). All survey datasets belong to NISR.