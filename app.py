"""
NISR 2026 Big Data Hackathon — Track 1: Agricultural Productivity
Interactive Streamlit Dashboard & Policy Decision Support System

Author: DamourDev / NISR Hackathon Team
Refactored: Hackathon Finals Release (Theme, Visual Hierarchy, Professional UI/UX)
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration and theme setup
st.set_page_config(
    page_title="NISR Agri-Intelligence Engine | Rwanda 2026",
    page_icon="🇷🇼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional Dark-Green Agricultural Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Institutional Header Card */
    .main-header {
        background: linear-gradient(135deg, #091f14 0%, #133923 50%, #1e5233 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(46, 125, 50, 0.35);
    }
    .main-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #c8e6c9;
        font-weight: 400;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255, 215, 0, 0.18);
        color: #ffd54f;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 215, 0, 0.4);
    }
    
    /* Dark Theme Native Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #1e222b !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 16px 18px !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4) !important;
        border-color: rgba(46, 125, 50, 0.5) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    
    /* Sleek Institutional Status Badges */
    .badge-urgent {
        background: rgba(220, 38, 38, 0.18);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.4px;
        display: inline-block;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }
    .badge-high {
        background: rgba(217, 119, 6, 0.18);
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.4px;
        display: inline-block;
        border: 1px solid rgba(251, 191, 36, 0.35);
    }
    .badge-moderate {
        background: rgba(22, 163, 74, 0.18);
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.4px;
        display: inline-block;
        border: 1px solid rgba(74, 222, 128, 0.35);
    }
    
    /* Dark Surface Container Cards */
    .surface-card {
        background: #1e222b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease;
    }
    .surface-card:hover {
        border-color: rgba(46, 125, 50, 0.4);
    }
    
    /* Policy Action Cards */
    .policy-box {
        background: #181c24;
        border-left: 5px solid #2e7d32;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .policy-box-urgent {
        border-left-color: #dc2626;
    }
    .policy-box-high {
        border-left-color: #d97706;
    }
    
    /* Methodology Box */
    .method-box {
        background: #151820;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Data and model loaders
@st.cache_data
def load_master_data():
    """Loads Phase 4 prediction master dataset."""
    path = os.path.join("data", "df_master_phase4_predictions.csv")
    if not os.path.exists(path):
        path = os.path.join("data", "df_master_phase3.csv")
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_ml_model():
    """Loads the champion Gradient Boosting Regressor pipeline."""
    path = os.path.join("models", "yield_predictor_best.pkl")
    if not os.path.exists(path):
        path = os.path.join("models", "yield_predictor_rf.pkl")
    model = joblib.load(path)
    return model

try:
    df_master = load_master_data()
    DATA_LOADED = True
except Exception as e:
    st.error(f"Error loading project artifacts: {e}")
    DATA_LOADED = False


# Sidebar filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/17/Flag_of_Rwanda.svg", width=50)
    st.markdown("### NISR Agri-Engine")
    st.caption("**Track 1: Agricultural Productivity & Food Security**")
    st.markdown("---")
    
    st.markdown("#### Filter Scope")
    
    all_districts = sorted(df_master['DISTRICT'].unique().tolist()) if DATA_LOADED else []
    all_crops = sorted(df_master['CROP'].unique().tolist()) if DATA_LOADED else []
    
    if "selected_districts" not in st.session_state:
        st.session_state.selected_districts = ["BURERA", "GASABO", "GATSIBO", "HUYE", "MUSANZE", "RUSIZI"]
    
    if "selected_crops" not in st.session_state:
        st.session_state.selected_crops = ["Irish Potato", "Maize", "Beans", "Cassava", "Rice", "Amaranth"]
        
    def reset_filters():
        st.session_state.selected_districts = ["BURERA", "GASABO", "GATSIBO", "HUYE", "MUSANZE", "RUSIZI"]
        st.session_state.selected_crops = ["Irish Potato", "Maize", "Beans", "Cassava", "Rice", "Amaranth"]
        
    selected_districts = st.multiselect(
        "Select Target Districts:",
        options=all_districts,
        key="selected_districts",
        help="Filter analysis across Rwanda's 6 strategic districts."
    )
    
    selected_crops = st.multiselect(
        "Select Target Crops:",
        options=all_crops,
        key="selected_crops",
        help="Filter analysis across Rwanda's 6 priority crops."
    )
    
    st.button("Reset Filters", on_click=reset_filters, width="stretch")
    
    # Active Scope Indicator
    filtered_count = len(df_master[
        (df_master['DISTRICT'].isin(selected_districts)) &
        (df_master['CROP'].isin(selected_crops))
    ]) if DATA_LOADED else 0
    st.caption(f"Filtered Scope: **{filtered_count} of 36** District-Crop Pairs")
    
    # Bottom Collapsible Project Metadata
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    with st.expander("About this Project", expanded=False):
        st.markdown("""
        **NISR Big Data Hackathon 2026**  
        *Track 1: Agricultural Productivity & Food Security*  
        
        - **Target Districts (6):** `BURERA`, `GATSIBO`, `MUSANZE`, `HUYE`, `RUSIZI`, `GASABO`
        - **Priority Crops (6):** `Irish Potato`, `Maize`, `Beans`, `Cassava`, `Rice`, `Amaranth`
        - **Champion Model:** Gradient Boosting Regressor
        - **Validation:** Leave-One-Out Cross-Validation (LOOCV R² = 0.8116)
        - **Data Fusion:** SAS 2025 + AHS 2024 + EICV7 + Open-Meteo Climate + Sentinel-2 GEE NDVI + eSoko Farmgate Prices.
        """)
        st.caption("National Institute of Statistics of Rwanda • Kigali, Rwanda")


# Filtered Master Dataset
if DATA_LOADED:
    df_filtered = df_master[
        (df_master['DISTRICT'].isin(selected_districts)) &
        (df_master['CROP'].isin(selected_crops))
    ].copy()
else:
    df_filtered = pd.DataFrame()


# Header banner
st.markdown("""
<div class="main-header">
    <div class="header-badge">NATIONAL INSTITUTE OF STATISTICS OF RWANDA • BIG DATA HACKATHON 2026</div>
    <h1>Rwanda Agricultural Productivity & Climate Vulnerability Engine</h1>
    <p>Multi-Source Data Fusion of NISR Agricultural Surveys, Open-Meteo Climate Deficits, Sentinel-2 Satellite NDVI, & Machine Learning Yield Gap Modeling</p>
</div>
""", unsafe_allow_html=True)


# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "Yield Gap Simulator",
    "Climate Vulnerability Matrix",
    "Policy & Decision Support",
    "Diagnostic Gallery"
])


# Tab 1: Executive overview and baseline indicators
with tab1:
    st.markdown("### Executive Overview & Baseline Indicator Fusion")
    st.markdown(
        "Consolidated agricultural indicators derived from the **Seasonal Agricultural Survey (SAS 2025)**, "
        "**Agricultural Household Survey (AHS 2024)**, and **EICV7 Living Conditions Survey**."
    )
    
    if len(df_filtered) == 0:
        st.warning("No data matches the selected filters. Please adjust your district and crop filters in the sidebar.")
    else:
        # Top Executive Metric Cards (6 Metrics in Responsive Layout)
        kpi_row1 = st.columns(3)
        kpi_row2 = st.columns(3)
        
        tot_area = df_filtered['Cultivated_Area_Ha'].sum()
        tot_prod = df_filtered['Production_MT'].sum()
        avg_yield = df_filtered['Yield_Kg_Ha'].mean()
        avg_deficit = df_filtered['Water_Deficit_mm'].mean()
        avg_phl = df_filtered['Post_Harvest_Loss_Pct'].mean()
        avg_price = df_filtered['Mean_Farmgate_Price_RWF'].mean()
        
        # PSTA 5 National Productivity Target = 6,000 kg/Ha
        PSTA5_TARGET_YIELD = 6000.0
        yield_delta = avg_yield - PSTA5_TARGET_YIELD
        
        # Aridity Stress Baseline threshold = 250 mm
        ARIDITY_BASELINE = 250.0
        deficit_delta = avg_deficit - ARIDITY_BASELINE
        
        # Post-Harvest Loss Ceiling Target = 2.0%
        PHL_CEILING = 2.0
        phl_delta = avg_phl - PHL_CEILING
        
        with kpi_row1[0]:
            st.metric(
                label="Cultivated Land Area",
                value=f"{tot_area:,.0f} Ha",
                delta=f"{len(selected_districts)} Active Districts",
                delta_color="off",
                help="Total cultivated land area across the selected districts."
            )
        with kpi_row1[1]:
            st.metric(
                label="Total Production Volume",
                value=f"{tot_prod:,.0f} MT",
                delta="Consolidated Harvest Volume",
                delta_color="off",
                help="Consolidated crop harvest production volume."
            )
        with kpi_row1[2]:
            st.metric(
                label="Average Crop Yield",
                value=f"{avg_yield:,.0f} kg/Ha",
                delta=f"{yield_delta:+,.0f} kg/Ha vs. 6,000 PSTA 5 Target",
                delta_color="normal",
                help="Comparison against Rwanda's PSTA 5 national crop productivity target (6,000 kg/ha)."
            )
            
        with kpi_row2[0]:
            st.metric(
                label="Mean Moisture Deficit",
                value=f"{avg_deficit:,.1f} mm",
                delta=f"{deficit_delta:+,.1f} mm vs. 250mm Aridity Threshold",
                delta_color="inverse",
                help="Annual Reference Evapotranspiration (ET0) minus Precipitation. Deficits exceeding 250mm indicate significant moisture stress."
            )
        with kpi_row2[1]:
            st.metric(
                label="Mean Post-Harvest Loss",
                value=f"{avg_phl:.2f}%",
                delta=f"{phl_delta:+.2f}% vs. 2.0% Loss Ceiling",
                delta_color="inverse",
                help="Mean post-harvest crop loss reported in SAS 2025."
            )
        with kpi_row2[2]:
            st.metric(
                label="Mean Farmgate Commodity Price",
                value=f"{avg_price:,.0f} RWF/kg",
                delta="eSoko Market Benchmark",
                delta_color="off",
                help="Average farmgate commodity price recorded via eSoko."
            )
            
        st.divider()
        
        # Row 1: Production vs Yield and Socioeconomic Scatter
        c1, c2 = st.columns([6, 5])
        
        with c1:
            st.markdown("#### Crop Yield Comparison by District (kg/ha)")
            fig_yield = px.bar(
                df_filtered,
                x="DISTRICT",
                y="Yield_Kg_Ha",
                color="CROP",
                barmode="group",
                text_auto=".0f",
                labels={"Yield_Kg_Ha": "Yield (kg/ha)", "DISTRICT": "District", "CROP": "Crop"},
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_yield.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(gridcolor="#334155", title="Yield (kg/ha)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=20)
            )
            fig_yield.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_yield, width="stretch")
            
            # Policy Insight Callout
            st.info(
                "**Key Policy Takeaway:** Substantial yield disparities exist across agro-ecological zones—ranging from over 14,000 kg/ha for Irish Potato in volcanic soils down to under 800 kg/ha for rainfed beans in low-input areas. "
                "Districts with access to irrigation and structured extension significantly stabilize regional crop yields."
            )
            
        with c2:
            st.markdown("#### Socioeconomic Fragility vs. Crop Yield")
            
            # Scatter Plot with OLS Trendline & Bubble Size = Production_MT
            try:
                fig_bubble = px.scatter(
                    df_filtered,
                    x="Poverty_Headcount_Rate_Pct",
                    y="Yield_Kg_Ha",
                    size="Production_MT",
                    color="DISTRICT",
                    hover_name="CROP",
                    trendline="ols",
                    trendline_color_override="#ef4444",
                    hover_data=["Extension_Reach_Pct", "Fertilizer_Pct", "Mean_Farmgate_Price_RWF"],
                    labels={
                        "Poverty_Headcount_Rate_Pct": "Poverty Headcount Rate (%)",
                        "Yield_Kg_Ha": "Yield (kg/ha)",
                        "Production_MT": "Production (MT)",
                        "DISTRICT": "District"
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
            except Exception:
                fig_bubble = px.scatter(
                    df_filtered,
                    x="Poverty_Headcount_Rate_Pct",
                    y="Yield_Kg_Ha",
                    size="Production_MT",
                    color="DISTRICT",
                    hover_name="CROP",
                    hover_data=["Extension_Reach_Pct", "Fertilizer_Pct", "Mean_Farmgate_Price_RWF"],
                    labels={
                        "Poverty_Headcount_Rate_Pct": "Poverty Headcount Rate (%)",
                        "Yield_Kg_Ha": "Yield (kg/ha)",
                        "Production_MT": "Production (MT)",
                        "DISTRICT": "District"
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                if len(df_filtered) > 1 and df_filtered["Poverty_Headcount_Rate_Pct"].std() > 0:
                    x_arr = df_filtered["Poverty_Headcount_Rate_Pct"].values
                    y_arr = df_filtered["Yield_Kg_Ha"].values
                    poly = np.polyfit(x_arr, y_arr, 1)
                    x_lin = np.linspace(x_arr.min(), x_arr.max(), 30)
                    fig_bubble.add_trace(go.Scatter(
                        x=x_lin, y=poly[0]*x_lin + poly[1],
                        mode="lines", name="OLS Trendline",
                        line=dict(color="#ef4444", dash="dash", width=2)
                    ))
                    
            fig_bubble.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#334155", title="Poverty Headcount Rate (%)"),
                yaxis=dict(gridcolor="#334155", title="Yield (kg/ha)"),
                margin=dict(l=10, r=10, t=50, b=20)
            )
            st.plotly_chart(fig_bubble, width="stretch")
            
            # Policy Insight Callout
            st.info(
                "**Executive Insight:** An inverse relationship exists between district poverty headcount and agricultural productivity. "
                "Districts with higher poverty (e.g., Rusizi at 44.2%) face input access constraints and under-capitalization. "
                "Integrated extension programs and targeted Smart Nkunganire input subsidies are essential for improving agricultural livelihoods in high-poverty districts."
            )
            
        st.divider()
        
        # Row 2: Climate & Water Deficit Profile with Conditional Bar Highlighting
        st.markdown("#### District Climate & Water Deficit Profile (2024 Open-Meteo)")
        df_climate = df_master.groupby("DISTRICT")[
            ["Annual_Precipitation_mm", "Total_ET0_mm", "Water_Deficit_mm", "Dry_Days_Count"]
        ].mean().reset_index().sort_values("Water_Deficit_mm", ascending=False)
        
        max_deficit_district = df_climate.iloc[0]["DISTRICT"]
        max_deficit_val = df_climate.iloc[0]["Water_Deficit_mm"]
        
        def get_deficit_color(district, deficit):
            if district == max_deficit_district:
                return "#e74c3c"  # Highlighted Warning Red
            elif deficit > 500:
                return "#e67e22"  # Warning Orange
            else:
                return "#34495e"  # Muted Slate
                
        deficit_bar_colors = [get_deficit_color(d, v) for d, v in zip(df_climate["DISTRICT"], df_climate["Water_Deficit_mm"])]
        
        c_clim1, c_clim2 = st.columns([5, 6])
        
        with c_clim1:
            # Standalone Conditionally Colored Deficit Bar Chart
            fig_def = go.Figure(go.Bar(
                x=df_climate["DISTRICT"],
                y=df_climate["Water_Deficit_mm"],
                marker_color=deficit_bar_colors,
                text=[f"{v:,.1f} mm" for v in df_climate["Water_Deficit_mm"]],
                textposition="auto"
            ))
            fig_def.update_layout(
                title="Annual Moisture Deficit by District (ET0 - Rainfall)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(tickangle=-45),
                yaxis=dict(title="Moisture Deficit (mm)", gridcolor="#334155"),
                margin=dict(l=10, r=10, t=50, b=20)
            )
            fig_def.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_def, width="stretch")
            
        with c_clim2:
            # 3-Metric Grouped Comparison with Legend placed at Bottom
            fig_climate = go.Figure()
            fig_climate.add_trace(go.Bar(
                name="Annual Precipitation",
                x=df_climate["DISTRICT"],
                y=df_climate["Annual_Precipitation_mm"],
                marker_color="#38bdf8"
            ))
            fig_climate.add_trace(go.Bar(
                name="Reference ET0",
                x=df_climate["DISTRICT"],
                y=df_climate["Total_ET0_mm"],
                marker_color="#fbbf24"
            ))
            fig_climate.add_trace(go.Bar(
                name="Moisture Deficit",
                x=df_climate["DISTRICT"],
                y=df_climate["Water_Deficit_mm"],
                marker_color="#f87171"
            ))
            fig_climate.update_layout(
                title="Rainfall vs. Evapotranspiration vs. Moisture Deficit",
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                yaxis=dict(title="Millimeters (mm)", gridcolor="#334155"),
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=50, b=20)
            )
            fig_climate.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_climate, width="stretch")
            
        # Priority Policy Action Callout
        st.warning(
            f"**Priority Policy Action:** **{max_deficit_district}** faces an acute climatic moisture deficit exceeding **{max_deficit_val:,.1f} mm** (evapotranspiration surpasses rainfall by 125%). "
            "Machine learning feature attribution reveals that water deficit is the primary environmental constraint on crop yields. "
            f"Prioritizing hillside water harvesting dams, valley-dam rehabilitation, and small-scale micro-irrigation schemes in {max_deficit_district} will maximize the impact of agricultural sector investments."
        )
        
        st.divider()
        
        # Filtered Master Feature Matrix Expander with Direct CSV Export
        with st.expander("Explore Filtered Master Feature Matrix (36 Rows × 31 Columns)", expanded=False):
            st.dataframe(df_filtered, width="stretch")
            csv_filtered_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Data (CSV)",
                data=csv_filtered_data,
                file_name="df_nisr_filtered_export.csv",
                mime="text/csv",
                width="content",
                help="Exports the current filtered tabular data as CSV."
            )
            
        # Data Methodology & Provenance Expander
        with st.expander("Data Methodology & Provenance", expanded=False):
            st.markdown("""
            #### Multi-Source Data Fusion Architecture
            This application unifies six independent national and global datasets into a consolidated 36-row master feature matrix:
            
            1. **Seasonal Agricultural Survey (SAS 2025 - NISR):**
               - *Metrics Extracted:* Cultivated Area ($Ha$), Harvested Area ($Ha$), Yield ($kg/Ha$), Production ($MT$), Post-Harvest Loss (%), Irrigation (%), Fertilizer Adoption (%).
               - *Tables Parsed:* Tables 13, 16, 19, 24, 27, 43, 46, 67.
               
            2. **Agricultural Household Survey (AHS 2024 - NISR):**
               - *Metrics Extracted:* Extension Service Reach (%), Smart Nkunganire Subsidy Reach (%).
               - *Tables Parsed:* Tables 27 & 28.
               
            3. **EICV7 Living Conditions & Poverty Survey (2024 - NISR):**
               - *Metrics Extracted:* District Poverty Headcount Rate (%).
               - *Tables Parsed:* Main Indicators & Poverty Profile Table A9.23.
               
            4. **Open-Meteo Historical Climate API (2024):**
               - *Metrics Extracted:* Daily Precipitation ($mm$), Mean Temperature (°C), Reference Evapotranspiration ($ET_0\text{ mm}$), Dry Days Count, and Annual Moisture Deficit ($mm$).
               
            5. **Google Earth Engine (Sentinel-2 MSI Harmonized):**
               - *Metrics Extracted:* Surface reflectance normalized difference vegetation index across Season A (Sept 2023 – Feb 2024) and Season B (March 2024 – July 2024).
               
            6. **eSoko Market Prices (MINAGRI / NAEB):**
               - *Metrics Extracted:* Farmgate commodity market prices (`Mean_Farmgate_Price_RWF`) aggregated across target district zones and crops.
            
            *All datasets were harmonized on standardized district keys (`DISTRICT`) and crop keys (`CROP`) using flexible regex synonym matching.*
            """)


# Tab 2: Interactive ML yield gap predictor and command center
with tab2:
    # Load champion ML model on demand (cached after first access)
    try:
        champion_pipeline = load_ml_model()
    except Exception as e:
        champion_pipeline = None
        st.error(f"Could not load ML pipeline: {e}")

    st.markdown("### Predictive Machine Learning & Scenario Simulator")
    st.markdown(
        "Evaluate yield predictions in real time using the champion **Gradient Boosting Regressor** pipeline. "
        "Test hypothetical climatic shocks or policy interventions (irrigation expansion, extension outreach) to observe simulated yield responses."
    )
    
    # Model Benchmarks Summary Banner
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Champion Model", "Gradient Boosting", "LOOCV Cross-Validated")
    with b2:
        st.metric("Coefficient of Determination", "R² = 0.8116", "+12.2% vs Baseline OLS")
    with b3:
        st.metric("Root Mean Squared Error", "1,920.4 kg/ha", "LOOCV RMSE")
    with b4:
        st.metric("Mean Absolute Error", "1,222.4 kg/ha", "LOOCV MAE")
        
    st.divider()
    
    # Command Center Simulator Setup
    st.markdown("#### Scenario Simulator & Command Center")
    st.caption("Select a district and crop, then adjust climate and agronomic levers or execute preset policy scenarios.")
    
    # Preset Scenario Buttons
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_choice = "none"
    with preset_col1:
        if st.button("Reset to 2024 Baseline", width="stretch"):
            preset_choice = "baseline"
    with preset_col2:
        if st.button("Simulate Drought Shock (-30% Rain)", width="stretch"):
            preset_choice = "drought"
    with preset_col3:
        if st.button("PSTA 5 Subsidy Push (+30% Inputs)", width="stretch"):
            preset_choice = "psta5"
            
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1.2, 1.2])
    
    with sim_col1:
        st.markdown("##### Target Scope")
        sim_district = st.selectbox(
            "Target District:",
            all_districts,
            index=all_districts.index("GATSIBO") if "GATSIBO" in all_districts else 0,
            key="sim_dist"
        )
        sim_crop = st.selectbox(
            "Target Crop:",
            all_crops,
            index=all_crops.index("Maize") if "Maize" in all_crops else 0,
            key="sim_crop"
        )
        
        # Retrieve baseline row
        baseline_row = df_master[(df_master['DISTRICT'] == sim_district) & (df_master['CROP'] == sim_crop)]
        if len(baseline_row) > 0:
            b_row = baseline_row.iloc[0].to_dict()
        else:
            b_row = df_master.iloc[0].to_dict()
            
        st.markdown(f"**Historical NISR Yield:** `{b_row['Yield_Kg_Ha']:,.1f} kg/ha`")
        st.markdown(f"**Baseline Prediction:** `{b_row.get('Predicted_Yield_Kg_Ha', b_row['Yield_Kg_Ha']):,.1f} kg/ha`")
        
    # Set default values according to preset
    base_rain = float(b_row['Annual_Precipitation_mm'])
    base_fert = float(b_row['Fertilizer_Pct'])
    base_ext = float(b_row['Extension_Reach_Pct'])
    base_irrig = float(b_row['Irrigation_Pct'])
    base_ndvi = float(b_row['NDVI_Mean_SeasonB'])
    
    if preset_choice == "drought":
        def_rain = max(400.0, base_rain * 0.7)
        def_fert = base_fert
        def_ext = base_ext
        def_irrig = base_irrig
        def_ndvi = max(0.30, base_ndvi - 0.08)
    elif preset_choice == "psta5":
        def_rain = base_rain
        def_fert = min(100.0, base_fert + 30.0)
        def_ext = min(100.0, base_ext + 25.0)
        def_irrig = min(100.0, base_irrig + 30.0)
        def_ndvi = min(0.85, base_ndvi + 0.05)
    else:
        def_rain = base_rain
        def_fert = base_fert
        def_ext = base_ext
        def_irrig = base_irrig
        def_ndvi = base_ndvi
        
    with sim_col2:
        st.markdown("##### Climate & Satellite Levers")
        sim_rain = st.slider(
            "Annual Precipitation (mm):",
            min_value=400.0,
            max_value=2200.0,
            value=def_rain,
            step=25.0,
            help="Simulates rainfall shocks or wetter microclimates."
        )
        sim_ndvi = st.slider(
            "Satellite NDVI (Season B Peak):",
            min_value=0.30,
            max_value=0.85,
            value=def_ndvi,
            step=0.01,
            help="Sentinel-2 canopy vegetation vigor proxy."
        )
        
    with sim_col3:
        st.markdown("##### Agronomic & Policy Levers")
        sim_fert = st.slider(
            "Fertilizer Adoption Rate (%):",
            min_value=0.0,
            max_value=100.0,
            value=def_fert,
            step=1.0,
            help="Simulates Smart Nkunganire subsidy coverage."
        )
        sim_ext = st.slider(
            "Extension Reach Rate (%):",
            min_value=0.0,
            max_value=100.0,
            value=def_ext,
            step=1.0,
            help="Simulates agronomist outreach scaling under PSTA 5."
        )
        sim_irrig = st.slider(
            "Irrigation Coverage (%):",
            min_value=0.0,
            max_value=100.0,
            value=def_irrig,
            step=1.0,
            help="Simulates hillside or marshland irrigation expansion."
        )
        
    # Re-calculate derived features for inference
    sim_et0 = float(b_row['Total_ET0_mm'])
    sim_input = b_row.copy()
    sim_input['Annual_Precipitation_mm'] = sim_rain
    sim_input['Fertilizer_Pct'] = sim_fert
    sim_input['Extension_Reach_Pct'] = sim_ext
    sim_input['Irrigation_Pct'] = sim_irrig
    sim_input['NDVI_Mean_SeasonB'] = sim_ndvi
    sim_input['Water_Deficit_mm'] = max(0.0, sim_et0 - sim_rain)
    sim_input['Moisture_Shortage_Index'] = sim_input['Water_Deficit_mm'] / (sim_rain + 1e-6)
    sim_input['NDVI_Seasonal_Change'] = sim_ndvi - sim_input['NDVI_Mean_SeasonA']
    sim_input['Infrastructure_Gap'] = 100.0 - sim_irrig
    sim_input['Price_Yield_Ratio'] = sim_input['Mean_Farmgate_Price_RWF'] / (b_row['Yield_Kg_Ha'] + 1.0)
    
    sim_df = pd.DataFrame([sim_input])
    
    # Execute Model Prediction
    try:
        live_pred = champion_pipeline.predict(sim_df)[0]
        actual_yield = b_row['Yield_Kg_Ha']
        delta_yield = live_pred - actual_yield
        pct_gain = (delta_yield / (actual_yield + 1e-6)) * 100.0
        
        st.markdown("<br>", unsafe_allow_html=True)
        r_c1, r_c2, r_c3 = st.columns(3)
        
        with r_c1:
            st.metric(
                label="Baseline Historical Yield",
                value=f"{actual_yield:,.1f} kg/ha",
                delta="NISR 2025 Survey Baseline",
                delta_color="off"
            )
        with r_c2:
            st.metric(
                label="Simulated Predicted Yield",
                value=f"{live_pred:,.1f} kg/ha",
                delta="Live Model Inference",
                delta_color="off"
            )
        with r_c3:
            sign = "+" if delta_yield >= 0 else ""
            st.metric(
                label="Expected Yield Impact",
                value=f"{sign}{delta_yield:,.1f} kg/ha",
                delta=f"{sign}{pct_gain:.1f}% vs. Baseline",
                delta_color="normal" if delta_yield >= 0 else "inverse"
            )
    except Exception as e:
        st.error(f"Inference simulation error: {e}")
        
    st.divider()
    
    # Model Evaluation and Feature Importance
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.markdown("#### Model Evaluation: Actual vs. Predicted Yield")
        fig_eval = px.scatter(
            df_master,
            x="Yield_Kg_Ha",
            y="Predicted_Yield_Kg_Ha",
            color="CROP",
            hover_name="DISTRICT",
            labels={"Yield_Kg_Ha": "Actual Yield (kg/ha)", "Predicted_Yield_Kg_Ha": "Predicted Yield (kg/ha)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        max_val = max(df_master["Yield_Kg_Ha"].max(), df_master["Predicted_Yield_Kg_Ha"].max()) * 1.05
        fig_eval.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Ideal 1:1 Fit"
        ))
        fig_eval.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#334155", title="Actual Yield (kg/ha)"),
            yaxis=dict(gridcolor="#334155", title="Predicted Yield (kg/ha)"),
            margin=dict(l=10, r=10, t=50, b=20)
        )
        st.plotly_chart(fig_eval, width="stretch")
        
        st.info(
            "**Model Reliability Insight:** The champion Gradient Boosting model achieves an R² = 0.8116 under Leave-One-Out Cross-Validation (LOOCV), with a Mean Absolute Error of 1,222.4 kg/ha across all 36 district-crop pairings. "
            "Predictions closely align with the 1:1 parity line across all 6 crops, enabling data-driven agricultural resource allocation."
        )
        
    with p_col2:
        st.markdown("#### Top Drivers of Yield Variance (Feature Importance)")
        try:
            reg = champion_pipeline.named_steps['regressor']
            prep = champion_pipeline.named_steps['preprocessor']
            cat_cols = prep.named_transformers_['cat'].get_feature_names_out(['DISTRICT', 'CROP']).tolist()
            num_cols = prep.named_transformers_['num'].feature_names_in_.tolist()
            all_feat_names = cat_cols + num_cols
            fi_series = pd.Series(reg.feature_importances_, index=all_feat_names).sort_values(ascending=True)
            top_fi = fi_series.tail(12).reset_index()
            top_fi.columns = ["Feature", "Importance"]
            
            top_fi['Feature_Clean'] = top_fi['Feature'].str.replace("CROP_", "Crop: ").str.replace("DISTRICT_", "District: ").str.replace("_", " ")
            
            fig_fi = px.bar(
                top_fi,
                x="Importance",
                y="Feature_Clean",
                orientation="h",
                color="Importance",
                color_continuous_scale="Greens",
                labels={"Importance": "Relative Feature Weight", "Feature_Clean": "Predictor Feature"}
            )
            fig_fi.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#334155"),
                margin=dict(l=10, r=10, t=50, b=20)
            )
            st.plotly_chart(fig_fi, width="stretch")
            
            st.info(
                "**Attribution Insight:** Farmgate commodity prices, post-harvest losses, and water deficits represent the dominant non-crop drivers of yield variation. "
                "Satellite vegetation indices (NDVI) capture in-season biomass trends, validating Earth Observation as a strong predictor."
            )
        except Exception as e:
            st.warning(f"Feature importance display: {e}")


# Tab 3: Climate vulnerability and irrigation priority matrix
with tab3:
    st.markdown("### District Climate Vulnerability & Irrigation Priority Matrix")
    st.markdown(
        "Multi-criteria composite scoring evaluating **Moisture Shortage Index** (40%), "
        "**Socioeconomic Fragility / Poverty Headcount** (30%), and **Irrigation Infrastructure Gap** (30%)."
    )
    
    df_dist_vuln = df_master.groupby("DISTRICT")[
        ["Climate_Vulnerability_Score", "Vulnerability_Rank", "Moisture_Shortage_Index",
         "Poverty_Headcount_Rate_Pct", "Irrigation_Pct", "Extension_Reach_Pct", "Water_Deficit_mm", "Post_Harvest_Loss_Pct"]
    ].first().reset_index().sort_values("Vulnerability_Rank")
    
    def get_priority_badge(rank):
        if rank in [1, 2]:
            return '<span class="badge-urgent">URGENT PRIORITY</span>'
        elif rank in [3, 4]:
            return '<span class="badge-high">HIGH PRIORITY</span>'
        else:
            return '<span class="badge-moderate">MODERATE PRIORITY</span>'
            
    df_dist_vuln['Priority_Badge'] = df_dist_vuln['Vulnerability_Rank'].apply(get_priority_badge)
    
    st.markdown("#### District Climate Risk & Investment Priority Rankings")
    
    v_cols = st.columns(3)
    for idx, row in df_dist_vuln.iterrows():
        col_target = v_cols[idx % 3]
        with col_target:
            st.markdown(f"""
            <div class="surface-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 800; font-size: 17px; color: #4ade80;">#{row['Vulnerability_Rank']} {row['DISTRICT']}</span>
                    {row['Priority_Badge']}
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                    <b>Vulnerability Score:</b> <span style="color:#f87171;font-weight:700;">{row['Climate_Vulnerability_Score']:.4f}</span><br>
                    <b>Annual Moisture Deficit:</b> {row['Water_Deficit_mm']:.1f} mm<br>
                    <b>Poverty Headcount Rate:</b> {row['Poverty_Headcount_Rate_Pct']:.1f}%<br>
                    <b>Irrigation Coverage:</b> {row['Irrigation_Pct']:.1f}%<br>
                    <b>Extension Outreach:</b> {row['Extension_Reach_Pct']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.divider()
    
    # Vulnerability Scatter Bubble Matrix
    st.markdown("#### Multi-Dimensional Risk Matrix: Moisture Deficit vs. Poverty Headcount")
    st.caption("Bubble size indicates Post-Harvest Loss (%) | Color indicates District Priority Group")
    
    def get_priority_cat(rank):
        if rank in [1, 2]: return "Urgent Priority"
        elif rank in [3, 4]: return "High Priority"
        else: return "Moderate Priority"
        
    df_dist_vuln['Priority_Category'] = df_dist_vuln['Vulnerability_Rank'].apply(get_priority_cat)
    
    fig_vuln_matrix = px.scatter(
        df_dist_vuln,
        x="Water_Deficit_mm",
        y="Poverty_Headcount_Rate_Pct",
        size="Post_Harvest_Loss_Pct",
        color="Priority_Category",
        text="DISTRICT",
        hover_data=["Climate_Vulnerability_Score", "Irrigation_Pct", "Extension_Reach_Pct"],
        labels={
            "Water_Deficit_mm": "Annual Water Deficit (mm)",
            "Poverty_Headcount_Rate_Pct": "Poverty Headcount Rate (%)",
            "Priority_Category": "Intervention Level"
        },
        color_discrete_map={
            "Urgent Priority": "#ef4444",
            "High Priority": "#f59e0b",
            "Moderate Priority": "#22c55e"
        }
    )
    fig_vuln_matrix.update_traces(textposition='top center', marker=dict(sizeref=0.1, sizemin=15))
    fig_vuln_matrix.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155", title="Annual Water Deficit (mm) [ET0 - Rainfall]"),
        yaxis=dict(gridcolor="#334155", title="Poverty Headcount Rate (%)"),
        margin=dict(l=10, r=10, t=50, b=20)
    )
    st.plotly_chart(fig_vuln_matrix, width="stretch")
    
    # Policy Action Callout
    st.warning(
        "**Vulnerability Matrix Takeaway:** **Gatsibo** and **Musanze** face critical agricultural vulnerabilities driven by distinct environmental and socioeconomic factors: "
        "Gatsibo suffers from extreme climatic drought stress (917 mm deficit), whereas Musanze suffers from an acute infrastructure bottleneck (only 8.3% irrigation coverage despite leading Irish Potato output). "
        "Policy interventions must be tailored: water-retention infrastructure in Gatsibo vs. drainage modernization and cold storage in Musanze."
    )


# Tab 4: Policy and decision support
with tab4:
    st.markdown("### Policy & Decision Support Portal")
    st.markdown(
        "Actionable recommendations aligned directly with Rwanda's **National Strategy for Transformation 2 (NST2)** "
        "and the **Strategic Plan for Agriculture Transformation (PSTA 5)**."
    )
    
    col_p1, col_p2 = st.columns([7, 5])
    
    with col_p1:
        st.markdown("#### Automated District Strategic Action Items")
        
        st.markdown("""
        <div class="policy-box policy-box-urgent">
            <h4 style="margin:0;color:#f87171;">#1 GATSIBO (Score: 0.5717) — URGENT INTERVENTION</h4>
            <p style="margin:6px 0 0 0;font-size:13px;color:#cbd5e1;line-height:1.6;">
                <b>Core Diagnostic:</b> Severe climatic water deficit (<b>916.8 mm</b>) with elevated evapotranspiration.<br>
                <b>Strategic Action:</b> Fast-track hillside water harvesting dams, valley-dam rehabilitation, and drought-tolerant seed distribution (cassava and hybrid maize). Scale Smart Nkunganire subsidy quotas for micro-drip kits.
            </p>
        </div>
        
        <div class="policy-box policy-box-urgent">
            <h4 style="margin:0;color:#f87171;">#2 MUSANZE (Score: 0.5379) — URGENT INTERVENTION</h4>
            <p style="margin:6px 0 0 0;font-size:13px;color:#cbd5e1;line-height:1.6;">
                <b>Core Diagnostic:</b> Critical irrigation infrastructure bottleneck (only <b>8.3%</b> formal coverage) despite high potato production.<br>
                <b>Strategic Action:</b> Modernize marshland drainage, develop gravity-fed irrigation systems from volcanic slopes, and establish climate-controlled cold storage facilities to reduce post-harvest losses.
            </p>
        </div>
        
        <div class="policy-box policy-box-high">
            <h4 style="margin:0;color:#fbbf24;">#3 GASABO & #4 RUSIZI — HIGH PRIORITY</h4>
            <p style="margin:6px 0 0 0;font-size:13px;color:#cbd5e1;line-height:1.6;">
                <b>Core Diagnostic:</b> Gasabo faces peri-urban water stress (651 mm deficit); Rusizi experiences the highest poverty rate (<b>44.16%</b>).<br>
                <b>Strategic Action:</b> Reinforce social protection linkage to agricultural inputs in Rusizi; implement intensive peri-urban horticulture and vegetable green-belts in Gasabo.
            </p>
        </div>
        
        <div class="policy-box">
            <h4 style="margin:0;color:#4ade80;">#5 BURERA & #6 HUYE — MODERATE OPTIMIZATION</h4>
            <p style="margin:6px 0 0 0;font-size:13px;color:#cbd5e1;line-height:1.6;">
                <b>Core Diagnostic:</b> Favorable precipitation and stable moisture profiles, but extension outreach can be scaled.<br>
                <b>Strategic Action:</b> Expand Smart Nkunganire farmer onboarding, encourage intercropping beans and maize, and upgrade local aggregation market centers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("#### Data Export & Reporting Portal")
        st.markdown("Download clean consolidated analytical datasets for policymaking, presentations, or further econometric modeling.")
        
        # Download filtered dataset
        csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Dataset (CSV)",
            data=csv_filtered,
            file_name="df_nisr_filtered_export.csv",
            mime="text/csv",
            width="stretch",
            help="Exports currently filtered districts and crops."
        )
        
        # Download complete master predictions dataset
        csv_master = df_master.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Feature Matrix & Predictions (CSV)",
            data=csv_master,
            file_name="df_master_phase4_predictions.csv",
            mime="text/csv",
            width="stretch",
            help="Exports the full 36-row dataset with model predictions and vulnerability indices."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### PSTA 5 Strategic Alignment Matrix")
        st.markdown("""
        | Pillar | Strategic Initiative | Engine Application |
        |---|---|---|
        | **Pillar 1** | Sustainable Land & Water Management | Irrigation Priority Index ranking Gatsibo/Musanze |
        | **Pillar 2** | Agronomic Productivity & Inputs | Smart Nkunganire & Fertilizer Yield Modeling |
        | **Pillar 3** | Inclusive Markets & Value Addition | eSoko Farmgate Price & Yield Optimization |
        | **Pillar 4** | Enabling Environment & Data Systems | NISR Big Data fusion with Open-Meteo & GEE |
        """)


# Tab 5: Diagnostic visualization gallery
with tab5:
    st.markdown("### Comprehensive Diagnostic Plot Gallery")
    st.markdown("Review the technical validation artifacts, model performance residuals, and exploratory data analysis.")
    
    data_dir = "data"
    plot_files = sorted([f for f in os.listdir(data_dir) if f.startswith("plot_") and f.endswith(".png")]) if os.path.exists(data_dir) else []
    
    if len(plot_files) == 0:
        st.info("No plot PNGs found in `./data/` directory.")
    else:
        plot_descriptions = {
            "plot_actual_vs_predicted_yield.png": "LOOCV Actual vs. Predicted Yield Regression Fit",
            "plot_feature_importance.png": "Top Predictors of Crop Yield Variance (Gradient Boosting)",
            "plot_irrigation_priority_index.png": "District Climate Vulnerability & Irrigation Priority Ranking",
            "plot_correlation_matrix.png": "Comprehensive Multi-Feature Pearson Correlation Heatmap",
            "plot_vulnerability_drivers.png": "Core Drivers of Agricultural Climate Vulnerability",
            "plot_agro_climatic_correlation.png": "Agro-Climatic Correlation Matrix",
            "plot_ndvi_seasonal_comparison.png": "Sentinel-2 Satellite NDVI Season A vs Season B",
            "plot_rainfall_vs_deficit.png": "District Rainfall vs. Evaporative Moisture Deficit",
            "plot_phl_vs_production.png": "Post-Harvest Loss vs. Production Volume",
            "plot_vulnerability_analysis.png": "District Yield vs. Poverty Headcount & Extension Reach",
            "plot_yield_by_district_crop.png": "Baseline Yield Distribution across Districts and Crops"
        }
        
        selected_plot = st.selectbox(
            "Select Diagnostic Visualization to Inspect:",
            options=plot_files,
            format_func=lambda x: f"{x} — {plot_descriptions.get(x, 'Diagnostic Figure')}"
        )
        
        plot_path = os.path.join(data_dir, selected_plot)
        st.image(plot_path, caption=plot_descriptions.get(selected_plot, selected_plot), width="stretch")
        
        with st.expander("View All 11 Diagnostic Visualizations in Grid"):
            grid_cols = st.columns(2)
            for i, p_file in enumerate(plot_files):
                with grid_cols[i % 2]:
                    st.image(os.path.join(data_dir, p_file), caption=plot_descriptions.get(p_file, p_file), width="stretch")
                    st.markdown("---")

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    "National Institute of Statistics of Rwanda (NISR) • 2026 Big Data Hackathon (Track 1: Agricultural Productivity) • "
    "Developed with Streamlit, Plotly, Scikit-Learn, and Python"
)
