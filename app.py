import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Cyclone Impact Dashboard",
    page_icon="🌪️",
    layout="wide"
)

# ==========================================
# 2. DATA LOADING (Cached for Speed)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cyclone_data.csv')
        # Standardize column names
        if 'Latitude' in df.columns and 'Lat' not in df.columns:
            df['Lat'] = df['Latitude']
        if 'Longitude' in df.columns and 'Lon' not in df.columns:
            df['Lon'] = df['Longitude']
        return df
    except FileNotFoundError:
        return None

df_clean = load_data()

# ==========================================
# 3. SIDEBAR & FILTERS
# ==========================================
st.sidebar.header("🌪️ Filter Controls")

if df_clean is not None:
    # A. Storm Selector
    unique_names = ['All'] + sorted(df_clean['Name'].unique().tolist())
    selected_storm = st.sidebar.selectbox("Select Storm:", unique_names)

    # B. Wind Speed Slider
    min_wind = int(df_clean['Max_Wind_Speed'].min())
    max_wind = int(df_clean['Max_Wind_Speed'].max())
    wind_range = st.sidebar.slider(
        "Wind Speed Range (km/h):",
        min_value=min_wind,
        max_value=max_wind,
        value=(min_wind, max_wind)
    )

    # Filter Data
    mask = (df_clean['Max_Wind_Speed'] >= wind_range[0]) & (df_clean['Max_Wind_Speed'] <= wind_range[1])
    df_filtered = df_clean[mask]

    if selected_storm != 'All':
        df_filtered = df_filtered[df_filtered['Name'] == selected_storm]
else:
    st.error("❌ 'cyclone_data.csv' not found. Please upload your dataset to the repository.")
    st.stop()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def plot_odisha_map(df):
    """Generates the interactive map focused on Odisha."""
    odisha_lat_min, odisha_lat_max = 17.5, 22.5
    odisha_lon_min, odisha_lon_max = 81.5, 87.5
    
    # Base Map
    m = folium.Map(location=[20.5, 84.5], zoom_start=7, tiles='CartoDB positron')
    
    # Odisha Boundary Box
    folium.Rectangle(
        bounds=[[odisha_lat_min, odisha_lon_min], [odisha_lat_max, odisha_lon_max]],
        color="black", weight=2, fill=False, dash_array='5, 5', tooltip="Odisha Region"
    ).add_to(m)

    if df.empty:
        return m

    # Plot Tracks
    if selected_storm == 'All':
        # Simplify view for 'All': Just dots
        for _, row in df.iterrows():
            folium.CircleMarker(
                [row['Lat'], row['Lon']], radius=2, color='blue', opacity=0.5
            ).add_to(m)
    else:
        # Detailed view for single storm
        locations = list(zip(df['Lat'], df['Lon']))
        folium.PolyLine(locations, color="red", weight=3, opacity=0.7).add_to(m)
        
        for _, row in df.iterrows():
            color = 'green'
            if row['Max_Wind_Speed'] > 150: color = 'red'
            elif row['Max_Wind_Speed'] > 100: color = 'orange'
            
            folium.CircleMarker(
                [row['Lat'], row['Lon']], radius=5, color=color, fill=True, 
                popup=f"{row['Name']}: {row['Max_Wind_Speed']} km/h"
            ).add_to(m)

    return m

def plot_ndvi_simulation(df, storm_name):
    """Simulates NDVI change using Matplotlib."""
    # Setup Grid
    lat_min, lat_max = 17.5, 22.5
    lon_min, lon_max = 81.5, 87.5
    grid_res = 0.05
    xx, yy = np.meshgrid(np.arange(lon_min, lon_max, grid_res), np.arange(lat_min, lat_max, grid_res))
    
    # "Before" State
    np.random.seed(42)
    ndvi_before = 0.7 + np.random.normal(0, 0.05, xx.shape)
    ndvi_before = np.clip(ndvi_before, 0.0, 0.9)
    
    # "After" State Calculation
    ndvi_after = ndvi_before.copy()
    
    # Optimization: Subsample points if 'All' is selected to prevent timeout
    points_to_process = df if len(df) < 500 else df.sample(500)

    for _, row in points_to_process.iterrows():
        dist = np.sqrt((xx - row['Lon'])**2 + (yy - row['Lat'])**2)
        radius = 0.5 + (row['Max_Wind_Speed'] / 300)
        severity = (row['Max_Wind_Speed'] / 250) * 0.8
        damage = np.exp(-0.5 * (dist / (radius/2))**2) * severity
        ndvi_after -= damage
    
    ndvi_after = np.clip(ndvi_after, 0.05, 0.9)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(ndvi_before, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[0].set_title("Pre-Event Vegetation")
    
    im2 = axes[1].imshow(ndvi_after, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[1].set_title(f"Post-Event Impact: {storm_name}")
    
    fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04, label="NDVI")
    return fig

# ==========================================
# 5. MAIN APP LAYOUT
# ==========================================

st.title("🌪️ Cyclone Impact & Vegetation Dashboard")
st.markdown("Analyze wind speeds, storm tracks, and simulated vegetation damage in the Odisha region.")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview & Stats", "🗺️ Map Analysis", "📉 NDVI Simulation"])

# --- TAB 1: OVERVIEW ---
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Storms", len(df_filtered['Name'].unique()))
    col2.metric("Max Wind Speed", f"{df_filtered['Max_Wind_Speed'].max()} km/h")
    col3.metric("Data Points", len(df_filtered))
    
    st.subheader("Wind Speed Distribution")
    fig_hist = plt.figure(figsize=(10, 4))
    sns.histplot(df_filtered['Max_Wind_Speed'], kde=True, color='teal')
    plt.xlabel("Wind Speed (km/h)")
    st.pyplot(fig_hist)
    
    st.subheader("Raw Data")
    st.dataframe(df_filtered)

# --- TAB 2: MAP ANALYSIS ---
with tab2:
    st.subheader("Odisha Region Cyclone Tracks")
    map_obj = plot_odisha_map(df_filtered)
    st_folium(map_obj, width=800, height=500)

# --- TAB 3: NDVI SIMULATION ---
with tab3:
    st.subheader("Vegetation Health (NDVI) Impact Simulation")
    st.info("This model simulates vegetation loss based on wind intensity and proximity to the storm track.")
    
    if st.button("Run Simulation"):
        with st.spinner('Running vegetation damage model...'):
            fig_ndvi = plot_ndvi_simulation(df_filtered, selected_storm)
            st.pyplot(fig_ndvi)
            
            st.markdown("""
            **Legend:**
            * 🟩 **Dark Green (0.8+):** Healthy Forest
            * 🟨 **Yellow (0.4-0.6):** Damaged/Stressed
            * 🟥 **Red (<0.3):** Severe Loss / Bare Ground
            """)
