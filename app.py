import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

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
        
        # Color Mapping for Pydeck (R, G, B, A)
        # Red for high wind, Green for low wind
        def get_color(wind):
            if wind > 150: return [255, 0, 0, 160]   # Red
            elif wind > 100: return [255, 165, 0, 160] # Orange
            else: return [0, 255, 0, 160]              # Green
            
        df['color'] = df['Max_Wind_Speed'].apply(get_color)
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

def render_pydeck_map(df):
    """Generates the interactive map using Pydeck (No Folium)."""
    
    # Define the View State (Focused on Odisha)
    view_state = pdk.ViewState(
        latitude=20.5,
        longitude=84.5,
        zoom=6,
        pitch=0,
    )

    # Layer 1: Scatterplot for Storm Points
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[Lon, Lat]',
        get_color='color',
        get_radius=20000,  # Radius in meters
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=10,
    )

    # Tooltip Configuration
    tooltip = {
        "html": "<b>Storm:</b> {Name} <br/> <b>Wind:</b> {Max_Wind_Speed} km/h",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    ))

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
    st.subheader("Odisha Region Cyclone Tracks (Pydeck)")
    st.markdown("🔴 **Red:** Extreme Wind (>150 km/h) | 🟠 **Orange:** Severe | 🟢 **Green:** Moderate")
    render_pydeck_map(df_filtered)

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
