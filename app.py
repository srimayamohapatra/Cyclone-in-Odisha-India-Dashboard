import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import sqlite3

# ==========================================
# 0. PAGE CONFIGURATION & DARK THEME
# ==========================================
st.set_page_config(
    page_title="Cyclone Impact & Vegetation Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to force the Dark/Dashboard look
st.markdown("""
    <style>
        /* Main Container */
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        
        /* Typography */
        h1, h2, h3 {color: #e0e0e0; font-family: 'Sans-serif';}
        p, label {color: #b3b3b3;}
        
        /* Metric Styling */
        div[data-testid="stMetricValue"] {font-size: 24px; color: #00e6e6;}
        
        /* Buttons */
        .stButton>button {
            border-radius: 6px;
            font-weight: bold;
            height: 3em;
            background-color: #262730;
            color: white;
            border: 1px solid #4b4b4b;
        }
        .stButton>button:hover {
            border-color: #00e6e6;
            color: #00e6e6;
        }
        
        /* SQL Input Area */
        .stTextArea textarea {
            font-family: 'Courier New', monospace;
            background-color: #1e1e1e;
            color: #00ff00;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA GENERATION ("WHOLE DATA" 39-259)
# ==========================================
@st.cache_data
def load_data():
    """
    Generates a synthetic dataset mimicking the 'df_clean' from Colab.
    - Specific Names: AMPHAN, FANI, etc.
    - Wind Range: Strictly clamped 39 - 259.
    """
    cyclone_names = [
        'AMPHAN', 'ASANI', 'BULBUL', 'DANA', 'FANI', 'GULAB', 
        'HIBARU', 'HUDHUD', 'PHAILIN', 'SuperCyclone', 'YAAS', 'cyclone.04B'
    ]
    
    data = []
    np.random.seed(42) # Ensure consistency
    
    for name in cyclone_names:
        # Generate track length
        steps = np.random.randint(50, 80)
        
        # Start coordinates (Bay of Bengal)
        start_lat = np.random.uniform(10, 14)
        start_lon = np.random.uniform(85, 92)
        
        # Movement vector (North-West)
        lat_step = np.random.uniform(0.15, 0.35)
        lon_step = np.random.uniform(-0.25, -0.05)
        
        # Initial Wind
        current_wind = np.random.randint(40, 60)
        
        for i in range(steps):
            # Update Position
            curr_lat = start_lat + (i * lat_step) + np.random.normal(0, 0.05)
            curr_lon = start_lon + (i * lon_step) + np.random.normal(0, 0.05)
            
            # Update Intensity (Bell Curve)
            if i < steps // 2:
                current_wind += np.random.randint(2, 8) 
            else:
                current_wind -= np.random.randint(2, 10)
            
            # --- STRICT CLAMPING (39 to 259) ---
            current_wind = max(39, min(current_wind, 259))
                
            # Determine Grade
            if current_wind < 50: grade = 'D'
            elif current_wind < 60: grade = 'DD'
            elif current_wind < 90: grade = 'CS'
            elif current_wind < 120: grade = 'SCS'
            elif current_wind < 170: grade = 'VSCS'
            elif current_wind < 220: grade = 'ESCS'
            else: grade = 'SuCS'
            
            # Pressure (Inverse relation)
            pressure = 1010 - (current_wind * 0.22) + np.random.normal(0, 2)
            
            data.append({
                'Name': name,
                'Lat': curr_lat,
                'Lon': curr_lon,
                'Latitude': curr_lat,  # Duplicate for compatibility
                'Longitude': curr_lon, # Duplicate for compatibility
                'Max_Wind_Speed': int(current_wind),
                'Pressure': int(pressure),
                'Grade': grade
            })
            
    return pd.DataFrame(data)

df_clean = load_data()

# ==========================================
# 2. HELPER FUNCTIONS (PLOTTING & SQL)
# ==========================================

def run_sql_query(query, df):
    """
    Executes SQL queries on an in-memory database version of the dataframe.
    """
    conn = sqlite3.connect(':memory:')
    df.to_sql('cyclones', conn, index=False, if_exists='replace')
    try:
        result = pd.read_sql_query(query, conn)
        return result, None
    except Exception as e:
        return None, str(e)

def plot_ndvi_analysis(df, cyclone_name):
    """
    Simulates Vegetation Damage (Red/Green Heatmap).
    """
    # 1. Define Grid
    lat_min, lat_max, lon_min, lon_max = 17.5, 22.5, 81.5, 87.5
    grid_res = 0.05
    lats = np.arange(lat_min, lat_max, grid_res)
    lons = np.arange(lon_min, lon_max, grid_res)
    xx, yy = np.meshgrid(lons, lats)
    
    # 2. Before State (Healthy)
    np.random.seed(42)
    ndvi_before = 0.7 + np.random.normal(0, 0.05, xx.shape)
    ndvi_before = np.clip(ndvi_before, 0.0, 0.9)
    
    # 3. After State (Damaged)
    ndvi_after = ndvi_before.copy()
    target_df = df if cyclone_name == 'All' else df[df['Name'] == cyclone_name]
    
    if target_df.empty:
        st.warning("No data in simulation zone.")
        return

    for _, row in target_df.iterrows():
        # Distance calculation
        dist = np.sqrt((xx - row['Lon'])**2 + (yy - row['Lat'])**2)
        
        # Gaussian Decay Parameters matching Colab
        radius = 0.5 + (row['Max_Wind_Speed'] / 300)
        severity = (row['Max_Wind_Speed'] / 250) * 0.8
        
        damage = np.exp(-0.5 * (dist / (radius/2))**2) * severity
        ndvi_after -= damage
        
    ndvi_after = np.clip(ndvi_after, 0.05, 0.9)

    # 4. Visualization
    plt.style.use("default") # Ensure colors display correctly
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Plot
    axes[0].imshow(ndvi_before, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[0].set_title("PRE-CYCLONE VEGETATION")
    axes[0].set_ylabel("Latitude")
    
    # Right Plot
    im = axes[1].imshow(ndvi_after, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[1].set_title(f"POST-EVENT IMPACT: {cyclone_name}")
    
    # Tracks Overlay
    if cyclone_name == 'All':
        axes[1].scatter(target_df['Lon'], target_df['Lat'], c='black', s=1, alpha=0.3)
    else:
        axes[1].plot(target_df['Lon'], target_df['Lat'], 'k--', alpha=0.7)
            
    fig.colorbar(im, ax=axes, fraction=0.02)
    st.pyplot(fig)

def plot_density(df):
    """
    Geospatial Density (Red Blobs + Points).
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # KDE (The blobs)
    sns.kdeplot(data=df, x='Lon', y='Lat', fill=True, cmap='Reds', alpha=0.6, thresh=0.05, ax=ax)
    # Scatter (The points)
    sns.scatterplot(data=df, x='Lon', y='Lat', hue='Max_Wind_Speed', palette='viridis', s=25, ax=ax)
    
    ax.set_title("Geospatial Density Analysis")
    st.pyplot(fig)

def plot_odisha_map(df):
    """
    Folium Map focused on Odisha.
    """
    m = folium.Map(location=[20.2, 84.4], zoom_start=7, tiles='CartoDB positron')
    
    # Draw Odisha Box
    folium.Rectangle(
        [[17.5, 81.5], [22.5, 87.5]], 
        color="black", weight=2, fill=False, tooltip="Odisha Region"
    ).add_to(m)
    
    # Plot Points
    for _, row in df.iterrows():
        color = 'red' if row['Max_Wind_Speed'] > 150 else 'orange' if row['Max_Wind_Speed'] > 100 else 'green'
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=3, color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=f"{row['Name']}: {row['Max_Wind_Speed']} km/h"
        ).add_to(m)
        
    st_folium(m, width=1200, height=500)

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Cyclone Controls")

# Navigation (Matches Action Buttons)
nav = st.sidebar.radio("Navigation Mode:", ["Dashboard", "NDVI Analysis", "Odisha Map"])
st.sidebar.markdown("---")

# Filters (Matches Widgets)
# 1. Dropdown
names = ['All'] + sorted(df_clean['Name'].unique().tolist())
sel_name = st.sidebar.selectbox("Select Storm:", names)

# 2. Slider (39 - 259 Range)
min_w, max_w = st.sidebar.slider("Wind Range:", 39, 259, (39, 259))

# Apply Filters
mask = (df_clean['Max_Wind_Speed'] >= min_w) & (df_clean['Max_Wind_Speed'] <= max_w)
df_filt = df_clean[mask]
if sel_name != 'All':
    df_filt = df_filt[df_filt['Name'] == sel_name]

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
st.title("🌪️ Cyclone Impact & Vegetation Dashboard")

if nav == "Dashboard":
    # Tabs replicating Colab Layout
    t1, t2, t3, t4 = st.tabs(["Overview", "Stats", "Density", "SQL & Viz"])
    
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cyclones", df_filt['Name'].nunique())
        c2.metric("Data Points", len(df_filt))
        c3.metric("Max Wind", f"{df_filt['Max_Wind_Speed'].max()} km/h")
        st.dataframe(df_filt.head(100))
        
    with t2:
        if not df_filt.empty:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            
            # Trajectory
            sns.scatterplot(data=df_filt, x='Lon', y='Lat', hue='Name', ax=ax[0,0], legend=False)
            ax[0,0].set_title("Trajectory Path")
            
            # Grades
            order = ['SuCS', 'ESCS', 'VSCS', 'SCS', 'CS', 'DD', 'D']
            present = [x for x in order if x in df_filt['Grade'].unique()]
            sns.countplot(y='Grade', data=df_filt, order=present, palette='magma', ax=ax[0,1])
            ax[0,1].set_title("Intensity Grade")
            
            # Pressure
            sns.regplot(x='Pressure', y='Max_Wind_Speed', data=df_filt, ax=ax[1,0], line_kws={'color':'red'})
            ax[1,0].set_title("Pressure vs Wind")
            
            # Wind Dist
            sns.histplot(df_filt['Max_Wind_Speed'], kde=True, ax=ax[1,1])
            ax[1,1].set_title("Wind Distribution")
            
            st.pyplot(fig)
            
    with t3:
        if not df_filt.empty:
            plot_density(df_filt)
            
    with t4:
        # --- EXACT SQL UPDATE ---
        st.subheader("🔍 SQL & Custom Graph Builder")
        st.markdown("Run SQL queries on `cyclones` table.")
        
        # Default query from prompt
        default_q = "SELECT * FROM cyclones LIMIT 10"
        query = st.text_area("SQL Query:", value=default_q, height=100)
        
        if st.button("Run Query", type="primary"):
            # Execute on whole data (df_clean)
            res, err = run_sql_query(query, df_clean)
            if err:
                st.error(f"SQL Error: {err}")
            else:
                st.success(f"Query returned {len(res)} rows.")
                st.dataframe(res)

elif nav == "NDVI Analysis":
    st.subheader(f"📉 Vegetation Simulation: {sel_name}")
    if not df_filt.empty:
        plot_ndvi_analysis(df_clean, sel_name)
    else:
        st.error("No data matching filters.")

elif nav == "Odisha Map":
    st.subheader("🗺️ Geographic Focus: Odisha")
    if not df_filt.empty:
        plot_odisha_map(df_filt)
    else:
        st.error("No data matching filters.")
