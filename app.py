import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import sqlite3

# ==========================================
# 0. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Cyclone Impact & Vegetation Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the dark aesthetic of your screenshots
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        h1, h2, h3 {color: #e0e0e0;} 
        .stButton>button {width: 100%; border-radius: 4px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA GENERATION
# ==========================================
@st.cache_data
def load_data():
    # Names taken exactly from your screenshot
    cyclone_names = [
        'AMPHAN', 'ASANI', 'BULBUL', 'DANA', 'FANI', 'GULAB', 
        'HIBARU', 'HUDHUD', 'PHAILIN', 'SuperCyclone', 'YAAS', 'cyclone.04B'
    ]
    
    data = []
    np.random.seed(42)
    
    for name in cyclone_names:
        # Generate realistic track data
        steps = 40
        # Start somewhere in Bay of Bengal
        start_lat = np.random.uniform(10, 16)
        start_lon = np.random.uniform(84, 92)
        
        # Movement vector (North-West towards Odisha)
        lat_step = np.random.uniform(0.15, 0.35)
        lon_step = np.random.uniform(-0.25, -0.05)
        
        current_wind = np.random.randint(45, 65)
        
        for i in range(steps):
            # Update position
            curr_lat = start_lat + (i * lat_step) + np.random.normal(0, 0.05)
            curr_lon = start_lon + (i * lon_step) + np.random.normal(0, 0.05)
            
            # Simulate Intensity (Bell curve)
            if i < steps/2:
                current_wind += np.random.randint(0, 12)
            else:
                current_wind -= np.random.randint(0, 12)
            
            current_wind = max(30, min(current_wind, 260))
            
            # Grade Classification
            if current_wind < 50: grade = 'D'
            elif current_wind < 60: grade = 'DD'
            elif current_wind < 90: grade = 'CS'
            elif current_wind < 120: grade = 'SCS'
            elif current_wind < 170: grade = 'VSCS'
            elif current_wind < 220: grade = 'ESCS'
            else: grade = 'SuCS' # Super Cyclonic Storm
            
            # Pressure (Inverse to wind)
            pressure = 1010 - (current_wind * 0.22) + np.random.normal(0, 2)
            
            data.append({
                'Name': name,
                'Lat': curr_lat,
                'Lon': curr_lon,
                'Max_Wind_Speed': int(current_wind),
                'Pressure': int(pressure),
                'Grade': grade
            })
            
    df = pd.DataFrame(data)
    df['Latitude'] = df['Lat']
    df['Longitude'] = df['Lon']
    return df

df_clean = load_data()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def run_sql_query(query, df):
    """Executes SQL query on the dataframe."""
    conn = sqlite3.connect(':memory:')
    df.to_sql('cyclones', conn, index=False, if_exists='replace')
    try:
        result = pd.read_sql_query(query, conn)
        return result, None
    except Exception as e:
        return None, str(e)

def plot_stats_overview(df):
    """Replicates the 4-panel grid from the 'Stats' screenshot."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 1. Trajectory Path
    sns.scatterplot(data=df, x='Lon', y='Lat', hue='Name', palette='viridis', ax=axes[0,0], s=40, legend=False)
    for name in df['Name'].unique():
        storm = df[df['Name'] == name]
        axes[0,0].plot(storm['Lon'], storm['Lat'], alpha=0.4, linewidth=1, color='white')
    axes[0,0].set_title("📍 Trajectory Path")
    axes[0,0].set_xlabel("Longitude"); axes[0,0].set_ylabel("Latitude")

    # 2. Intensity Grade Distribution
    grade_order = ['D', 'DD', 'CS', 'SCS', 'VSCS', 'ESCS', 'SuCS']
    existing_grades = [g for g in grade_order if g in df['Grade'].unique()]
    sns.countplot(y='Grade', data=df, order=existing_grades, palette='magma', ax=axes[0,1])
    axes[0,1].set_title("📊 Intensity Grade Distribution")

    # 3. Pressure vs Wind
    sns.regplot(x='Pressure', y='Max_Wind_Speed', data=df, scatter_kws={'alpha':0.6, 'color':'#00ccff'}, line_kws={'color':'red'}, ax=axes[1,0])
    axes[1,0].set_title("📉 Pressure vs Wind Relationship")

    # 4. Wind Speed Distribution
    sns.histplot(df['Max_Wind_Speed'], kde=True, color='skyblue', ax=axes[1,1])
    axes[1,1].set_title("🌬️ Wind Speed Distribution")

    plt.tight_layout()
    st.pyplot(fig)

def plot_wind_by_grade(df):
    """Replicates the Box Plot screenshot."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grade_order = ['D', 'DD', 'CS', 'SCS', 'VSCS', 'ESCS', 'SuCS']
    existing_grades = [g for g in grade_order if g in df['Grade'].unique()]
    
    sns.boxplot(x='Grade', y='Max_Wind_Speed', data=df, order=existing_grades, palette='cool', ax=ax)
    ax.set_title("Wind Speed by Grade")
    st.pyplot(fig)

def plot_density(df):
    """Replicates the Geospatial Density screenshot."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Kernel Density Estimate (The red blobs)
    sns.kdeplot(data=df, x='Lon', y='Lat', fill=True, cmap='Reds', alpha=0.5, ax=ax, thresh=0.05)
    # Scatter overlay
    sns.scatterplot(data=df, x='Lon', y='Lat', hue='Max_Wind_Speed', palette='viridis', ax=ax, s=30)
    
    ax.set_title("Geospatial Density")
    st.pyplot(fig)

def plot_ndvi_simulation(df, cyclone_name):
    """Replicates the specific green-to-red side-by-side map."""
    lat_min, lat_max = 17.5, 22.5
    lon_min, lon_max = 81.5, 87.5
    grid_res = 0.05
    
    lats = np.arange(lat_min, lat_max, grid_res)
    lons = np.arange(lon_min, lon_max, grid_res)
    xx, yy = np.meshgrid(lons, lats)
    
    # Before State
    np.random.seed(42)
    ndvi_before = 0.7 + np.random.normal(0, 0.05, xx.shape)
    ndvi_before = np.clip(ndvi_before, 0.0, 0.9)
    
    # After State
    ndvi_after = ndvi_before.copy()
    target_storms = df if cyclone_name == 'All' else df[df['Name'] == cyclone_name]
    
    for _, row in target_storms.iterrows():
        c_lat, c_lon = row['Lat'], row['Lon']
        wind = row['Max_Wind_Speed']
        radius = 0.5 + (wind / 300)
        severity = (wind / 250) * 0.8
        dist = np.sqrt((xx - c_lon)**2 + (yy - c_lat)**2)
        damage_mask = np.exp(-0.5 * (dist / (radius/2))**2) * severity
        ndvi_after -= damage_mask

    ndvi_after = np.clip(ndvi_after, 0.05, 0.9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Before
    axes[0].imshow(ndvi_before, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[0].set_title("PRE-CYCLONE VEGETATION (Simulated)", fontweight='bold')
    axes[0].set_ylabel("Latitude")
    
    # Right: After
    im = axes[1].imshow(ndvi_after, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[1].set_title(f"POST-EVENT IMPACT: {cyclone_name}", fontweight='bold')
    
    # Tracks
    if not target_storms.empty:
        if cyclone_name == 'All':
            axes[1].scatter(target_storms['Lon'], target_storms['Lat'], c='black', s=1, alpha=0.3)
        else:
            axes[1].plot(target_storms['Lon'], target_storms['Lat'], 'k--', alpha=0.6)

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.03)
    cbar.set_label('NDVI (Greenness Index)')
    st.pyplot(fig)

def plot_odisha_map_folium(df):
    """Replicates the Odisha Map View."""
    odisha_lat_min, odisha_lat_max = 17.5, 22.5
    odisha_lon_min, odisha_lon_max = 81.5, 87.5
    
    # Filter for map display
    odisha_storms = df[
        (df['Lat'] >= odisha_lat_min) & (df['Lat'] <= odisha_lat_max) & 
        (df['Lon'] >= odisha_lon_min) & (df['Lon'] <= odisha_lon_max)
    ]['Name'].unique()
    
    m = folium.Map(location=[20.2, 84.4], zoom_start=7, tiles='CartoDB positron')
    
    folium.Rectangle(
        bounds=[[odisha_lat_min, odisha_lon_min], [odisha_lat_max, odisha_lon_max]],
        color="black", weight=2, fill=False, dash_array='5, 5', tooltip="Odisha Region"
    ).add_to(m)
    
    for name in odisha_storms:
        storm_data = df[df['Name'] == name]
        points = list(zip(storm_data['Lat'], storm_data['Lon']))
        folium.PolyLine(points, color="blue", weight=2, opacity=0.5).add_to(m)
        
        for _, row in storm_data.iterrows():
            if (odisha_lat_min <= row['Lat'] <= odisha_lat_max) and (odisha_lon_min <= row['Lon'] <= odisha_lon_max):
                color = 'green'
                if row['Max_Wind_Speed'] > 150: color = 'red'
                elif row['Max_Wind_Speed'] > 100: color = 'orange'
                
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']], radius=4,
                    color=color, fill=True, fill_color=color, popup=f"{row['Name']}"
                ).add_to(m)
    return m

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.title("Controls")

# Navigation Mode
nav_mode = st.sidebar.radio("Navigation:", ["Dashboard Overview", "NDVI Analysis", "Odisha Map"])

st.sidebar.markdown("---")

# Filters
unique_names = ['All'] + sorted(df_clean['Name'].unique().tolist())
selected_storm = st.sidebar.selectbox("Select Storm:", unique_names)

min_wind = int(df_clean['Max_Wind_Speed'].min())
max_wind = int(df_clean['Max_Wind_Speed'].max())
wind_range = st.sidebar.slider("Wind Range:", min_wind, max_wind, (min_wind, max_wind))

# Filter Data
mask = (df_clean['Max_Wind_Speed'] >= wind_range[0]) & (df_clean['Max_Wind_Speed'] <= wind_range[1])
filtered_df = df_clean[mask]
if selected_storm != 'All':
    filtered_df = filtered_df[filtered_df['Name'] == selected_storm]

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
st.title("🌪️ Cyclone Impact & Vegetation Dashboard")

if nav_mode == "Dashboard Overview":
    # The Exact Tabs from your Screenshot
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Stats", "Density", "SQL & Viz"])
    
    with tab1:
        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cyclones in View", filtered_df['Name'].nunique())
        col2.metric("Max Wind Speed", f"{filtered_df['Max_Wind_Speed'].max()} km/h")
        col3.metric("Data Points", len(filtered_df))
        st.dataframe(filtered_df.head(50))
        
    with tab2:
        if not filtered_df.empty:
            plot_stats_overview(filtered_df)
            st.markdown("---")
            plot_wind_by_grade(filtered_df)
    
    with tab3:
        if not filtered_df.empty:
            plot_density(filtered_df)

    with tab4:
        st.subheader("🔍 SQL & Custom Graph Builder")
        st.markdown("Run SQL queries directly on the dataset (Table name: `cyclones`).")
        
        default_query = "SELECT * FROM cyclones LIMIT 10"
        query_input = st.text_area("SQL Query:", value=default_query, height=100)
        
        if st.button("Run Query", type="primary"):
            result, error = run_sql_query(query_input, df_clean)
            if error:
                st.error(f"SQL Error: {error}")
            else:
                st.dataframe(result)

elif nav_mode == "NDVI Analysis":
    st.subheader(f"📉 Vegetation Damage Simulation: {selected_storm}")
    if not filtered_df.empty:
        plot_ndvi_simulation(df_clean, selected_storm)
    else:
        st.warning("No data matches filters.")

elif nav_mode == "Odisha Map":
    st.subheader("🗺️ Geographic Analysis: Odisha Region Focus")
    if not filtered_df.empty:
        m = plot_odisha_map_folium(filtered_df)
        st_folium(m, width=1200, height=500)
    else:
        st.warning("No data matches filters.")
