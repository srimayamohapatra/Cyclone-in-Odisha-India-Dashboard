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

# Custom CSS for Dashboard Look & Green Buttons
st.markdown("""
    <style>
        /* General Dark Theme Overrides */
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3 {color: #e0e0e0; font-family: 'Sans-serif';}
        p, label {color: #b3b3b3;}
        
        /* Metric Styling */
        div[data-testid="stMetricValue"] {font-size: 24px; color: #00e6e6;}
        
        /* Standard Buttons */
        .stButton>button {
            border-radius: 6px; font-weight: bold; height: 3em;
            background-color: #262730; color: white; border: 1px solid #4b4b4b;
        }
        .stButton>button:hover {border-color: #00e6e6; color: #00e6e6;}

        /* GREEN Action Buttons (like Update Graph) */
        div.stButton > button.green-btn {
            background-color: #2ecc71;
            color: white;
            border: none;
        }
        div.stButton > button.green-btn:hover {
            background-color: #27ae60;
            color: white;
        }
        
        /* Text Area Code Font */
        .stTextArea textarea {
            font-family: 'Courier New', monospace;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA GENERATION
# ==========================================
@st.cache_data
def load_data():
    """Generates synthetic dataset mimicking df_clean."""
    cyclone_names = [
        'AMPHAN', 'ASANI', 'BULBUL', 'DANA', 'FANI', 'GULAB', 
        'HIBARU', 'HUDHUD', 'PHAILIN', 'SuperCyclone', 'YAAS', 'cyclone.04B'
    ]
    data = []
    np.random.seed(42)
    
    for name in cyclone_names:
        steps = np.random.randint(50, 80)
        start_lat, start_lon = np.random.uniform(10, 14), np.random.uniform(85, 92)
        lat_step, lon_step = np.random.uniform(0.15, 0.35), np.random.uniform(-0.25, -0.05)
        current_wind = np.random.randint(40, 60)
        
        for i in range(steps):
            curr_lat = start_lat + (i * lat_step) + np.random.normal(0, 0.05)
            curr_lon = start_lon + (i * lon_step) + np.random.normal(0, 0.05)
            
            if i < steps // 2: current_wind += np.random.randint(2, 8) 
            else: current_wind -= np.random.randint(2, 10)
            
            current_wind = max(39, min(current_wind, 259)) # Strict clamp
            
            if current_wind < 50: grade = 'D'
            elif current_wind < 60: grade = 'DD'
            elif current_wind < 90: grade = 'CS'
            elif current_wind < 120: grade = 'SCS'
            elif current_wind < 170: grade = 'VSCS'
            elif current_wind < 220: grade = 'ESCS'
            else: grade = 'SuCS'
            
            pressure = 1010 - (current_wind * 0.22) + np.random.normal(0, 2)
            
            data.append({
                'Name': name,
                'Lat': curr_lat, 'Lon': curr_lon,
                'Latitude': curr_lat, 'Longitude': curr_lon,
                'Max_Wind_Speed': int(current_wind),
                'Pressure': int(pressure),
                'Grade': grade
            })
            
    return pd.DataFrame(data)

df_clean = load_data()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def run_sql_query(query, df):
    conn = sqlite3.connect(':memory:')
    df.to_sql('cyclones', conn, index=False, if_exists='replace')
    try:
        return pd.read_sql_query(query, conn), None
    except Exception as e:
        return None, str(e)

def plot_ndvi_chart_logic(df, cyclone_name):
    """Scatter/Bar chart logic."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5))

    if cyclone_name == 'All':
        sim_df = df.copy()
        sim_df['Est_NDVI_Drop'] = (sim_df['Max_Wind_Speed'] / 250) * 0.4
        sns.scatterplot(data=sim_df, x='Max_Wind_Speed', y='Est_NDVI_Drop', hue='Name', palette='viridis', s=100, ax=ax)
        ax.set_title("🌿 Aggregate Analysis: Wind Speed vs. Est. Vegetation Loss")
    else:
        cyclone_data = df[df['Name'] == cyclone_name]
        if cyclone_data.empty:
            st.warning(f"No data available for {cyclone_name}")
            return
        avg_wind = cyclone_data['Max_Wind_Speed'].max()
        before_ndvi = 0.65
        drop_factor = (avg_wind / 250) * 0.4
        after_ndvi = max(0.1, before_ndvi - drop_factor)
        
        data = {'Period': ['Pre-Cyclone', 'Post-Cyclone'], 'NDVI Value': [before_ndvi, after_ndvi]}
        ndvi_df = pd.DataFrame(data)
        
        sns.barplot(x='Period', y='NDVI Value', data=ndvi_df, palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.axhline(0.3, color='grey', linestyle='--', alpha=0.6, label='Healthy Threshold')
        ax.set_ylim(0, 1.0)
        ax.set_title(f"🌿 Vegetation Health (NDVI) Impact: {cyclone_name}")

    st.pyplot(fig)

def plot_ndvi_heatmap(df, cyclone_name):
    """Spatial Heatmap simulation."""
    # 1. Setup Grid
    lat_min, lat_max, lon_min, lon_max = 17.5, 22.5, 81.5, 87.5
    grid_res = 0.05
    lats = np.arange(lat_min, lat_max, grid_res)
    lons = np.arange(lon_min, lon_max, grid_res)
    xx, yy = np.meshgrid(lons, lats)

    # 2. Before State
    np.random.seed(42)
    ndvi_before = 0.7 + np.random.normal(0, 0.05, xx.shape)
    ndvi_before = np.clip(ndvi_before, 0.0, 0.9)

    # 3. After State
    ndvi_after = ndvi_before.copy()
    target_storms = df if cyclone_name == 'All' else df[df['Name'] == cyclone_name]

    for _, row in target_storms.iterrows():
        dist = np.sqrt((xx - row['Lon'])**2 + (yy - row['Lat'])**2)
        radius = 0.5 + (row['Max_Wind_Speed'] / 300)
        severity = (row['Max_Wind_Speed'] / 250) * 0.8
        damage = np.exp(-0.5 * (dist / (radius/2))**2) * severity
        ndvi_after -= damage

    ndvi_after = np.clip(ndvi_after, 0.05, 0.9)

    # 4. Visualization
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(ndvi_before, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[0].set_title("PRE-CYCLONE VEGETATION (Simulated)")

    im2 = axes[1].imshow(ndvi_after, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[1].set_title(f"POST-EVENT IMPACT: {cyclone_name}")

    if not target_storms.empty:
        if cyclone_name == 'All':
            axes[1].scatter(target_storms['Lon'], target_storms['Lat'], c='black', s=1, alpha=0.3)
        else:
            axes[1].plot(target_storms['Lon'], target_storms['Lat'], 'k--', linewidth=1, alpha=0.7)

    fig.colorbar(im2, ax=axes, fraction=0.02)
    st.pyplot(fig)

def plot_odisha_map_locked(df):
    """Locked View Folium Map."""
    odisha_lat_min, odisha_lat_max = 17.5, 22.5
    odisha_lon_min, odisha_lon_max = 81.5, 87.5

    m = folium.Map(
        location=[20.2, 84.4], zoom_start=7, tiles='CartoDB positron',
        zoom_control=False, scrollWheelZoom=False, doubleClickZoom=False, boxZoom=False, keyboard=False
    )

    folium.Rectangle(
        bounds=[[odisha_lat_min, odisha_lon_min], [odisha_lat_max, odisha_lon_max]],
        color="black", weight=3, fill=False, dash_array='5, 5', tooltip="Odisha Region"
    ).add_to(m)

    odisha_storms = df[
        (df['Lat'] >= odisha_lat_min) & (df['Lat'] <= odisha_lat_max) &
        (df['Lon'] >= odisha_lon_min) & (df['Lon'] <= odisha_lon_max)
    ]['Name'].unique()

    if len(odisha_storms) == 0:
        st.warning("No cyclones found passing through Odisha with current filters.")
        st_folium(m, width=1000, height=500)
        return

    for name in odisha_storms:
        storm_data = df[df['Name'] == name]
        locations = list(zip(storm_data['Lat'], storm_data['Lon']))
        folium.PolyLine(locations=locations, color="blue", weight=2, opacity=0.5).add_to(m)
        
        for _, row in storm_data.iterrows():
            if (odisha_lat_min <= row['Lat'] <= odisha_lat_max) and (odisha_lon_min <= row['Lon'] <= odisha_lon_max):
                fill_color = 'red' if row['Max_Wind_Speed'] > 150 else 'orange' if row['Max_Wind_Speed'] > 100 else 'green'
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']], radius=4, color='black', weight=1,
                    fill=True, fill_color=fill_color, fill_opacity=0.9,
                    popup=f"<b>{row['Name']}</b><br>Wind: {row['Max_Wind_Speed']} km/h"
                ).add_to(m)

    st_folium(m, width=1000, height=600)

def plot_density(df):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='Lon', y='Lat', fill=True, cmap='Reds', alpha=0.6, thresh=0.05, ax=ax)
    sns.scatterplot(data=df, x='Lon', y='Lat', hue='Max_Wind_Speed', palette='viridis', s=25, ax=ax)
    ax.set_title("Geospatial Density Analysis")
    st.pyplot(fig)

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Cyclone Controls")
nav = st.sidebar.radio("Navigation Mode:", ["Dashboard", "NDVI Analysis", "Odisha Map"])
st.sidebar.markdown("---")

names = ['All'] + sorted(df_clean['Name'].unique().tolist())
sel_name = st.sidebar.selectbox("Select Storm:", names)

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
            sns.scatterplot(data=df_filt, x='Lon', y='Lat', hue='Name', ax=ax[0,0], legend=False)
            ax[0,0].set_title("Trajectory Path")
            
            order = ['SuCS', 'ESCS', 'VSCS', 'SCS', 'CS', 'DD', 'D']
            present = [x for x in order if x in df_filt['Grade'].unique()]
            sns.countplot(y='Grade', data=df_filt, order=present, palette='magma', ax=ax[0,1])
            ax[0,1].set_title("Intensity Grade")
            
            sns.regplot(x='Pressure', y='Max_Wind_Speed', data=df_filt, ax=ax[1,0], line_kws={'color':'red'})
            ax[1,0].set_title("Pressure vs Wind")
            
            sns.histplot(df_filt['Max_Wind_Speed'], kde=True, ax=ax[1,1])
            ax[1,1].set_title("Wind Distribution")
            st.pyplot(fig)
            
    with t3:
        if not df_filt.empty:
            plot_density(df_filt)
            
    with t4:
        # --- SQL & VISUALIZATION TAB ---
        st.subheader("🔍 SQL & Custom Graph Builder")
        
        # 1. PRESETS
        presets = {
            "Select a Preset...": "",
            "1. Max Wind by Cyclone Name": "SELECT Name, MAX(Max_Wind_Speed) as Max_Wind FROM cyclones GROUP BY Name ORDER BY Max_Wind DESC",
            "2. Avg Wind & Pressure by Grade": "SELECT Grade, AVG(Max_Wind_Speed) as Avg_Wind, AVG(Pressure) as Avg_Pressure FROM cyclones GROUP BY Grade",
            "3. Pressure vs Wind Data": "SELECT Pressure, Max_Wind_Speed FROM cyclones LIMIT 500",
            "4. Count of Cyclones by Grade": "SELECT Grade, COUNT(DISTINCT Name) as Count FROM cyclones GROUP BY Grade"
        }
        
        selected_preset = st.selectbox("Presets:", list(presets.keys()))
        
        # 2. QUERY INPUT
        # Use session state to handle manual edits vs preset selections
        if "sql_query" not in st.session_state:
            st.session_state.sql_query = presets["Select a Preset..."]
        
        # Update session state only if preset changes and isn't default
        if selected_preset != "Select a Preset...":
            st.session_state.sql_query = presets[selected_preset]

        query = st.text_area("SQL Query:", value=st.session_state.sql_query, height=100)
        
        # 3. RUN BUTTON
        if st.button("Run Query", type="primary"):
            res, err = run_sql_query(query, df_clean) # Query runs on WHOLE DATA (df_clean) as implied by SQL
            if err:
                st.error(f"SQL Error: {err}")
                st.session_state.query_result = None
            else:
                st.session_state.query_result = res
                st.success(f"Query Successful! Rows: {len(res)}. Configure graph below.")

        # 4. GRAPH SETTINGS (Only if data exists)
        if "query_result" in st.session_state and st.session_state.query_result is not None:
            res_df = st.session_state.query_result
            
            st.markdown("### 📊 Graph Settings:")
            
            col_set1, col_set2, col_set3, col_set4 = st.columns([1, 1, 1, 1])
            
            with col_set1:
                viz_type = st.selectbox("Viz Type:", ["Table View", "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"])
            with col_set2:
                x_axis = st.selectbox("X Axis:", res_df.columns)
            with col_set3:
                # Default Y axis to second column if available
                default_y = res_df.columns[1] if len(res_df.columns) > 1 else res_df.columns[0]
                y_axis = st.selectbox("Y Axis:", res_df.columns, index=list(res_df.columns).index(default_y))
            with col_set4:
                st.write("") # Spacer
                st.write("") 
                update_btn = st.button("Update Graph", key="upd_graph")
                
            # 5. VISUALIZATION RENDER
            st.markdown("---")
            
            if viz_type == "Table View":
                st.dataframe(res_df)
                
            else:
                # Prepare Plot
                plt.style.use("default") # White background for charts as requested
                fig, ax = plt.subplots(figsize=(10, 4))
                
                try:
                    if viz_type == "Bar Chart":
                        sns.barplot(data=res_df, x=x_axis, y=y_axis, ax=ax, palette="viridis")
                    elif viz_type == "Line Chart":
                        sns.lineplot(data=res_df, x=x_axis, y=y_axis, ax=ax, marker='o')
                    elif viz_type == "Scatter Plot":
                        sns.scatterplot(data=res_df, x=x_axis, y=y_axis, ax=ax, s=100)
                    elif viz_type == "Pie Chart":
                        # For Pie, X is labels, Y is values
                        ax.pie(res_df[y_axis], labels=res_df[x_axis], autopct='%1.1f%%')
                    
                    ax.set_title(f"{viz_type}: {y_axis} vs {x_axis}")
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Could not plot data. Check axis selection. Error: {e}")


elif nav == "NDVI Analysis":
    st.subheader(f"📉 Vegetation Analysis: {sel_name}")
    if not df_filt.empty:
        st.markdown("### 1. Statistical Impact Assessment")
        plot_ndvi_chart_logic(df_clean, sel_name)
        st.markdown("---")
        st.markdown("### 2. Spatial Vegetation Impact Simulation")
        plot_ndvi_heatmap(df_clean, sel_name)
    else:
        st.error("No data matching filters.")

elif nav == "Odisha Map":
    st.subheader("🗺️ Geographic Focus: Odisha (Locked View)")
    st.markdown("Displays storm tracks strictly within the Odisha region bounds (17.5N - 22.5N). Zoom is locked.")
    if not df_filt.empty:
        plot_odisha_map_locked(df_filt)
    else:
        st.error("No data matching filters.")
