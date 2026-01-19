import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from streamlit_folium import st_folium

# ==========================================
# 0. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Cyclone Impact Dashboard",
    page_icon="🌪️",
    layout="wide"
)

# ==========================================
# 1. DATA LOADING (Placeholder/Mock Data)
# ==========================================
# NOTE: Replace this function with your actual data loading logic.
# e.g., df_clean = pd.read_csv("your_data.csv")
@st.cache_data
def load_data():
    # Creating dummy data to make the app runnable immediately
    data = {
        'Name': ['FANI', 'AMPHAN', 'HUDHUD', 'PHAILIN', 'TITLI'] * 20,
        'Max_Wind_Speed': np.random.randint(60, 260, 100),
        'Lat': np.random.uniform(17.5, 22.5, 100),  # Odisha Latitudes
        'Lon': np.random.uniform(81.5, 87.5, 100),  # Odisha Longitudes
    }
    df = pd.DataFrame(data)
    # Ensure standard column names used in your logic
    df['Latitude'] = df['Lat']
    df['Longitude'] = df['Lon']
    return df

df_clean = load_data()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def plot_ndvi_analysis(df, cyclone_name):
    """
    Generates a spatial heatmap comparing vegetation before and after.
    """
    # 1. Setup Grid over Odisha Region
    lat_min, lat_max = 17.5, 22.5
    lon_min, lon_max = 81.5, 87.5
    grid_res = 0.05 
    
    lats = np.arange(lat_min, lat_max, grid_res)
    lons = np.arange(lon_min, lon_max, grid_res)
    xx, yy = np.meshgrid(lons, lats)
    
    # 2. Initialize "Before" State
    np.random.seed(42) 
    ndvi_before = 0.7 + np.random.normal(0, 0.05, xx.shape)
    ndvi_before = np.clip(ndvi_before, 0.0, 0.9)
    
    # 3. Compute "After" State
    ndvi_after = ndvi_before.copy()
    
    if cyclone_name == 'All':
        target_storms = df
    else:
        target_storms = df[df['Name'] == cyclone_name]
    
    if target_storms.empty:
        st.warning("No track data available for this selection within the simulation bounds.")
        return

    # Iterate through points
    for _, row in target_storms.iterrows():
        c_lat, c_lon = row['Lat'], row['Lon']
        wind = row['Max_Wind_Speed']
        
        radius = 0.5 + (wind / 300)
        severity = (wind / 250) * 0.8
        
        dist = np.sqrt((xx - c_lon)**2 + (yy - c_lat)**2)
        damage_mask = np.exp(-0.5 * (dist / (radius/2))**2) * severity
        ndvi_after -= damage_mask

    ndvi_after = np.clip(ndvi_after, 0.05, 0.9)

    # 4. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot A: BEFORE
    axes[0].imshow(ndvi_before, extent=[lon_min, lon_max, lat_min, lat_max], 
                   origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[0].set_title("PRE-CYCLONE VEGETATION (Simulated)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    
    # Plot B: AFTER
    im2 = axes[1].imshow(ndvi_after, extent=[lon_min, lon_max, lat_min, lat_max], 
                   origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[1].set_title(f"POST-EVENT IMPACT: {cyclone_name}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Longitude")
    
    # Add storm tracks
    if cyclone_name == 'All':
        axes[1].scatter(target_storms['Lon'], target_storms['Lat'], c='black', s=1, alpha=0.3)
    else:
        axes[1].plot(target_storms['Lon'], target_storms['Lat'], 'k--', linewidth=1, alpha=0.7, label='Track')
        axes[1].legend()

    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('NDVI (Greenness Index)')
    
    plt.suptitle(f"Vegetation Damage Simulation: {cyclone_name}", fontsize=16)
    st.pyplot(fig)

def plot_odisha_map_folium(df):
    """
    Plots the interactive Folium map focused on Odisha.
    """
    odisha_lat_min, odisha_lat_max = 17.5, 22.5
    odisha_lon_min, odisha_lon_max = 81.5, 87.5
    
    # Filter for Odisha storms
    odisha_storms = df[
        (df['Lat'] >= odisha_lat_min) & (df['Lat'] <= odisha_lat_max) & 
        (df['Lon'] >= odisha_lon_min) & (df['Lon'] <= odisha_lon_max)
    ]['Name'].unique()
    
    m = folium.Map(
        location=[20.2, 84.4], 
        zoom_start=7,
        tiles='CartoDB positron'
    )
    
    # Draw Odisha Box
    folium.Rectangle(
        bounds=[[odisha_lat_min, odisha_lon_min], [odisha_lat_max, odisha_lon_max]],
        color="black", weight=3, fill=False, dash_array='5, 5', tooltip="Odisha Region"
    ).add_to(m)

    if len(odisha_storms) == 0:
        st.warning("No cyclones found passing through Odisha with current filters.")
        return m

    for name in odisha_storms:
        storm_data = df[df['Name'] == name]
        locations = list(zip(storm_data['Lat'], storm_data['Lon']))
        
        # Path
        folium.PolyLine(
            locations=locations, color="blue", weight=2, opacity=0.5, tooltip=f"Path: {name}"
        ).add_to(m)
        
        # Points
        for _, row in storm_data.iterrows():
            in_odisha = (odisha_lat_min <= row['Lat'] <= odisha_lat_max) and \
                        (odisha_lon_min <= row['Lon'] <= odisha_lon_max)
            
            fill_color = 'green'
            if row['Max_Wind_Speed'] > 150: fill_color = 'red'
            elif row['Max_Wind_Speed'] > 100: fill_color = 'orange'
            
            if in_odisha:
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']],
                    radius=4,
                    popup=f"<b>{row['Name']}</b><br>Wind: {row['Max_Wind_Speed']} km/h",
                    color='black', weight=1, fill=True, fill_color=fill_color, fill_opacity=0.9
                ).add_to(m)
    return m

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Control Panel")

# Navigation Mode (Replaces the "Action Buttons")
view_mode = st.sidebar.radio(
    "Select Analysis View:",
    ("Dashboard Overview", "NDVI Vegetation Analysis", "Odisha Geo-Analysis")
)

st.sidebar.markdown("---")

# 1. Dropdown
unique_names = ['All'] + sorted(df_clean['Name'].unique().tolist())
selected_storm = st.sidebar.selectbox("Select Storm:", unique_names, index=0)

# 2. Slider
min_wind = int(df_clean['Max_Wind_Speed'].min())
max_wind = int(df_clean['Max_Wind_Speed'].max())
wind_range = st.sidebar.slider("Wind Range (km/h):", min_wind, max_wind, (min_wind, max_wind))

# ==========================================
# 4. DATA FILTERING
# ==========================================
mask = (df_clean['Max_Wind_Speed'] >= wind_range[0]) & (df_clean['Max_Wind_Speed'] <= wind_range[1])
filtered_df = df_clean[mask]

if selected_storm != 'All':
    filtered_df = filtered_df[filtered_df['Name'] == selected_storm]

# ==========================================
# 5. MAIN CONTENT LOGIC
# ==========================================

st.title("🌪️ Cyclone Impact & Vegetation Dashboard")

if view_mode == "Dashboard Overview":
    # Using Tabs for the standard dashboard views
    tab1, tab2, tab3 = st.tabs(["Overview", "Statistics", "About"])
    
    with tab1:
        st.subheader("Current Selection Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cyclones Selected", filtered_df['Name'].nunique())
        with col2:
            st.metric("Avg Wind Speed", f"{filtered_df['Max_Wind_Speed'].mean():.1f} km/h")
        
        st.dataframe(filtered_df.head(10))
        
    with tab2:
        if not filtered_df.empty:
            fig, ax = plt.subplots()
            sns.histplot(filtered_df['Max_Wind_Speed'], kde=True, ax=ax, color='skyblue')
            ax.set_title("Wind Speed Distribution")
            st.pyplot(fig)
        else:
            st.info("No data matches the filters.")

    with tab3:
        st.markdown("""
        ### About this Dashboard
        This tool visualizes cyclone tracks, wind speeds, and simulates vegetation impact (NDVI) 
        specifically focusing on the Odisha region.
        """)

elif view_mode == "NDVI Vegetation Analysis":
    st.markdown(f"### 📉 Spatial Vegetation Impact Simulation: {selected_storm}")
    st.info("Generating heatmap based on wind intensity and track proximity...")
    
    if not filtered_df.empty:
        plot_ndvi_analysis(filtered_df, selected_storm)
    else:
        st.error("No data available to plot.")

    st.markdown("""
    > **Color Guide:**
    > * 🟩 **Dark Green:** Healthy, dense vegetation.
    > * 🟨 **Yellow/Orange:** Damaged or stressed vegetation.
    > * 🟥 **Red:** Severe loss of canopy / bare ground.
    """)

elif view_mode == "Odisha Geo-Analysis":
    st.markdown(f"### 🗺️ Geographic Analysis: Odisha Region Focus")
    
    if not filtered_df.empty:
        # Folium map rendering in Streamlit
        m = plot_odisha_map_folium(filtered_df)
        st_folium(m, width=1000, height=500)
    else:
        st.error("No data available to plot.")

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("---")
st.caption("Generated via Streamlit • Odisha Cyclone Analysis Module")
