import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import numpy as np
import folium

# ==========================================
# 1. MOCK DATA GENERATION (Crucial for Render)
# ==========================================
def create_mock_data():
    """Generates synthetic cyclone track data so the dashboard runs without CSVs."""
    np.random.seed(42)
    
    # Cyclone paths and characteristics
    cyclones = {
        'Fani': {'lat_start': 15.0, 'lon_start': 82.0, 'lat_step': 0.8, 'lon_step': 0.5, 'wind_base': 200},
        'Amphan': {'lat_start': 10.0, 'lon_start': 85.0, 'lat_step': 1.2, 'lon_step': 0.1, 'wind_base': 240},
        'Titli': {'lat_start': 14.0, 'lon_start': 83.0, 'lat_step': 0.6, 'lon_step': -0.2, 'wind_base': 150},
        'Hudhud': {'lat_start': 12.0, 'lon_start': 86.0, 'lat_step': 0.7, 'lon_step': -0.6, 'wind_base': 180},
        'Phailin': {'lat_start': 16.0, 'lon_start': 88.0, 'lat_step': 0.5, 'lon_step': -0.7, 'wind_base': 210},
    }
    
    data = []
    
    for name, params in cyclones.items():
        lat = params['lat_start']
        lon = params['lon_start']
        steps = 15 
        
        for i in range(steps):
            # Wind intensity curve
            if i < steps/2: wind = params['wind_base'] + np.random.randint(0, 30)
            else: wind = params['wind_base'] - (i * 10)
            wind = max(40, wind)
            
            data.append({
                'Name': name,
                'Lat': lat + np.random.normal(0, 0.05),
                'Lon': lon + np.random.normal(0, 0.05),
                'Max_Wind_Speed': wind,
                'Pressure': 1000 - (wind / 5)
            })
            lat += params['lat_step']
            lon += params['lon_step']

    df = pd.DataFrame(data)
    # Standardize column names
    df['Latitude'] = df['Lat'] 
    df['Longitude'] = df['Lon']
    return df

# Initialize the data
df_clean = create_mock_data()

# ==========================================
# 2. VISUALIZATION FUNCTIONS (Standard)
# ==========================================

def plot_overview(df, selection):
    """Tab 1: Overview Chart."""
    plt.figure(figsize=(8, 4))
    if selection == 'All':
        avg_wind = df.groupby('Name')['Max_Wind_Speed'].max().sort_values()
        sns.barplot(x=avg_wind.index, y=avg_wind.values, palette='Blues_d')
        plt.title("Peak Wind Speed by Cyclone")
    else:
        sns.lineplot(data=df, x=df.index, y='Max_Wind_Speed', marker='o')
        plt.title(f"Wind Speed Progression: {selection}")
    plt.ylabel("Wind Speed (km/h)")
    plt.tight_layout()
    plt.show()

def plot_statistics(df):
    """Tab 2: Statistics Charts."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df['Max_Wind_Speed'], kde=True, ax=ax[0], color='purple')
    ax[0].set_title("Wind Speed Distribution")
    sns.scatterplot(data=df, x='Pressure', y='Max_Wind_Speed', ax=ax[1], color='orange')
    ax[1].set_title("Wind vs. Pressure")
    plt.tight_layout()
    plt.show()

def plot_geospatial(df):
    """Tab 3: Simple Global Map."""
    m = folium.Map(location=[20, 85], zoom_start=5)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']], radius=2, color='blue', fill=True
        ).add_to(m)
    display(m)

def plot_sql_analysis(df):
    """Tab 4: Data Table."""
    display(Markdown("### 📋 Data Extract"))
    display(df.head(10))

# ==========================================
# 3. ADVANCED FUNCTIONS (Heatmap & Locked Zoom)
# ==========================================

def plot_odisha_map_focused(df):
    """
    Advanced Map: Focused on Odisha with LOCKED ZOOM and storm tracks.
    """
    odisha_lat_min, odisha_lat_max = 17.5, 22.5
    odisha_lon_min, odisha_lon_max = 81.5, 87.5
    
    # Identify relevant storms
    odisha_storms = df[
        (df['Lat'] >= odisha_lat_min) & (df['Lat'] <= odisha_lat_max) & 
        (df['Lon'] >= odisha_lon_min) & (df['Lon'] <= odisha_lon_max)
    ]['Name'].unique()
    
    # LOCKED ZOOM Map Setup
    m = folium.Map(
        location=[20.2, 84.4], 
        zoom_start=7, 
        tiles='CartoDB positron',
        zoom_control=False,       # Disable +/- buttons
        scrollWheelZoom=False,    # Disable mouse wheel
        doubleClickZoom=False,    # Disable double click
        boxZoom=False,            # Disable box zoom
        keyboard=False            # Disable keyboard
    )
    
    # Odisha Boundary Box
    folium.Rectangle(
        bounds=[[odisha_lat_min, odisha_lon_min], [odisha_lat_max, odisha_lon_max]],
        color="black", weight=2, fill=False, dash_array='5, 5', tooltip="Odisha Region"
    ).add_to(m)

    if len(odisha_storms) == 0:
        display(Markdown("**No data in Odisha region for current selection.**"))
        return m

    # Plot Tracks
    # Use global df_clean to get full tracks, but filter by names found in Odisha
    relevant_data = df_clean[df_clean['Name'].isin(odisha_storms)]
    
    for name in odisha_storms:
        storm_data = relevant_data[relevant_data['Name'] == name]
        locations = list(zip(storm_data['Lat'], storm_data['Lon']))
        
        # Path Line
        folium.PolyLine(locations=locations, color="blue", weight=2, opacity=0.5).add_to(m)
        
        # Points (Only if inside Odisha)
        for _, row in storm_data.iterrows():
            if (odisha_lat_min <= row['Lat'] <= odisha_lat_max) and (odisha_lon_min <= row['Lon'] <= odisha_lon_max):
                color = 'green'
                if row['Max_Wind_Speed'] > 150: color = 'red'
                elif row['Max_Wind_Speed'] > 100: color = 'orange'
                
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']], radius=4,
                    popup=f"{row['Name']}: {row['Max_Wind_Speed']} km/h",
                    color='black', weight=1, fill=True, fill_color=color, fill_opacity=0.9
                ).add_to(m)
    display(m)

def plot_ndvi_spatial_heatmap(df, cyclone_name):
    """
    Advanced Viz: Simulated SPATIAL VEGETATION DAMAGE HEATMAP.
    """
    # Grid setup
    lat_min, lat_max = 17.5, 22.5
    lon_min, lon_max = 81.5, 87.5
    grid_res = 0.05
    lats = np.arange(lat_min, lat_max, grid_res)
    lons = np.arange(lon_min, lon_max, grid_res)
    xx, yy = np.meshgrid(lons, lats)
    
    # Pre-cyclone state (random healthy forest)
    np.random.seed(42)
    ndvi_before = 0.7 + np.random.normal(0, 0.05, xx.shape)
    ndvi_before = np.clip(ndvi_before, 0.0, 0.9)
    
    # Post-cyclone state calculation
    ndvi_after = ndvi_before.copy()
    
    target_storms = df if cyclone_name == 'All' else df[df['Name'] == cyclone_name]
    
    for _, row in target_storms.iterrows():
        # Optimization: Only process points inside the grid
        if not (lat_min-1 < row['Lat'] < lat_max+1): continue
            
        wind = row['Max_Wind_Speed']
        radius = 0.5 + (wind / 300)
        severity = (wind / 250) * 0.8
        
        dist = np.sqrt((xx - row['Lon'])**2 + (yy - row['Lat'])**2)
        damage = np.exp(-0.5 * (dist / (radius/2))**2) * severity
        ndvi_after -= damage

    ndvi_after = np.clip(ndvi_after, 0.05, 0.9)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Map 1: Before
    axes[0].imshow(ndvi_before, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[0].set_title("Pre-Cyclone Vegetation (Simulated)")
    axes[0].set_ylabel("Latitude")
    axes[0].set_xlabel("Longitude")
    
    # Map 2: After
    im = axes[1].imshow(ndvi_after, extent=[lon_min, lon_max, lat_min, lat_max], origin='lower', cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[1].set_title(f"Post-Event Impact: {cyclone_name}")
    axes[1].set_xlabel("Longitude")
    
    # Overlay tracks
    if not target_storms.empty:
        if cyclone_name == 'All':
            axes[1].scatter(target_storms['Lon'], target_storms['Lat'], c='black', s=1, alpha=0.3)
        else:
            axes[1].plot(target_storms['Lon'], target_storms['Lat'], 'k--', alpha=0.6)

    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='NDVI Index')
    plt.suptitle("Vegetation Damage Spatial Simulation", fontsize=16)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. DASHBOARD WIDGETS & LOGIC
# ==========================================

# Widgets
dropdown_name = widgets.Dropdown(options=['All'] + sorted(df_clean['Name'].unique().tolist()), value='All', description='Storm:')
slider_wind = widgets.IntRangeSlider(value=[40, 250], min=40, max=250, description='Wind Range:')
btn_damage = widgets.Button(description='NDVI Analysis', button_style='success', icon='leaf')
btn_map = widgets.Button(description='Odisha Map', button_style='info', icon='map-marker')
out_display = widgets.Output()

def update_dashboard(change=None):
    """Main refresh logic."""
    with out_display:
        clear_output(wait=True)
        
        # Filter
        mask = (df_clean['Max_Wind_Speed'] >= slider_wind.value[0]) & (df_clean['Max_Wind_Speed'] <= slider_wind.value[1])
        df_filtered = df_clean[mask]
        if dropdown_name.value != 'All':
            df_filtered = df_filtered[df_filtered['Name'] == dropdown_name.value]

        # Tabs
        tabs = widgets.Tab()
        tab_outputs = [widgets.Output() for _ in range(4)]
        tabs.children = tab_outputs
        for i, t in enumerate(['Overview', 'Stats', 'Global Map', 'Data']): tabs.set_title(i, t)
        display(tabs)

        with tab_outputs[0]: plot_overview(df_filtered, dropdown_name.value) if not df_filtered.empty else print("No Data")
        with tab_outputs[1]: plot_statistics(df_filtered) if not df_filtered.empty else None
        with tab_outputs[2]: plot_geospatial(df_filtered) if not df_filtered.empty else None
        with tab_outputs[3]: plot_sql_analysis(df_filtered) if not df_filtered.empty else None

def show_damage_view(b):
    """NDVI Button Logic - Triggers Heatmap."""
    with out_display:
        clear_output(wait=True)
        btn_back = widgets.Button(description="⬅ Back", button_style='warning')
        btn_back.on_click(lambda x: update_dashboard())
        display(btn_back)
        
        display(Markdown("### 📉 Spatial Vegetation Impact Simulation"))
        # Use unfiltered data for 'All' or specific storm data based on dropdown
        mask = (df_clean['Max_Wind_Speed'] >= slider_wind.value[0]) & (df_clean['Max_Wind_Speed'] <= slider_wind.value[1])
        filtered_for_viz = df_clean[mask]
        
        plot_ndvi_spatial_heatmap(filtered_for_viz, dropdown_name.value)
        
        display(Markdown("> **Note:** This heatmap uses a distance-decay model to simulate vegetation loss based on wind intensity."))

def show_map_view(b):
    """Map Button Logic - Triggers Locked Zoom Map."""
    with out_display:
        clear_output(wait=True)
        btn_back = widgets.Button(description="⬅ Back", button_style='warning')
        btn_back.on_click(lambda x: update_dashboard())
        display(btn_back)
        
        display(Markdown("### 🗺️ Geographic Analysis: Odisha Region Focus"))
        
        mask = (df_clean['Max_Wind_Speed'] >= slider_wind.value[0]) & (df_clean['Max_Wind_Speed'] <= slider_wind.value[1])
        plot_odisha_map_focused(df_clean[mask])

# Bindings
dropdown_name.observe(update_dashboard, names='value')
slider_wind.observe(update_dashboard, names='value')
btn_damage.on_click(show_damage_view)
btn_map.on_click(show_map_view)

# Initial Render
display(widgets.VBox([
    widgets.HTML("<h2>🌪️ Cyclone Impact Dashboard</h2>"),
    widgets.HBox([dropdown_name, slider_wind, btn_damage, btn_map]),
    out_display
]))
update_dashboard()
