import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import numpy as np
import folium

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
try:
    # Load data
    df_clean = pd.read_csv('cyclone_data.csv')
    
    # standardize column names
    if 'Latitude' in df_clean.columns and 'Lat' not in df_clean.columns:
        df_clean['Lat'] = df_clean['Latitude']
    if 'Longitude' in df_clean.columns and 'Lon' not in df_clean.columns:
        df_clean['Lon'] = df_clean['Longitude']
        
except FileNotFoundError:
    df_clean = None
    print("❌ Error: 'cyclone_data.csv' not found. Please upload your dataset to the repository.")

# ==========================================
# 2. DEFINING MISSING HELPER FUNCTIONS
# (These were called in your code but not defined)
# ==========================================

def plot_overview(df, name):
    """Generates the Overview tab content."""
    display(Markdown(f"### 🌪️ Overview: {name}"))
    if name == 'All':
        display(Markdown(f"**Total Storms:** {len(df['Name'].unique())}"))
        display(Markdown(f"**Max Wind Speed Recorded:** {df['Max_Wind_Speed'].max()} km/h"))
    else:
        storm = df.iloc[0]
        display(Markdown(f"**Max Wind Speed:** {df['Max_Wind_Speed'].max()} km/h"))
        display(Markdown(f"**Total Track Points:** {len(df)}"))

def plot_statistics(df):
    """Generates the Statistics tab content."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Max_Wind_Speed'], kde=True, color='skyblue')
    plt.title("Wind Speed Distribution")
    plt.xlabel("Speed (km/h)")
    plt.show()

def plot_geospatial(df):
    """Generates the Global Map tab content."""
    center = [df['Lat'].mean(), df['Lon'].mean()]
    m = folium.Map(location=center, zoom_start=4)
    for _, row in df.iterrows():
        folium.CircleMarker([row['Lat'], row['Lon']], radius=1, color='blue').add_to(m)
    display(m)

def plot_sql_analysis(df):
    """Generates the SQL/Viz tab content."""
    display(Markdown("### 📊 Data Table"))
    display(df.head(10))

# ==========================================
# 3. DASHBOARD LOGIC (Your Custom Code)
# ==========================================

if df_clean is not None:
    # --- WIDGETS ---
    unique_names = ['All'] + sorted(df_clean['Name'].unique().tolist())
    
    dropdown_name = widgets.Dropdown(
        options=unique_names, 
        value='All',
        description='Select Storm:'
    )
    
    min_wind = int(df_clean['Max_Wind_Speed'].min())
    max_wind = int(df_clean['Max_Wind_Speed'].max())
    slider_wind = widgets.IntRangeSlider(
        value=[min_wind, max_wind], 
        min=min_wind, 
        max=max_wind, 
        description='Wind Range:'
    )

    btn_damage = widgets.Button(description='NDVI Analysis', button_style='success', icon='leaf')
    btn_map = widgets.Button(description='Odisha Map Analysis', button_style='info', icon='map-marker')
    
    out_display = widgets.Output()

    # --- HELPER: SPATIAL NDVI MAP SIMULATION ---
    def plot_ndvi_analysis(df, cyclone_name):
        """
        Generates a spatial heatmap comparing vegetation before and after the cyclone.
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
        
        # Iterate through storm tracks
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
        im1 = axes[0].imshow(ndvi_before, extent=[lon_min, lon_max, lat_min, lat_max], 
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
        if not target_storms.empty:
            if cyclone_name == 'All':
                axes[1].scatter(target_storms['Lon'], target_storms['Lat'], c='black', s=1, alpha=0.3)
            else:
                axes[1].plot(target_storms['Lon'], target_storms['Lat'], 'k--', linewidth=1, alpha=0.7, label='Track')
                axes[1].legend()

        cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('NDVI (Greenness Index)')
        plt.suptitle(f"Vegetation Damage Simulation: {cyclone_name}", fontsize=16)
        plt.show()

    # --- HELPER: ODISHA MAP (LOCKED VIEW) ---
    def plot_odisha_map_locked(df):
        odisha_lat_min, odisha_lat_max = 17.5, 22.5
        odisha_lon_min, odisha_lon_max = 81.5, 87.5
        
        odisha_storms = df[
            (df['Lat'] >= odisha_lat_min) & (df['Lat'] <= odisha_lat_max) & 
            (df['Lon'] >= odisha_lon_min) & (df['Lon'] <= odisha_lon_max)
        ]['Name'].unique()
        
        m = folium.Map(
            location=[20.2, 84.4], 
            zoom_start=7, 
            tiles='CartoDB positron',
            zoom_control=False, scrollWheelZoom=False, 
            doubleClickZoom=False, boxZoom=False, keyboard=False
        )
        
        folium.Rectangle(
            bounds=[[odisha_lat_min, odisha_lon_min], [odisha_lat_max, odisha_lon_max]],
            color="black", weight=3, fill=False, dash_array='5, 5', tooltip="Odisha Region"
        ).add_to(m)

        if len(odisha_storms) == 0:
            display(Markdown("**No cyclones found passing through Odisha with current filters.**"))
            return m

        for name in odisha_storms:
            storm_data = df[df['Name'] == name]
            locations = list(zip(storm_data['Lat'], storm_data['Lon']))
            
            folium.PolyLine(locations=locations, color="blue", weight=2, opacity=0.5).add_to(m)
            
            for _, row in storm_data.iterrows():
                in_odisha = (odisha_lat_min <= row['Lat'] <= odisha_lat_max) and \
                            (odisha_lon_min <= row['Lon'] <= odisha_lon_max)
                
                fill_color = 'green'
                if row['Max_Wind_Speed'] > 150: fill_color = 'red'
                elif row['Max_Wind_Speed'] > 100: fill_color = 'orange'
                
                if in_odisha:
                    folium.CircleMarker(
                        location=[row['Lat'], row['Lon']], radius=4,
                        popup=f"<b>{row['Name']}</b><br>Wind: {row['Max_Wind_Speed']} km/h",
                        color='black', weight=1, fill=True, fill_color=fill_color, fill_opacity=0.9
                    ).add_to(m)
        display(m)

    # --- VIEW HANDLERS ---
    def update_dashboard(change=None):
        with out_display:
            clear_output(wait=True)
            
            # Filter Data
            mask = (df_clean['Max_Wind_Speed'] >= slider_wind.value[0]) & (df_clean['Max_Wind_Speed'] <= slider_wind.value[1])
            df_filtered = df_clean[mask]
            
            if dropdown_name.value != 'All':
                df_filtered = df_filtered[df_filtered['Name'] == dropdown_name.value]

            # Create Tabs
            tab1, tab2, tab3, tab4 = [widgets.Output() for _ in range(4)]
            tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4])
            titles = ['Overview', 'Stats', 'Density', 'SQL & Viz']
            for i, t in enumerate(titles): tabs.set_title(i, t)
            display(tabs)

            with tab1: 
                if not df_filtered.empty: plot_overview(df_filtered, dropdown_name.value)
            with tab2:
                if not df_filtered.empty: plot_statistics(df_filtered)
            with tab3:
                if not df_filtered.empty: plot_geospatial(df_filtered)
            with tab4:
                 plot_sql_analysis(df_filtered)

    def show_damage_view(b):
        with out_display:
            clear_output(wait=True)
            
            btn_back = widgets.Button(description="⬅ Back to Dashboard", button_style='warning')
            btn_back.on_click(lambda x: update_dashboard())
            display(btn_back)

            current_cyclone = dropdown_name.value
            mask = (df_clean['Max_Wind_Speed'] >= slider_wind.value[0]) & (df_clean['Max_Wind_Speed'] <= slider_wind.value[1])
            filtered_df = df_clean[mask]

            display(Markdown(f"### 📉 Spatial Vegetation Impact Simulation"))
            plot_ndvi_analysis(filtered_df, current_cyclone)
            
            display(Markdown("""
            > **Color Guide:**
            > * 🟩 **Dark Green:** Healthy, dense vegetation.
            > * 🟨 **Yellow/Orange:** Damaged or stressed vegetation.
            > * 🟥 **Red:** Severe loss of canopy / bare ground.
            """))

    def show_map_view(b):
        with out_display:
            clear_output(wait=True)
            btn_back = widgets.Button(description="⬅ Back to Dashboard", button_style='warning')
            btn_back.on_click(lambda x: update_dashboard())
            display(btn_back)
            
            display(Markdown(f"### 🗺️ Geographic Analysis: Odisha Region Focus"))
            
            mask = (df_clean['Max_Wind_Speed'] >= slider_wind.value[0]) & (df_clean['Max_Wind_Speed'] <= slider_wind.value[1])
            filtered_df = df_clean[mask]
            
            if dropdown_name.value != 'All':
                filtered_df = filtered_df[filtered_df['Name'] == dropdown_name.value]
            
            plot_odisha_map_locked(filtered_df)

    # Observers & Buttons
    dropdown_name.observe(update_dashboard, names='value')
    slider_wind.observe(update_dashboard, names='value')
    btn_damage.on_click(show_damage_view)
    btn_map.on_click(show_map_view)

    # Initial Display
    display(widgets.VBox([
        widgets.HTML("<h2>🌪️ Cyclone Impact & Vegetation Dashboard</h2>"),
        widgets.HBox([dropdown_name, slider_wind, btn_damage, btn_map]),
        out_display
    ]))
    
    # Trigger initial load
    update_dashboard()
