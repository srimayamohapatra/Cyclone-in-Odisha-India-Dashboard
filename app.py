import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cyclone Dashboard", layout="wide")

# ==========================================
# 1. IMAGE LOADING LOGIC
# ==========================================
@st.cache_resource
def load_image():
    """
    Tries to load the image. Prioritizes the compressed version 
    to be faster and stay under GitHub limits.
    """
    possible_names = ["Puri_compressed.jpg", "Puri.tif.jpg", "Puri.tif"]
    
    for name in possible_names:
        if os.path.exists(name):
            try:
                return Image.open(name)
            except Exception:
                continue
    return None

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    # Look for files in the current directory
    possible_files = ['Cyclone.xlsx', 'Cyclone.csv']
    file_path = None
    for f in possible_files:
        if os.path.exists(f):
            file_path = f
            break
    
    if not file_path:
        return None, "Data file not found. Please upload 'Cyclone.xlsx' to your GitHub repository."

    try:
        if file_path.endswith('.csv'): 
            df = pd.read_csv(file_path)
        else: 
            df = pd.read_excel(file_path)

        # Standardize column names
        rename_map = {
            'Maximum Sustained Surface Wind (km/hr) ': 'Max_Wind_Speed',
            'Estimated Central Pressure (hPa) [or "E.C.P"]': 'Pressure',
            'Longitude (lon.)': 'Lon', 'Latitude (lat.)': 'Lat',
            'Grade (text)': 'Grade', 'Pressure Drop (hPa)[or "delta P"]': 'Pressure_Drop',
            'Name': 'Name', 'Serial Number of system during year': 'Serial_No'
        }
        df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
        
        # Numeric cleanup
        if 'CI No [or "T. No"]' in df.columns:
            df['CI No [or "T. No"]'] = pd.to_numeric(df['CI No [or "T. No"]'], errors='coerce')
            
        return df, "Success"
    except Exception as e:
        return None, str(e)

df_clean, msg = load_data()

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.title("🌪️ Ultimate Cyclone Dashboard")

if df_clean is not None:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Filter Settings")
    
    # Handle cleaning of Names to prevent errors
    clean_names = df_clean['Name'].dropna().unique().tolist()
    unique_names = ['All'] + sorted([str(x) for x in clean_names])
    
    selected_name = st.sidebar.selectbox("Select Cyclone", unique_names)
    
    min_wind = int(df_clean['Max_Wind_Speed'].min())
    max_wind = int(df_clean['Max_Wind_Speed'].max())
    
    wind_range = st.sidebar.slider("Wind Speed Range (km/h)", min_wind, max_wind, (min_wind, max_wind))

    # Filter Data based on selection
    mask = (df_clean['Max_Wind_Speed'] >= wind_range[0]) & (df_clean['Max_Wind_Speed'] <= wind_range[1])
    df_filtered = df_clean[mask]
    
    if selected_name != 'All':
        df_filtered = df_filtered[df_filtered['Name'] == selected_name]

    # --- DAMAGE REPORT SECTION ---
    if st.sidebar.button("📸 View Puri Damage Report"):
        st.subheader("🌪️ Impact: Puri 2019 Fani Cyclone")
        st.markdown("""
        **Cyclone Fani** was one of the strongest tropical cyclones to hit Odisha.
        * **Wind Speed:** 205 km/h
        * **Impact:** Massive infrastructure collapse, loss of green cover, and heritage damage.
        """)
        
        img = load_image()
        if img:
            st.image(img, caption="Damage in Puri", use_container_width=True)
        else:
            st.warning("Image file (Puri_compressed.jpg) not found in the repository.")

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overview', 'Stats', 'Density', 'Map', 'SQL Playground'])

    # 1. OVERVIEW TAB
    with tab1:
        st.subheader("Cyclone Overview")
        col1, col2 = st.columns(2)
        
        # Trajectory Path
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df_filtered, x='Lon', y='Lat', hue='Name', ax=ax1, palette='viridis', legend=False)
        ax1.set_title("Trajectory Path")
        col1.pyplot(fig1)

        # Grade Count
        if 'Grade' in df_filtered.columns:
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df_filtered, y='Grade', ax=ax2, palette='magma', order=df_filtered['Grade'].value_counts().index)
            ax2.set_title("Intensity Grade Distribution")
            col2.pyplot(fig2)

    # 2. STATS TAB
    with tab2:
        if 'Grade' in df_filtered.columns:
            st.subheader("Wind Speed by Grade")
            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                order = df_filtered.groupby('Grade')['Max_Wind_Speed'].median().sort_values().index
                sns.boxenplot(data=df_filtered, x='Grade', y='Max_Wind_Speed', order=order, palette='cool', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.info("Not enough data to generate statistics.")

    # 3. DENSITY TAB
    with tab3:
        st.subheader("Geospatial Density")
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            sns.kdeplot(data=df_filtered, x='Lon', y='Lat', fill=True, cmap='Reds', alpha=0.5, ax=ax)
            sns.scatterplot(data=df_filtered, x='Lon', y='Lat', size='Max_Wind_Speed', hue='Max_Wind_Speed', palette='viridis', ax=ax)
            st.pyplot(fig)
        except Exception:
            st.info("Not enough data points for density plot.")
        
    # 4. MAP TAB (FIXED HERE)
    with tab4:
        st.subheader("Geographical Map")
        if not df_filtered.empty:
            # We explicitly rename columns to lowercase 'lat' and 'lon' 
            # because Streamlit map function is case-sensitive.
            map_data = df_filtered[['Lat', 'Lon']].rename(columns={'Lat': 'lat', 'Lon': 'lon'})
            st.map(map_data)
        else:
            st.write("No data available for map.")

    # 5. SQL TAB
    with tab5:
        st.subheader("🔍 SQL Playground")
        
        # Setup In-Memory SQL
        conn = sqlite3.connect(':memory:')
        
        # Convert objects to strings to prevent SQL errors
        df_sql = df_filtered.copy()
        for col in df_sql.columns:
            if df_sql[col].dtype == 'object':
                df_sql[col] = df_sql[col].astype(str)
                
        df_sql.to_sql('cyclones', conn, index=False, if_exists='replace')

        presets = {
            "Custom Query": "",
            "Max Wind by Cyclone": "SELECT Name, MAX(Max_Wind_Speed) as Max_Wind FROM cyclones GROUP BY Name ORDER BY Max_Wind DESC LIMIT 10",
            "Avg Wind by Grade": "SELECT Grade, AVG(Max_Wind_Speed) as Avg_Wind FROM cyclones GROUP BY Grade"
        }
        
        selected_preset = st.selectbox("Choose a Query Preset:", list(presets.keys()))
        default_query = presets[selected_preset] if selected_preset != "Custom Query" else "SELECT * FROM cyclones LIMIT 5"
        
        query = st.text_area("SQL Query:", value=default_query)
        
        if st.button("Run Query"):
            try:
                query_df = pd.read_sql_query(query, conn)
                st.dataframe(query_df)
                
                # Simple Plotting for SQL results
                if not query_df.empty and len(query_df.columns) >= 2:
                    st.write("### Quick Viz")
                    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])
                    x_axis = st.selectbox("X Axis", query_df.columns)
                    y_axis = st.selectbox("Y Axis", query_df.columns, index=1)
                    
                    fig, ax = plt.subplots()
                    if chart_type == "Bar": sns.barplot(data=query_df, x=x_axis, y=y_axis, ax=ax)
                    elif chart_type == "Line": sns.lineplot(data=query_df, x=x_axis, y=y_axis, ax=ax)
                    elif chart_type == "Scatter": sns.scatterplot(data=query_df, x=x_axis, y=y_axis, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"SQL Error: {e}")

else:
    st.error(msg)
