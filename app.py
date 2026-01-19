import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
from PIL import Image
import warnings

# Optional: Geopandas for the specific map view requested
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Suppress warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Ultimate Cyclone Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Intro
st.title("🌪️ Ultimate Cyclone Dashboard")
st.markdown("Analyze Cyclone Fani and others with interactive geospatial data, SQL querying, and damage assessment.")

# ==========================================
# 1. IMAGE COMPRESSION & PRE-PROCESSING
# ==========================================
# We cache this function so it doesn't run on every interaction
@st.cache_resource
def process_image(input_path, output_path, target_mb):
    if not os.path.exists(input_path):
        return False, f"Input file not found: {input_path}"
    
    # If output already exists and is small enough, skip
    if os.path.exists(output_path):
        return True, "Using existing compressed file."

    try:
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(input_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        quality = 95
        img.save(output_path, "JPEG", optimize=True, quality=quality)
        file_size = os.path.getsize(output_path) / (1024 * 1024)

        step = 5
        while file_size > target_mb and quality > 10:
            quality -= step
            img.save(output_path, "JPEG", optimize=True, quality=quality)
            file_size = os.path.getsize(output_path) / (1024 * 1024)

        if file_size > target_mb:
            while file_size > target_mb:
                width, height = img.size
                img = img.resize((int(width * 0.8), int(height * 0.8)), Image.LANCZOS)
                img.save(output_path, "JPEG", optimize=True, quality=30)
                file_size = os.path.getsize(output_path) / (1024 * 1024)

        return True, "Compression Successful"
    except Exception as e:
        return False, str(e)

# ==========================================
# 2. DATA LOADING & CONFIGURATION
# ==========================================
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            return None, str(e)
    else:
        # Fallback to local files if no upload
        possible_files = ['Cyclone.xlsx', 'Cyclone.csv']
        file_path = None
        for f in possible_files:
            if os.path.exists(f):
                file_path = f
                break
        
        if not file_path:
            return None, "No data file found. Please upload a dataset."
            
        try:
            if file_path.endswith('.csv'): df = pd.read_csv(file_path)
            else: df = pd.read_excel(file_path)
        except Exception as e:
            return None, str(e)

    # Standardizing Columns
    rename_map = {
        'Maximum Sustained Surface Wind (km/hr) ': 'Max_Wind_Speed',
        'Estimated Central Pressure (hPa) [or "E.C.P"]': 'Pressure',
        'Longitude (lon.)': 'Lon', 'Latitude (lat.)': 'Lat',
        'Grade (text)': 'Grade', 'Pressure Drop (hPa)[or "delta P"]': 'Pressure_Drop',
        'Name': 'Name', 'Serial Number of system during year': 'Serial_No'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # Date Parsing
    if 'Date(DD-MM-YYYY)' in df.columns and 'Time (UTC)' in df.columns:
        try:
            df['Datetime'] = pd.to_datetime(df['Date(DD-MM-YYYY)'].astype(str) + ' ' + df['Time (UTC)'].astype(str), errors='coerce')
        except:
            pass

    cols_to_drop = ['Basin of origin', 'Date(DD-MM-YYYY)', 'Time (UTC)']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    if 'CI No [or "T. No"]' in df.columns:
        df['CI No [or "T. No"]'] = pd.to_numeric(df['CI No [or "T. No"]'], errors='coerce')

    return df, "Success"

# Sidebar: File Upload
st.sidebar.header("📂 Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload Cyclone Data (Excel/CSV)", type=['xlsx', 'csv'])
df_clean, data_msg = load_data(uploaded_file)

if df_clean is None:
    st.error(data_msg)
    st.stop()
else:
    st.sidebar.success(f"Data Loaded: {len(df_clean)} records")

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Dashboard Controls")

# Cyclone Filter
unique_names = ['All'] + sorted(df_clean['Name'].unique().tolist())
selected_name = st.sidebar.selectbox("Select Cyclone:", unique_names)

# Wind Speed Filter
min_wind = int(df_clean['Max_Wind_Speed'].min())
max_wind = int(df_clean['Max_Wind_Speed'].max())
selected_wind = st.sidebar.slider("Wind Speed Range (km/h):", min_wind, max_wind, (min_wind, max_wind))

# Filter Logic
mask = (df_clean['Max_Wind_Speed'] >= selected_wind[0]) & (df_clean['Max_Wind_Speed'] <= selected_wind[1])
df_filtered = df_clean[mask]
if selected_name != 'All':
    df_filtered = df_filtered[df_filtered['Name'] == selected_name]

# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================

def plot_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 1. Trajectory
    sns.scatterplot(data=df, x='Lon', y='Lat', hue='Name', ax=axes[0, 0], palette='viridis', legend=False)
    axes[0, 0].set_title(f"📍 Trajectory Path")

    # 2. Frequency by Grade
    if 'Grade' in df.columns:
        sns.countplot(data=df, y='Grade', ax=axes[0, 1], palette='magma', order=df['Grade'].value_counts().index)
        axes[0, 1].set_title("🌪️ Intensity Grade Distribution")
        axes[0, 1].set_ylabel("")
    else:
        axes[0, 1].text(0.5, 0.5, "Grade Data Missing", ha='center')

    # 3. Regression
    sns.regplot(data=df, x='Pressure', y='Max_Wind_Speed', ax=axes[1, 0], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title("Pressure vs Wind")

    # 4. Distribution
    sns.histplot(df['Max_Wind_Speed'], kde=True, ax=axes[1, 1], color='skyblue')
    axes[1, 1].set_title("Wind Speed Distribution")
    
    return fig

def plot_statistics(df):
    if 'Grade' in df.columns and df['Grade'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        order = df.groupby('Grade')['Max_Wind_Speed'].median().sort_values().index
        sns.boxenplot(data=df, x='Grade', y='Max_Wind_Speed', order=order, palette='cool', ax=ax)
        plt.xticks(rotation=45)
        plt.title('Wind Speed by Grade')
        return fig
    else:
        return None

def plot_geospatial(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.kdeplot(data=df, x='Lon', y='Lat', fill=True, cmap='Reds', alpha=0.5, ax=ax)
    sns.scatterplot(data=df, x='Lon', y='Lat', size='Max_Wind_Speed', hue='Max_Wind_Speed', palette='viridis', ax=ax)
    plt.title('Geospatial Density')
    return fig

def plot_geopandas_map(df):
    if not HAS_GEOPANDAS:
        st.warning("Geopandas not installed. Falling back to basic map.")
        st.map(df[['Lat', 'Lon']].rename(columns={'Lat':'latitude', 'Lon':'longitude'}))
        return None

    try:
        geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)

        # Try loading world map
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except:
            world = None

        fig, ax = plt.subplots(figsize=(12, 10))
        if world is not None:
            world.plot(ax=ax, color='lightgrey', edgecolor='white')
        else:
            ax.set_facecolor('#f0f0f0')

        ax.set_xlim(df['Lon'].min()-5, df['Lon'].max()+5)
        ax.set_ylim(df['Lat'].min()-5, df['Lat'].max()+5)
        gdf.plot(ax=ax, column='Name', markersize=30, cmap='tab20', legend=True)
        plt.title("GeoPandas Map View")
        return fig
    except Exception as e:
        st.error(f"Error creating map: {e}")
        return None

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "📊 Stats", "🗺️ Density", "🌍 Map", "💾 SQL & Viz", "🚨 Damage Report"
])

with tab1:
    st.subheader("Overview Analysis")
    st.pyplot(plot_overview(df_filtered))

with tab2:
    st.subheader("Statistical Analysis")
    fig_stats = plot_statistics(df_filtered)
    if fig_stats:
        st.pyplot(fig_stats)
    else:
        st.info("Not enough data or missing 'Grade' column for Boxenplot.")

with tab3:
    st.subheader("Geospatial Density")
    st.pyplot(plot_geospatial(df_filtered))

with tab4:
    st.subheader("World Map View")
    fig_map = plot_geopandas_map(df_filtered)
    if fig_map:
        st.pyplot(fig_map)

# ==========================================
# 6. SQL INTERFACE (Tab 5)
# ==========================================
with tab5:
    st.subheader("🔍 SQL & Custom Graph Builder")
    
    # Setup In-Memory DB
    conn = sqlite3.connect(':memory:')
    df_sql = df_filtered.copy()
    if 'Datetime' in df_sql.columns:
        df_sql['Datetime'] = df_sql['Datetime'].astype(str)
    df_sql.to_sql('cyclones', conn, index=False, if_exists='replace')

    col1, col2 = st.columns([1, 2])
    
    with col1:
        presets = {
            "Select a Preset...": "",
            "1. Max Wind by Cyclone Name": "SELECT Name, MAX(Max_Wind_Speed) as Max_Wind FROM cyclones GROUP BY Name ORDER BY Max_Wind DESC LIMIT 10",
            "2. Avg Wind & Pressure by Grade": "SELECT Grade, AVG(Max_Wind_Speed) as Avg_Wind, AVG(Pressure) as Avg_Pressure FROM cyclones GROUP BY Grade",
            "3. Pressure vs Wind Data": "SELECT Pressure, Max_Wind_Speed FROM cyclones",
            "4. Count of Cyclones by Grade": "SELECT Grade, COUNT(*) as Count FROM cyclones GROUP BY Grade"
        }
        selected_preset = st.selectbox("SQL Presets:", list(presets.keys()))
    
    with col2:
        # If preset changes, use that query, otherwise use text area
        default_query = "SELECT * FROM cyclones LIMIT 10"
        query_val = presets[selected_preset] if selected_preset != "Select a Preset..." else default_query
        
        # Use session state to handle text area updates
        if 'query_text' not in st.session_state:
            st.session_state.query_text = default_query
        
        # Update session state if preset changed
        if selected_preset != "Select a Preset...":
             st.session_state.query_text = presets[selected_preset]

        txt_query = st.text_area("SQL Query:", value=st.session_state.query_text, height=100)
        run_query = st.button("Run Query")

    if run_query or txt_query:
        try:
            res = pd.read_sql_query(txt_query, conn)
            st.dataframe(res)
            
            st.markdown("---")
            st.markdown("#### 📊 Custom Visualization from Query Results")
            
            if not res.empty:
                cols = res.columns.tolist()
                c1, c2, c3 = st.columns(3)
                viz_type = c1.selectbox('Viz Type:', ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Pie Chart'])
                x_col = c2.selectbox('X Axis:', cols, index=0)
                y_col = c3.selectbox('Y Axis:', cols, index=1 if len(cols) > 1 else 0)
                
                fig_custom = plt.figure(figsize=(10, 5))
                try:
                    if viz_type == 'Bar Chart': sns.barplot(data=res, x=x_col, y=y_col, palette='viridis')
                    elif viz_type == 'Line Chart': sns.lineplot(data=res, x=x_col, y=y_col, marker='o')
                    elif viz_type == 'Scatter Plot': sns.scatterplot(data=res, x=x_col, y=y_col, s=100, alpha=0.7)
                    elif viz_type == 'Pie Chart':
                         if pd.api.types.is_numeric_dtype(res[y_col]):
                             plt.pie(res[y_col], labels=res[x_col], autopct='%1.1f%%')
                         else: st.warning("Y-Axis must be numeric for Pie Chart")
                    
                    plt.title(f"{viz_type}: {y_col} vs {x_col}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_custom)
                except Exception as e:
                    st.error(f"Could not generate graph: {e}")

        except Exception as e:
            st.error(f"SQL Error: {e}")

# ==========================================
# 7. DAMAGE REPORT (Tab 6)
# ==========================================
with tab6:
    st.subheader("🚨 Impact: Puri 2019 Fani Cyclone")
    
    # Image Input Settings
    # Allow user to upload the image since path is hardcoded
    img_file = st.file_uploader("Upload Puri Image (optional)", type=['jpg', 'jpeg', 'tif'])
    
    # Logic: Use uploaded file, OR local file "Puri.tif.jpg", OR placeholder
    input_image_path = "Puri.tif.jpg"
    output_image_path = "Puri_compressed.jpg"
    
    display_img = None
    
    if img_file:
        display_img = Image.open(img_file)
    elif os.path.exists(input_image_path):
        # Run compression logic
        success, msg = process_image(input_image_path, output_image_path, target_mb=20)
        if success:
            display_img = Image.open(output_image_path)
        else:
            st.warning(f"Compression failed: {msg}")
    
    col_img, col_text = st.columns([1, 1])
    
    with col_img:
        if display_img:
            st.image(display_img, caption="Aerial View of Puri", use_column_width=True)
        else:
            st.info("Upload 'Puri.tif.jpg' to see the damage view.")

    with col_text:
        st.markdown("""
        ### **Cyclone Fani (2019)**
        **Cyclone Fani** was one of the strongest tropical cyclones to hit Odisha since the 1999 Odisha cyclone.
        
        * **Max Wind Speed:** 205 km/h
        * **Landfall:** Near Puri, Odisha
        * **Impact:** * Massive infrastructure collapse.
            * Significant loss of green cover (millions of trees uprooted).
            * Severe damage to heritage sites including the Jagannath Temple.
            * Power and telecommunication blackout for weeks.
        """)
