import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Import utilitas
sys.path.append(str(Path(__file__).parent.parent))
from utils.backend import (
    load_data_from_url, 
    load_data_from_file, 
    preprocess_data,
    split_and_scale_data,
    get_data_summary
)

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(page_title="Upload Data", page_icon="📁", layout="wide")

# Inisialisasi session state
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# =============================================================================
# HEADER
# =============================================================================

st.markdown("# Upload Data")
st.markdown("Muat dataset untuk analisis dan prediksi")
st.markdown("---")

# =============================================================================
# SUMBER DATA
# =============================================================================

st.markdown("### 📥 Pilih Sumber Data")

data_source = st.radio(
    "Bagaimana Anda ingin memuat data?",
    ["🌐 URL Default (ASII)", "📤 Upload File", "🔗 URL Custom"],
    horizontal=True
)

df = st.session_state.uploaded_df

# URL Default
if data_source == "🌐 URL Default (ASII)":
    st.info("Dataset: Harga Penutupan Saham ASII dari GitHub")
    url = "https://raw.githubusercontent.com/dyahprimasarii/Data-Saham/refs/heads/main/Harga%20Penutupan%20ASII"
    
    if st.button("📥 Load Data", type="primary"):
        with st.spinner("Memuat data..."):
            loaded_df, error = load_data_from_url(url)
            if error:
                st.error(f"❌ Error: {error}")
            elif loaded_df is not None:
                st.session_state.uploaded_df = loaded_df
                st.success("✅ Data berhasil dimuat!")
                st.rerun()

# Upload File
elif data_source == "📤 Upload File":
    st.info("Upload file CSV atau Excel (.xlsx, .xls)")
    
    uploaded_file = st.file_uploader(
        "Pilih file",
        type=['csv', 'xlsx', 'xls'],
        help="File harus memiliki kolom tanggal dan kolom harga"
    )
    
    if uploaded_file is not None:
        if st.button("📥 Load File", type="primary"):
            with st.spinner("Memproses file..."):
                loaded_df, error = load_data_from_file(uploaded_file)
                if error:
                    st.error(f"❌ Error: {error}")
                elif loaded_df is not None:
                    st.session_state.uploaded_df = loaded_df
                    st.success(f"✅ '{uploaded_file.name}' berhasil dimuat!")
                    st.rerun()

# URL Custom
elif data_source == "🔗 URL Custom":
    st.info("Masukkan URL ke file CSV")
    
    custom_url = st.text_input("URL Dataset", placeholder="https://example.com/data.csv")
    
    if custom_url and st.button("📥 Load URL", type="primary"):
        with st.spinner("Memuat data..."):
            loaded_df, error = load_data_from_url(custom_url)
            if error:
                st.error(f"❌ Error: {error}")
            elif loaded_df is not None:
                st.session_state.uploaded_df = loaded_df
                st.success("✅ Data berhasil dimuat!")
                st.rerun()

# =============================================================================
# PREVIEW & PREPROCESSING
# =============================================================================

df = st.session_state.uploaded_df  # Refresh df

if df is not None:
    st.markdown("---")
    st.markdown("### 👀 Preview Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("**📊 Info Dataset:**")
        st.write(f"- Baris: {len(df)}")
        st.write(f"- Kolom: {len(df.columns)}")
        st.write(f"- {', '.join(df.columns.tolist())}")
    
    # Konfigurasi preprocessing
    st.markdown("---")
    st.markdown("### ⚙️ Konfigurasi Preprocessing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Auto-detect kolom Date
        date_idx = 0
        for i, col in enumerate(df.columns):
            if col.lower() == 'date':
                date_idx = i
                break
        date_col = st.selectbox("Kolom Tanggal", df.columns.tolist(), index=date_idx)
    
    with col2:
        # Auto-detect kolom Close
        close_idx = 0
        for i, col in enumerate(df.columns):
            if col.lower() == 'close':
                close_idx = i
                break
        target_col = st.selectbox("Kolom Target", df.columns.tolist(), index=close_idx)
    
    with col3:
        test_size = st.slider("Test Size (%)", 5, 30, 10, 5) / 100
    
    # Tombol proses
    if st.button("🚀 Proses Data", type="primary"):
        with st.spinner("Memproses data..."):
            df_proc, X, y, error = preprocess_data(df, date_col=date_col, target_col=target_col)
            
            if error:
                st.error(f"❌ Error: {error}")
            elif df_proc is not None:
                # Split dan scale
                data_dict = split_and_scale_data(X, y, test_size=test_size)
                summary = get_data_summary(df_proc, target_col=target_col, date_col=date_col)
                
                # Simpan ke session
                st.session_state.df = df_proc
                st.session_state.data_dict = data_dict
                st.session_state.data_summary = summary
                st.session_state.target_col = target_col
                st.session_state.date_col = date_col
                st.session_state.data_loaded = True
                
                # Reset model results
                st.session_state.default_results = None
                st.session_state.gwo_results = None
                
                st.success("✅ Data berhasil diproses!")
                st.rerun()

# =============================================================================
# RINGKASAN DATA TERPROSES
# =============================================================================

if st.session_state.get('data_loaded'):
    st.markdown("### 📊 Ringkasan Data Terproses")
    
    summary = st.session_state.data_summary
    data_dict = st.session_state.data_dict
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data", summary['total_data'])
    with col2:
        st.metric("Training", data_dict['train_size'])
    with col3:
        st.metric("Testing", data_dict['test_size'])
    with col4:
        st.metric("Outliers", summary['outliers']['count'])
    
    st.info("✅ Data siap! Lanjut ke **Dashboard** atau **EDA**.")

# =============================================================================
# STATUS SESSION
# =============================================================================

st.markdown("---")
st.markdown("### 📌 Status")

if st.session_state.data_loaded:
    st.success(f"✅ Data dimuat: {len(st.session_state.df)} baris")
    
    if st.button("🗑️ Reset Data"):
        # Reset semua session state
        keys = ['uploaded_df', 'data_loaded', 'df', 'data_dict', 
                'default_results', 'gwo_results', 'forecast_df']
        for key in keys:
            st.session_state[key] = None
        st.session_state.data_loaded = False
        st.rerun()

elif df is not None:
    st.warning("⚠️ Data belum diproses. Klik 'Proses Data' di atas.")
else:
    st.warning("⏳ Belum ada data yang dimuat")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("""
    **SVR-GWO Prediction**
    
    Upload data CSV/Excel dengan kolom:
    - Tanggal (Date)
    - Harga (Close)
    """)
