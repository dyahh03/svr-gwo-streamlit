import streamlit as st

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================

st.set_page_config(
    page_title="SVR-GWO Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #4F8BF9, #9B59B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4F8BF9;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4F8BF9, #9B59B6);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #3a7be0, #8e44ad);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INISIALISASI SESSION STATE
# =============================================================================

defaults = {
    'data_loaded': False,
    'df': None,
    'data_dict': None,
    'default_results': None,
    'gwo_results': None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🐺 SVR-GWO Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prediksi Harga Saham dengan Support Vector Regression + Grey Wolf Optimizer</p>', unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# FITUR APLIKASI
# =============================================================================

st.markdown("## ✨ Fitur Aplikasi")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Dashboard</h3>
        <p>Ringkasan statistik, outliers, dan overview dataset</p>
    </div>
    <div class="feature-card">
        <h3>📈 EDA</h3>
        <p>Visualisasi time series, histogram, boxplot, PACF</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 SVR Default</h3>
        <p>Model SVR dengan parameter default (C=1.0, ε=0.1)</p>
    </div>
    <div class="feature-card">
        <h3>🐺 SVR + GWO</h3>
        <p>SVR dengan optimasi Grey Wolf Optimizer</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>⚖️ Perbandingan</h3>
        <p>Bandingkan performa model Default vs GWO</p>
    </div>
    <div class="feature-card">
        <h3>🔮 Forecasting</h3>
        <p>Prediksi harga untuk periode mendatang</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# CARA MENGGUNAKAN
# =============================================================================

st.markdown("## 📖 Cara Menggunakan")

st.markdown("""
1. **📁 Upload Data** → Muat dataset dari file atau URL
2. **📊 Dashboard** → Lihat ringkasan dan statistik data
3. **📈 EDA** → Eksplorasi dengan visualisasi interaktif
4. **🤖 Training** → Jalankan model SVR Default dan SVR-GWO
5. **⚖️ Perbandingan** → Bandingkan hasil kedua model
6. **🔮 Forecast** → Prediksi harga untuk hari mendatang
""")

st.markdown("---")

# =============================================================================
# STATUS SAAT INI
# =============================================================================

st.markdown("## 📌 Status Saat Ini")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.data_loaded:
        st.success("✅ Data Loaded")
        total = len(st.session_state.df) if st.session_state.df is not None else 0
        st.metric("Total Data", total)
    else:
        st.warning("⏳ Data Belum Dimuat")

with col2:
    if st.session_state.default_results:
        st.success("✅ SVR Default")
        mape = st.session_state.default_results['metrics']['test']['mape']
        st.metric("MAPE Test", f"{mape:.2f}%")
    else:
        st.info("⏳ Belum Ditraining")

with col3:
    if st.session_state.gwo_results:
        st.success("✅ SVR-GWO")
        mape = st.session_state.gwo_results['metrics']['test']['mape']
        st.metric("MAPE Test", f"{mape:.2f}%")
    else:
        st.info("⏳ Belum Ditraining")

with col4:
    if st.session_state.gwo_results and st.session_state.default_results:
        mape_def = st.session_state.default_results['metrics']['test']['mape']
        mape_gwo = st.session_state.gwo_results['metrics']['test']['mape']
        improvement = ((mape_def - mape_gwo) / mape_def) * 100
        st.success("✅ Ready")
        st.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    else:
        st.info("⏳ Belum Dibandingkan")

st.markdown("---")

# =============================================================================
# SIDEBAR & FOOTER
# =============================================================================

with st.sidebar:
    st.markdown("### 📚 Tentang")
    st.info("""
    **SVR-GWO Prediction**
    
    Aplikasi ini menggunakan:
    - SVR (Support Vector Regression)
    - GWO (Grey Wolf Optimizer)
    
    Untuk prediksi harga saham.
    """)

st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>SVR-GWO Stock Price Prediction | 2025</p>
</div>
""", unsafe_allow_html=True)
