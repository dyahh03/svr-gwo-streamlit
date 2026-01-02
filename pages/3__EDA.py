import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import sys
from pathlib import Path

# Import utilitas
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_processing import terasvirta_test

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")

st.markdown("# 📈 Exploratory Data Analysis")
st.markdown("Visualisasi dan analisis data eksploratori")
st.markdown("---")

# Cek data
if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Data belum dimuat. Silakan ke halaman **Upload Data**.")
    st.stop()

# =============================================================================
# AMBIL DATA
# =============================================================================

df = st.session_state.df
data_dict = st.session_state.data_dict
target_col = st.session_state.get('target_col', 'Close')
date_col = st.session_state.get('date_col', 'Date')

# Auto-detect kolom tanggal
if date_col not in df.columns:
    for c in ['Date', 'DATE', 'date', 'Tanggal']:
        if c in df.columns:
            date_col = c
            break

# =============================================================================
# TABS VISUALISASI
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Time Series", "📊 Histogram", "📦 Boxplot", "🔗 Split", "📉 PACF", "🧪 Uji Linearitas"
])

# --- Tab 1: Time Series ---
with tab1:
    st.markdown("### 📈 Time Series Harga")
    
    fig = px.line(df, x=date_col, y=target_col, title=f'Harga {target_col}')
    fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
    fig.update_traces(line=dict(color='#4F8BF9', width=1.5))
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistik ringkas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tertinggi", f"Rp {df[target_col].max():,.0f}")
    col2.metric("Terendah", f"Rp {df[target_col].min():,.0f}")
    col3.metric("Rata-rata", f"Rp {df[target_col].mean():,.0f}")
    change = df[target_col].iloc[-1] - df[target_col].iloc[0]
    col4.metric("Perubahan", f"Rp {change:,.0f}", f"{change/df[target_col].iloc[0]*100:.1f}%")

# --- Tab 2: Histogram ---
with tab2:
    st.markdown("### 📊 Distribusi Harga")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        n_bins = st.slider("Bins", 10, 50, 20)
    
    with col1:
        fig = px.histogram(df, x=target_col, nbins=n_bins, title=f'Distribusi {target_col}')
        fig.update_layout(template='plotly_dark', height=500)
        fig.update_traces(marker_color='#4F8BF9', opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Skewness", f"{df[target_col].skew():.4f}")
    col2.metric("Kurtosis", f"{df[target_col].kurtosis():.4f}")
    col3.metric("Std Dev", f"Rp {df[target_col].std():,.0f}")

# --- Tab 3: Boxplot ---
with tab3:
    st.markdown("### 📦 Boxplot (Outliers)")
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[target_col], name=target_col, 
                         marker_color='#4F8BF9', boxmean=True, boxpoints='outliers'))
    fig.update_layout(template='plotly_dark', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hitung outliers
    Q1, Q3 = df[target_col].quantile(0.25), df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[target_col] < lower) | (df[target_col] > upper)]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Lower Bound", f"Rp {lower:,.0f}")
    col2.metric("Upper Bound", f"Rp {upper:,.0f}")
    col3.metric("Outliers", len(outliers))
    
    if len(outliers) > 0:
        with st.expander("📋 Lihat Outliers"):
            st.dataframe(outliers, use_container_width=True)

# --- Tab 4: Train-Test Split ---
with tab4:
    st.markdown("### 🔗 Train-Test Split")
    
    y_train, y_test = data_dict['y_train'], data_dict['y_test']
    train_idx = list(range(len(y_train)))
    test_idx = list(range(len(y_train), len(y_train) + len(y_test)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_idx, y=y_train, mode='lines', 
                              name='Training', line=dict(color='#4F8BF9')))
    fig.add_trace(go.Scatter(x=test_idx, y=y_test, mode='lines', 
                              name='Testing', line=dict(color='#E74C3C')))
    fig.add_vline(x=len(y_train), line_dash="dash", line_color="green", 
                  annotation_text="Split")
    fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    total = len(y_train) + len(y_test)
    col1, col2 = st.columns(2)
    col1.info(f"**Training:** {len(y_train)} ({len(y_train)/total*100:.0f}%)")
    col2.info(f"**Testing:** {len(y_test)} ({len(y_test)/total*100:.0f}%)")

# --- Tab 5: PACF ---
with tab5:
    st.markdown("### 📉 PACF (Partial Autocorrelation)")
    st.info("PACF membantu menentukan jumlah lag optimal untuk time series.")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        n_lags = st.slider("Lag", 10, 50, 30)
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_pacf(df[target_col], lags=n_lags, ax=ax)
        ax.set_title('PACF', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    st.markdown("**Interpretasi:** Lag 1 biasanya paling signifikan untuk data saham.")

# --- Tab 6: Uji Linearitas Terasvirta ---
with tab6:
    st.markdown("### 🧪 Uji Linearitas Terasvirta")
    st.info("""
    **Uji Terasvirta** digunakan untuk menguji apakah hubungan antara variabel independen (lag) 
    dan variabel dependen (harga) bersifat **linear** atau **non-linear**.
    
    - **H0 (Null Hypothesis):** Hubungan bersifat linear
    - **H1 (Alternative Hypothesis):** Hubungan bersifat non-linear
    """)
    
    # Ambil X dan y dari data_dict
    X = data_dict['X_train']
    y = data_dict['y_train']
    
    # Jalankan uji Terasvirta
    try:
        terasvirta_result = terasvirta_test(X, y, alpha=0.05)
        
        st.markdown("---")
        st.markdown("#### 📊 Hasil Uji Statistik")
        
        # Tampilkan metrik dalam kolom
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="LM Statistic",
                value=f"{terasvirta_result['lm_statistic']:.4f}",
                help="Lagrange Multiplier statistic (chi-square)"
            )
        
        with col2:
            st.metric(
                label="P-Value",
                value=f"{terasvirta_result['p_value']:.6f}",
                help="Nilai probabilitas untuk menolak H0"
            )
        
        with col3:
            st.metric(
                label="Critical Value (α=0.05)",
                value=f"{terasvirta_result['critical_value']:.4f}",
                help="Nilai kritis chi-square pada α=0.05"
            )
        
        with col4:
            st.metric(
                label="Degrees of Freedom",
                value=f"{terasvirta_result['degrees_of_freedom']}",
                help="Derajat kebebasan uji chi-square"
            )
        
        st.markdown("---")
        st.markdown("#### 🎯 Kesimpulan")
        
        # Tampilkan kesimpulan dengan warna berbeda
        if terasvirta_result['is_nonlinear']:
            st.warning(f"""
            ### 🔶 Data Bersifat **NON-LINEAR**
            
            **Keputusan:** Tolak H0 (p-value = {terasvirta_result['p_value']:.6f} < α = 0.05)
            
            **Interpretasi:** 
            - Hubungan antara lag dan harga **tidak linear**
            - **SVR dengan kernel RBF** cocok karena dapat menangkap pola non-linear
            - Kedua model (SVR Default & SVR-GWO) menggunakan kernel RBF
            
            ℹ️ **Catatan:** Bandingkan kedua model di halaman Perbandingan untuk melihat mana yang lebih baik.
            """)
        else:
            st.success(f"""
            ### ✅ Data Bersifat **LINEAR**
            
            **Keputusan:** Gagal Tolak H0 (p-value = {terasvirta_result['p_value']:.6f} ≥ α = 0.05)
            
            **Interpretasi:**
            - Hubungan antara lag dan harga **linear**
            - Model linear seperti Linear Regression bisa digunakan
            - SVR dengan kernel RBF juga tetap bisa digunakan
            
            ℹ️ **Catatan:** SVR tetap dapat memberikan hasil baik untuk data linear.
            """)
        
        # Tabel ringkasan
        st.markdown("---")
        st.markdown("#### 📋 Ringkasan Uji")
        
        summary_df = pd.DataFrame({
            'Parameter': ['LM Statistic', 'Degrees of Freedom', 'P-Value', 'Critical Value (α=0.05)', 'Kesimpulan'],
            'Nilai': [
                f"{terasvirta_result['lm_statistic']:.4f}",
                f"{terasvirta_result['degrees_of_freedom']}",
                f"{terasvirta_result['p_value']:.6f}",
                f"{terasvirta_result['critical_value']:.4f}",
                terasvirta_result['conclusion']
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Expander untuk detail teknis
        with st.expander("📖 Detail Teknis Uji Terasvirta"):
            st.markdown("""
            **Metode Uji Terasvirta:**
            
            1. **Step 1:** Fit model linear (OLS) antara X dan y
            2. **Step 2:** Hitung residual dari model linear
            3. **Step 3:** Regresi auxiliary: residual² terhadap X, fitted², fitted³
            4. **Step 4:** Hitung LM statistic = n × R² (dari regresi auxiliary)
            5. **Step 5:** Bandingkan dengan distribusi chi-square (df=2)
            
            **Interpretasi:**
            - Jika **p-value < 0.05**: Tolak H0, data bersifat non-linear
            - Jika **p-value ≥ 0.05**: Gagal tolak H0, data bersifat linear
            
            **Referensi:** Teräsvirta, T. (1994). Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models.
            """)
            
    except Exception as e:
        st.error(f"❌ Gagal menjalankan uji Terasvirta: {str(e)}")
        st.info("Pastikan data sudah dipreprocessing dengan benar.")

# =============================================================================
# KORELASI LAG
# =============================================================================

st.markdown("---")
st.markdown("### 🔄 Korelasi Lag")

# Hitung korelasi untuk lag 1-5
df_corr = df.copy()
for i in range(1, 6):
    df_corr[f'Lag_{i}'] = df_corr[target_col].shift(i)
df_corr = df_corr.dropna()

corr_vals = [df_corr[target_col].corr(df_corr[f'Lag_{i}']) for i in range(1, 6)]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[f'Lag {i}' for i in range(1, 6)],
    y=corr_vals,
    marker_color=['#4F8BF9' if c > 0.9 else '#9B59B6' for c in corr_vals],
    text=[f'{c:.4f}' for c in corr_vals],
    textposition='outside'
))
fig.update_layout(template='plotly_dark', height=400, yaxis_range=[0, 1.1])
st.plotly_chart(fig, use_container_width=True)

st.success(f"✅ Korelasi Lag 1: **{corr_vals[0]:.4f}** - Lag 1 cocok sebagai prediktor.")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("EDA menampilkan visualisasi interaktif untuk memahami karakteristik data.")