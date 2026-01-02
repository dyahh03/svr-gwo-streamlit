import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
from io import BytesIO

# Import utilitas
sys.path.append(str(Path(__file__).parent.parent))
from utils.backend import svr_default

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(page_title="SVR Default", page_icon="🤖", layout="wide")

st.markdown("# SVR Default")
st.markdown("SVR dengan parameter default (C=1.0, ε=0.1, γ='scale')")
st.markdown("---")

# Cek data
if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Data belum dimuat. Silakan ke halaman **Upload Data**.")
    st.stop()

# Ambil data
data_dict = st.session_state.data_dict
target_col = st.session_state.get('target_col', 'Close')

# =============================================================================
# INFO PARAMETER
# =============================================================================

st.markdown("### ⚙️ Parameter Default")

col1, col2, col3 = st.columns(3)
col1.info("**C = 1.0**\n\nRegularization parameter")
col2.info("**ε = 0.1**\n\nLebar epsilon-tube")
col3.info("**γ = 'scale'**\n\nKernel coefficient")

# =============================================================================
# TRAINING
# =============================================================================

st.markdown("---")
st.markdown("### 🏋️ Training Model")

# State untuk tombol
if 'is_training_default' not in st.session_state:
    st.session_state.is_training_default = False

# Tombol training
col_btn, col_status = st.columns([3, 1])

with col_btn:
    is_training = st.session_state.is_training_default
    btn_text = "⏳ Training..." if is_training else "🚀 Train SVR Default"
    train_btn = st.button(btn_text, type="primary", use_container_width=True, disabled=is_training)

with col_status:
    if st.session_state.is_training_default:
        st.info("⏳ Loading...")

# Trigger training
if train_btn and not st.session_state.is_training_default:
    st.session_state.is_training_default = True
    st.rerun()

# Eksekusi training
if st.session_state.is_training_default:
    with st.spinner("🔄 Training SVR..."):
        results = svr_default(
            data_dict['X_train_scaled'], data_dict['y_train_scaled'],
            data_dict['X_test_scaled'], data_dict['y_test_scaled'],
            scaler_y=data_dict['scaler_y']
        )
        st.session_state.default_results = results
        st.session_state.is_training_default = False
    st.success("✅ Training selesai!")
    st.rerun()

# =============================================================================
# HASIL
# =============================================================================

if st.session_state.get('default_results'):
    results = st.session_state.default_results
    
    st.markdown("---")
    st.markdown("### 📊 Hasil Evaluasi")
    
    # Metrik
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏋️ Training")
        st.dataframe(pd.DataFrame({
            'Metrik': ['MAPE', 'MSE', 'RMSE', 'R²'],
            'Nilai': [
                f"{results['metrics']['train']['mape']:.4f}%",
                f"{results['metrics']['train']['mse']:,.4f}",
                f"{results['metrics']['train']['rmse']:,.4f}",
                f"{results['metrics']['train']['r2']:.4f}"
            ]
        }), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🧪 Testing")
        st.dataframe(pd.DataFrame({
            'Metrik': ['MAPE', 'MSE', 'RMSE', 'R²'],
            'Nilai': [
                f"{results['metrics']['test']['mape']:.4f}%",
                f"{results['metrics']['test']['mse']:,.4f}",
                f"{results['metrics']['test']['rmse']:,.4f}",
                f"{results['metrics']['test']['r2']:.4f}"
            ]
        }), use_container_width=True, hide_index=True)
    
    # Metrik besar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAPE Train", f"{results['metrics']['train']['mape']:.2f}%")
    col2.metric("MAPE Test", f"{results['metrics']['test']['mape']:.2f}%")
    col3.metric("R² Test", f"{results['metrics']['test']['r2']:.4f}")
    col4.metric("Time", f"{results['training_time']:.3f}s")
    
    # Info model
    st.markdown("---")
    st.markdown("### 🔧 Info Model")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Support Vectors", results['model_info']['n_support_vectors'])
    col2.metric("Bias", f"{results['model_info']['bias']:.6f}")
    col3.metric("Gamma", f"{results['params']['actual_gamma']:.6f}")
    
    # Plot
    st.markdown("---")
    st.markdown("### 📈 Actual vs Predicted")
    
    y_actual = results['predictions']['test_actual']
    y_pred = results['predictions']['test']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_actual))), y=y_actual,
                              mode='lines+markers', name='Actual', 
                              line=dict(color='#4F8BF9')))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred,
                              mode='lines+markers', name='Predicted',
                              line=dict(color='#E74C3C', dash='dash')))
    fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel perbandingan
    st.markdown("### 📋 Tabel Perbandingan")
    
    df_comp = pd.DataFrame({
        'No': range(1, len(y_actual) + 1),
        'Actual': y_actual,
        'Predicted': y_pred,
        'Error': np.abs(y_actual - y_pred),
        'Error (%)': np.abs((y_actual - y_pred) / y_actual) * 100
    })
    
    st.dataframe(df_comp.style.format({
        'Actual': 'Rp {:,.0f}', 'Predicted': 'Rp {:,.0f}',
        'Error': 'Rp {:,.0f}', 'Error (%)': '{:.2f}%'
    }), use_container_width=True, height=300)
    
    # Download
    st.markdown("---")
    st.markdown("### 💾 Download")
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_comp.to_excel(writer, sheet_name='Predictions', index=False)
    
    st.download_button("📥 Download Excel", buffer.getvalue(), 
                       "svr_default_results.xlsx", 
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("👆 Klik tombol untuk training model SVR Default")
    st.markdown("""
    **Langkah:**
    1. Train model dengan C=1.0, ε=0.1, γ='scale'
    2. Evaluasi performa (MAPE, MSE, RMSE, R²)
    3. Visualisasi hasil prediksi
    """)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("SVR Default menggunakan parameter standar tanpa optimasi.")
