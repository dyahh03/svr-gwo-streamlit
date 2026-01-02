import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
from io import BytesIO

# Import utilitas
sys.path.append(str(Path(__file__).parent.parent))
from utils.backend import svr_with_gwo

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(page_title="SVR-GWO", page_icon="🐺", layout="wide")

st.markdown("# SVR + Grey Wolf Optimizer")
st.markdown("Optimasi hyperparameter SVR dengan algoritma GWO")
st.markdown("---")

# Cek data
if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Data belum dimuat. Silakan ke halaman **Upload Data**.")
    st.stop()

# Ambil data
data_dict = st.session_state.data_dict
target_col = st.session_state.get('target_col', 'Close')

# =============================================================================
# PENJELASAN GWO
# =============================================================================

with st.expander("📚 Apa itu Grey Wolf Optimizer?", expanded=False):
    st.markdown("""
    **GWO** adalah algoritma optimasi berbasis perilaku berburu serigala abu-abu.
    
    **Hierarki:** Alpha (α) → Beta (β) → Delta (δ) → Omega (ω)
    
    **Parameter yang dioptimasi:**
    - **C**: Regularization (0.01-100)
    - **ε**: Epsilon-tube (0-1)
    - **γ**: Kernel coefficient (0.001-100)
    """)

# =============================================================================
# KONFIGURASI GWO
# =============================================================================

st.markdown("---")
st.markdown("### ⚙️ Konfigurasi GWO")

col1, col2 = st.columns(2)

with col1:
    n_wolves = st.slider("🐺 Jumlah Wolves", 10, 100, 30, 5,
                          help="Lebih banyak = eksplorasi lebih luas")

with col2:
    max_iter = st.slider("🔄 Iterasi", 20, 300, 50, 10,
                          help="Lebih banyak = hasil lebih optimal")

st.info(f"**Konfigurasi:** {n_wolves} wolves × {max_iter} iterasi = {n_wolves * max_iter} evaluasi")

# =============================================================================
# TRAINING
# =============================================================================

st.markdown("---")
st.markdown("### 🏋️ Training Model")

# State training
if 'is_training_gwo' not in st.session_state:
    st.session_state.is_training_gwo = False

# Tombol
col_btn, col_status = st.columns([3, 1])

with col_btn:
    is_training = st.session_state.is_training_gwo
    btn_text = "⏳ Optimasi..." if is_training else "🚀 Jalankan GWO"
    train_btn = st.button(btn_text, type="primary", use_container_width=True, disabled=is_training)

with col_status:
    if st.session_state.is_training_gwo:
        st.info("⏳ Loading...")

# Trigger training
if train_btn and not st.session_state.is_training_gwo:
    st.session_state.is_training_gwo = True
    st.rerun()

# Eksekusi training
if st.session_state.is_training_gwo:
    progress = st.progress(0, text="Memulai...")
    status = st.empty()
    
    def callback(i, m, f):
        progress.progress(i / m, text=f"Iterasi {i}/{m}")
        status.text(f"🐺 Best Fitness: {f:.8f}")
    
    with st.spinner("🐺 Menjalankan GWO..."):
        results = svr_with_gwo(
            data_dict['X_train_scaled'], data_dict['y_train_scaled'],
            data_dict['X_test_scaled'], data_dict['y_test_scaled'],
            data_dict['scaler_y'], n_wolves=n_wolves, max_iter=max_iter,
            progress_callback=callback
        )
        st.session_state.gwo_results = results
        st.session_state.is_training_gwo = False
    
    progress.progress(1.0, text="✅ Selesai!")
    status.text(f"✅ Runtime: {results['gwo_result']['runtime']:.2f}s")
    st.success("✅ Optimasi selesai!")
    st.rerun()

# =============================================================================
# HASIL
# =============================================================================

if st.session_state.get('gwo_results'):
    results = st.session_state.gwo_results
    gwo = results['gwo_result']
    
    # Parameter optimal
    st.markdown("---")
    st.markdown("### 🏆 Parameter Optimal")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("C", f"{results['best_params']['C']:.6f}")
    col2.metric("ε", f"{results['best_params']['epsilon']:.6f}")
    col3.metric("γ", f"{results['best_params']['gamma']:.6f}")
    col4.metric("Runtime", f"{gwo['runtime']:.2f}s")
    
    # Metrik
    st.markdown("---")
    st.markdown("### 📊 Hasil Evaluasi")
    
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
    col4.metric("Fitness", f"{gwo['best_fitness']:.8f}")
    
    # Convergence curve
    st.markdown("---")
    st.markdown("### 📉 Convergence Curve")
    
    conv = gwo['convergence']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(conv) + 1)), y=conv,
                              mode='lines+markers', line=dict(color='#4F8BF9')))
    fig.update_layout(template='plotly_dark', height=400, yaxis_type='log',
                      xaxis_title='Iterasi', yaxis_title='Fitness (MSE)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot prediksi
    st.markdown("---")
    st.markdown("### 📈 Actual vs Predicted")
    
    y_actual = results['predictions']['test_actual']
    y_pred = results['predictions']['test']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_actual))), y=y_actual,
                              mode='lines+markers', name='Actual', line=dict(color='#4F8BF9')))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred,
                              mode='lines+markers', name='Predicted', line=dict(color='#2ECC71', dash='dash')))
    fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Info model
    st.markdown("---")
    st.markdown("### 🔧 Info Model")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Support Vectors", results['model_info']['n_support_vectors'])
    col2.metric("Bias", f"{results['model_info']['bias']:.6f}")
    col3.metric("Evaluasi", f"{gwo['config']['n_wolves'] * gwo['config']['max_iter']}")
    
    # Tabel perbandingan
    st.markdown("---")
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
        pd.DataFrame({'Iter': range(1, len(conv)+1), 'Fitness': conv}).to_excel(writer, sheet_name='Convergence', index=False)
        pd.DataFrame({
            'Param': ['C', 'Epsilon', 'Gamma', 'Fitness', 'Runtime'],
            'Value': [results['best_params']['C'], results['best_params']['epsilon'],
                      results['best_params']['gamma'], gwo['best_fitness'], gwo['runtime']]
        }).to_excel(writer, sheet_name='Params', index=False)
    
    st.download_button("📥 Download Excel", buffer.getvalue(), "svr_gwo_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("👆 Konfigurasi parameter dan klik tombol untuk optimasi")
    st.markdown("""
    **Proses:**
    1. Inisialisasi wolves secara random
    2. Evaluasi fitness (MSE) tiap wolf
    3. Update posisi berdasarkan alpha, beta, delta
    4. Konvergensi ke parameter optimal
    5. Train SVR dan evaluasi
    """)
    st.warning("⚠️ Rekomendasi: 30×50 (cepat) atau 50×100 (optimal)")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("GWO mengoptimasi parameter C, ε, γ untuk menemukan kombinasi terbaik.")
