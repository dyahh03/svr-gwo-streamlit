import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
from io import BytesIO

# Import utilitas
sys.path.append(str(Path(__file__).parent.parent))
from utils.backend import compare_models

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(page_title="Perbandingan", page_icon="⚖️", layout="wide")

st.markdown("# Perbandingan Model")
st.markdown("SVR Default vs SVR-GWO")
st.markdown("---")

# =============================================================================
# CEK MODEL
# =============================================================================

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Data belum dimuat.")
    st.stop()

default_res = st.session_state.get('default_results')
gwo_res = st.session_state.get('gwo_results')

if default_res is None or gwo_res is None:
    st.warning("⚠️ Kedua model harus ditraining terlebih dahulu.")
    if default_res is None:
        st.info("→ Train **SVR Default** di halaman SVR Default")
    if gwo_res is None:
        st.info("→ Train **SVR-GWO** di halaman SVR GWO")
    st.stop()

# Ambil data
comparison = compare_models(default_res, gwo_res)
target_col = st.session_state.get('target_col', 'Close')
data_dict = st.session_state.data_dict

# =============================================================================
# PEMENANG
# =============================================================================

st.markdown("### 🏆 Model Terbaik")

if comparison['best_model'] == 'SVR-GWO':
    st.success(f"🐺 **SVR-GWO Menang!** Peningkatan **{comparison['improvement']:.2f}%**")
else:
    st.info("🤖 **SVR Default Menang!** GWO tidak menemukan parameter lebih baik.")

# =============================================================================
# TABEL PERBANDINGAN
# =============================================================================

st.markdown("---")
st.markdown("### 📊 Tabel Perbandingan")

df_comp = pd.DataFrame({
    'Metrik': ['MAPE Train (%)', 'MAPE Test (%)', 'RMSE Test', 'R² Test', 'C', 'ε', 'γ'],
    'SVR Default': [
        default_res['metrics']['train']['mape'],
        default_res['metrics']['test']['mape'],
        default_res['metrics']['test']['rmse'],
        default_res['metrics']['test']['r2'],
        default_res['params']['C'],
        default_res['params']['epsilon'],
        default_res['params'].get('actual_gamma', default_res['params']['gamma'])
    ],
    'SVR-GWO': [
        gwo_res['metrics']['train']['mape'],
        gwo_res['metrics']['test']['mape'],
        gwo_res['metrics']['test']['rmse'],
        gwo_res['metrics']['test']['r2'],
        gwo_res['best_params']['C'],
        gwo_res['best_params']['epsilon'],
        gwo_res['best_params']['gamma']
    ]
})

# Tentukan pemenang per metrik
def get_winner(row):
    if row['Metrik'] in ['MAPE Train (%)', 'MAPE Test (%)', 'RMSE Test']:
        return '🐺' if row['SVR-GWO'] < row['SVR Default'] else '🤖'
    elif row['Metrik'] == 'R² Test':
        return '🐺' if row['SVR-GWO'] > row['SVR Default'] else '🤖'
    return '-'

df_comp['Best'] = df_comp.apply(get_winner, axis=1)
st.dataframe(df_comp, use_container_width=True, hide_index=True)

# =============================================================================
# VISUALISASI METRIK
# =============================================================================

st.markdown("---")
st.markdown("### 📈 Visualisasi Perbandingan")

col1, col2 = st.columns(2)

with col1:
    # MAPE
    fig = go.Figure(data=[
        go.Bar(name='Train', x=['Default', 'GWO'], 
               y=[default_res['metrics']['train']['mape'], gwo_res['metrics']['train']['mape']],
               marker_color=['#3498DB', '#2ECC71']),
        go.Bar(name='Test', x=['Default', 'GWO'],
               y=[default_res['metrics']['test']['mape'], gwo_res['metrics']['test']['mape']],
               marker_color=['#E74C3C', '#9B59B6'])
    ])
    fig.update_layout(title='MAPE (%)', barmode='group', template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # R²
    fig = go.Figure(data=[
        go.Bar(name='Train', x=['Default', 'GWO'],
               y=[default_res['metrics']['train']['r2'], gwo_res['metrics']['train']['r2']],
               marker_color=['#3498DB', '#2ECC71']),
        go.Bar(name='Test', x=['Default', 'GWO'],
               y=[default_res['metrics']['test']['r2'], gwo_res['metrics']['test']['r2']],
               marker_color=['#E74C3C', '#9B59B6'])
    ])
    fig.update_layout(title='R²', barmode='group', template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PERBANDINGAN PREDIKSI
# =============================================================================

st.markdown("---")
st.markdown("### 📈 Perbandingan Prediksi")

y_actual = default_res['predictions']['test_actual']
y_def = default_res['predictions']['test']
y_gwo = gwo_res['predictions']['test']

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(y_actual))), y=y_actual, 
                          mode='lines+markers', name='Actual', line=dict(color='#4F8BF9')))
fig.add_trace(go.Scatter(x=list(range(len(y_def))), y=y_def,
                          mode='lines+markers', name='Default', line=dict(color='#E74C3C', dash='dash')))
fig.add_trace(go.Scatter(x=list(range(len(y_gwo))), y=y_gwo,
                          mode='lines+markers', name='GWO', line=dict(color='#2ECC71', dash='dot')))
fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PERBANDINGAN ERROR
# =============================================================================

st.markdown("---")
st.markdown("### 📉 Perbandingan Error")

err_def = np.abs(y_actual - y_def)
err_gwo = np.abs(y_actual - y_gwo)

fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1, len(y_actual)+1)), y=err_def, 
                      name='Default', marker_color='#E74C3C', opacity=0.7))
fig.add_trace(go.Bar(x=list(range(1, len(y_actual)+1)), y=err_gwo,
                      name='GWO', marker_color='#2ECC71', opacity=0.7))
fig.update_layout(barmode='group', template='plotly_dark', height=400)
st.plotly_chart(fig, use_container_width=True)

# Statistik error
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🤖 Default")
    st.metric("MAE", f"Rp {err_def.mean():,.2f}")
    st.metric("Max Error", f"Rp {err_def.max():,.2f}")
with col2:
    st.markdown("#### 🐺 GWO")
    st.metric("MAE", f"Rp {err_gwo.mean():,.2f}")
    st.metric("Max Error", f"Rp {err_gwo.max():,.2f}")

# =============================================================================
# DOWNLOAD
# =============================================================================

st.markdown("---")
st.markdown("### 💾 Download")

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df_comp.to_excel(writer, sheet_name='Comparison', index=False)
    pd.DataFrame({
        'No': range(1, len(y_actual)+1), 'Actual': y_actual,
        'Pred_Default': y_def, 'Pred_GWO': y_gwo,
        'Err_Default': err_def, 'Err_GWO': err_gwo
    }).to_excel(writer, sheet_name='Predictions', index=False)

st.download_button("📥 Download Excel", buffer.getvalue(), "comparison.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================================================================
# KESIMPULAN
# =============================================================================

st.markdown("---")
st.markdown("### 📝 Kesimpulan")

if comparison['best_model'] == 'SVR-GWO':
    st.success(f"""
    **SVR-GWO** lebih baik dengan peningkatan **{comparison['improvement']:.2f}%**.
    
    Parameter optimal: C={gwo_res['best_params']['C']:.6f}, 
    ε={gwo_res['best_params']['epsilon']:.6f}, γ={gwo_res['best_params']['gamma']:.6f}
    
    ✅ Gunakan SVR-GWO untuk forecasting.
    """)
else:
    st.info("""
    **SVR Default** lebih baik. Kemungkinan:
    - Data cocok dengan parameter default
    - GWO terjebak di local optima
    
    → Coba konfigurasi GWO berbeda atau gunakan SVR Default.
    """)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("Halaman ini membandingkan performa kedua model untuk menentukan yang terbaik.")
