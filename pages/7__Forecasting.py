# -*- coding: utf-8 -*-
"""
🔮 Forecasting - SVR-GWO Streamlit App
======================================
Prediksi harga saham untuk periode mendatang.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
from io import BytesIO

# Import utilitas
sys.path.append(str(Path(__file__).parent.parent))
from utils.backend import forecast_future

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(page_title="Forecasting", page_icon="✨", layout="wide")

st.markdown("# Forecasting")
st.markdown("Prediksi harga untuk periode mendatang")
st.markdown("---")

# =============================================================================
# CEK DATA & MODEL
# =============================================================================

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Data belum dimuat.")
    st.stop()

gwo_res = st.session_state.get('gwo_results')
def_res = st.session_state.get('default_results')

if gwo_res is None and def_res is None:
    st.warning("⚠️ Belum ada model. Training di halaman SVR Default/GWO dulu.")
    st.stop()

# Ambil data
df = st.session_state.df
data_dict = st.session_state.data_dict
target_col = st.session_state.get('target_col', 'Close')
date_col = st.session_state.get('date_col', 'Date')

# Auto-detect date column
if date_col not in df.columns:
    for c in ['Date', 'DATE', 'date', 'Tanggal']:
        if c in df.columns:
            date_col = c
            break

# =============================================================================
# PILIH MODEL
# =============================================================================

st.markdown("### 🤖 Pilih Model")

models = []
if def_res: models.append("🤖 SVR Default")
if gwo_res: models.append("🐺 SVR-GWO")

selected = st.radio("Model:", models, horizontal=True)

if "GWO" in selected:
    model, model_name = gwo_res['model'], "SVR-GWO"
    mape = gwo_res['metrics']['test']['mape']
else:
    model, model_name = def_res['model'], "SVR Default"
    mape = def_res['metrics']['test']['mape']

st.info(f"**{model_name}** | MAPE: {mape:.4f}%")

# =============================================================================
# KONFIGURASI FORECAST
# =============================================================================

st.markdown("---")
st.markdown("### ⚙️ Konfigurasi")

col1, col2 = st.columns(2)

with col1:
    n_days = st.slider("📅 Jumlah Hari", 1, 60, 15, help="Hari kerja yang diprediksi")

with col2:
    apply_trend = st.checkbox("📈 Trend Adjustment", value=True, help="Tambahkan trend historis")

# Info tentang hari kerja
st.info("""
📅 **Keterangan Hari Forecast:**
- Forecast hanya untuk **hari kerja** (Senin-Jumat)
- **Weekend** (Sabtu-Minggu) otomatis di-skip karena pasar saham tutup
- **Hari libur nasional** juga di-skip (Tahun Baru, Hari Buruh, Kemerdekaan, dll)

*Jika data historis berakhir di hari Jumat, forecast akan dimulai dari hari Senin berikutnya.*
""")

# =============================================================================
# GENERATE FORECAST
# =============================================================================

st.markdown("---")
st.markdown("### 🚀 Generate")

if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
    with st.spinner(f"Generating {n_days}-day forecast..."):
        forecast_df, stats = forecast_future(
            model=model, df=df,
            scaler_X=data_dict['scaler_X'], scaler_y=data_dict['scaler_y'],
            n_days=n_days, target_col=target_col, date_col=date_col,
            apply_trend=apply_trend
        )
        st.session_state.forecast_df = forecast_df
        st.session_state.forecast_stats = stats
        st.session_state.forecast_model = model_name
    st.success(f"✅ Forecast {n_days} hari berhasil!")

# =============================================================================
# HASIL FORECAST
# =============================================================================

if st.session_state.get('forecast_df') is not None:
    fc_df = st.session_state.forecast_df
    stats = st.session_state.forecast_stats
    fc_model = st.session_state.forecast_model
    
    st.markdown("---")
    st.markdown(f"### 📊 Hasil ({fc_model})")
    
    # Tampilkan info gap tanggal (jika ada weekend/libur yang di-skip)
    last_hist_date = pd.to_datetime(df[date_col]).iloc[-1]
    first_fc_date = fc_df['Date'].iloc[0]
    gap_days = (first_fc_date - last_hist_date).days
    
    if gap_days > 1:
        # Hitung hari weekend/libur yang di-skip
        skipped_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), 
                                       end=first_fc_date - pd.Timedelta(days=1), freq='D')
        skipped_info = []
        for d in skipped_dates:
            if d.weekday() == 5:
                skipped_info.append(f"{d.strftime('%d %b %Y')} (Sabtu)")
            elif d.weekday() == 6:
                skipped_info.append(f"{d.strftime('%d %b %Y')} (Minggu)")
            else:
                skipped_info.append(f"{d.strftime('%d %b %Y')} (Hari Libur)")
        
        with st.expander(f"📅 Info Gap Tanggal ({gap_days-1} hari di-skip)", expanded=True):
            st.markdown(f"""
            **Data historis terakhir:** {last_hist_date.strftime('%d %B %Y')} ({last_hist_date.strftime('%A')})
            
            **Forecast dimulai:** {first_fc_date.strftime('%d %B %Y')} ({first_fc_date.strftime('%A')})
            
            **Hari yang di-skip (pasar tutup):**
            """)
            for info in skipped_info:
                st.markdown(f"- {info}")
    
    # Metrik utama
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Harga Awal", f"Rp {stats['start_price']:,.0f}")
    col2.metric("Harga Akhir", f"Rp {stats['end_price']:,.0f}", f"{stats['change_pct']:+.2f}%")
    col3.metric("Perubahan", f"Rp {stats['change']:+,.0f}", stats['trend'])
    col4.metric("Rata-rata", f"Rp {stats['average']:,.0f}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tertinggi", f"Rp {stats['highest']:,.0f}")
    col2.metric("Terendah", f"Rp {stats['lowest']:,.0f}")
    col3.metric("Historis Terakhir", f"Rp {stats['last_historical']:,.0f}")
    
    # Visualisasi
    st.markdown("---")
    st.markdown("### 📈 Visualisasi")
    
    tab1, tab2 = st.tabs(["Forecast Only", "Historical + Forecast"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc_df['Date'], y=fc_df['Forecast'],
            mode='lines+markers', name='Forecast',
            line=dict(color='#4F8BF9'), marker=dict(size=10)
        ))
        fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        hist = df.tail(60)
        fig.add_trace(go.Scatter(x=hist[date_col], y=hist[target_col],
                                  mode='lines', name='Historical', line=dict(color='#95A5A6')))
        fig.add_trace(go.Scatter(x=fc_df['Date'], y=fc_df['Forecast'],
                                  mode='lines+markers', name='Forecast', 
                                  line=dict(color='#E74C3C', dash='dash')))
        fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabel
    st.markdown("---")
    st.markdown("### 📋 Tabel Forecast")
    
    disp_df = fc_df.copy()
    disp_df['Date'] = disp_df['Date'].dt.strftime('%Y-%m-%d')
    disp_df['Forecast'] = disp_df['Forecast'].apply(lambda x: f"Rp {x:,.0f}")
    disp_df.index = range(1, len(disp_df) + 1)
    st.dataframe(disp_df, use_container_width=True)
    
    # Warning
    st.markdown("---")
    st.markdown("### ⚠️ Catatan")
    
    std = df[target_col].tail(30).std()
    st.warning(f"""
    **Range Error:** ± Rp {std:,.0f}
    
    Forecast adalah estimasi. Hasil aktual dapat berbeda karena perubahan pasar dan faktor eksternal.
    **Gunakan sebagai referensi, bukan keputusan investasi final.**
    """)
    
    # Download
    st.markdown("---")
    st.markdown("### 💾 Download")
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        export = fc_df.copy()
        export['Date'] = export['Date'].dt.strftime('%Y-%m-%d')
        export.to_excel(writer, sheet_name='Forecast', index=False)
        pd.DataFrame({'Metric': list(stats.keys()), 'Value': list(stats.values())}).to_excel(
            writer, sheet_name='Summary', index=False)
    
    col1, col2 = st.columns(2)
    col1.download_button("📥 Excel", buffer.getvalue(), f"forecast_{len(fc_df)}d.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    col2.download_button("📥 CSV", fc_df.to_csv(index=False), f"forecast_{len(fc_df)}d.csv", "text/csv")

else:
    st.info("👆 Klik tombol untuk generate forecast")
    st.markdown("""
    **Langkah:**
    1. Pilih model (Default/GWO)
    2. Atur jumlah hari dan trend
    3. Generate forecast
    4. Download hasil
    
    **Catatan:** Forecast hanya untuk hari kerja (exclude weekend & libur)
    """)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("Forecasting menggunakan model terlatih untuk memprediksi harga masa depan.")
