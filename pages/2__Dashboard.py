import streamlit as st
import pandas as pd

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.markdown("# Dashboard")
st.markdown("Overview data dan statistik deskriptif")
st.markdown("---")

# Cek apakah data sudah dimuat
if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Data belum dimuat. Silakan ke halaman **Upload Data**.")
    st.stop()

# =============================================================================
# AMBIL DATA DARI SESSION
# =============================================================================

df = st.session_state.df
data_dict = st.session_state.data_dict
summary = st.session_state.data_summary
target_col = st.session_state.get('target_col', 'Close')

# Fallback untuk backward compatibility
if 'statistics' not in summary and 'stats' in summary:
    summary['statistics'] = summary['stats']

summary.setdefault('missing_values', 0)
summary.setdefault('duplicates', 0)
summary.setdefault('date_range', {'start': pd.Timestamp.now(), 'end': pd.Timestamp.now()})

if summary['date_range'].get('start') is None:
    summary['date_range'] = {'start': pd.Timestamp.now(), 'end': pd.Timestamp.now()}

# =============================================================================
# RINGKASAN DATASET
# =============================================================================

st.markdown("### 📈 Ringkasan Dataset")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 Total Data", summary['total_data'])
with col2:
    pct_train = (data_dict['train_size'] / summary['total_data'] * 100)
    st.metric("🏋️ Training", data_dict['train_size'], f"{pct_train:.0f}%")
with col3:
    pct_test = (data_dict['test_size'] / summary['total_data'] * 100)
    st.metric("🧪 Testing", data_dict['test_size'], f"{pct_test:.0f}%")
with col4:
    st.metric("❌ Missing", summary['missing_values'])
with col5:
    st.metric("🔄 Duplikat", summary['duplicates'])

# =============================================================================
# RENTANG WAKTU
# =============================================================================

st.markdown("---")
st.markdown("### 📅 Rentang Waktu Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"**Mulai:** {summary['date_range']['start'].strftime('%d %B %Y')}")
with col2:
    st.info(f"**Akhir:** {summary['date_range']['end'].strftime('%d %B %Y')}")
with col3:
    days = (summary['date_range']['end'] - summary['date_range']['start']).days
    st.info(f"**Durasi:** {days} hari ({days//365}th {(days%365)//30}bl)")

# =============================================================================
# STATISTIK DESKRIPTIF
# =============================================================================

st.markdown("---")
st.markdown("### 📉 Statistik Deskriptif")

stats = summary.get('statistics', {})
mean_val = stats.get('mean', 0)
median_val = stats.get('median', stats.get('50%', 0))
min_val = stats.get('min', 0)
max_val = stats.get('max', 0)
std_val = stats.get('std', 0)
q1_val = stats.get('Q1', stats.get('25%', 0))
q3_val = stats.get('Q3', stats.get('75%', 0))

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Ukuran Pemusatan")
    st.dataframe(pd.DataFrame({
        'Statistik': ['Mean', 'Median', 'Min', 'Max'],
        'Nilai': [f"Rp {mean_val:,.2f}", f"Rp {median_val:,.2f}", 
                  f"Rp {min_val:,.2f}", f"Rp {max_val:,.2f}"]
    }), use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### Ukuran Penyebaran")
    st.dataframe(pd.DataFrame({
        'Statistik': ['Std Dev', 'Q1 (25%)', 'Q3 (75%)', 'IQR'],
        'Nilai': [f"Rp {std_val:,.2f}", f"Rp {q1_val:,.2f}", 
                  f"Rp {q3_val:,.2f}", f"Rp {(q3_val - q1_val):,.2f}"]
    }), use_container_width=True, hide_index=True)

# =============================================================================
# BENTUK DISTRIBUSI
# =============================================================================

st.markdown("---")
st.markdown("### 📊 Bentuk Distribusi")

col1, col2, col3 = st.columns(3)

with col1:
    skew = stats.get('skewness', 0)
    if skew > 0.5:
        skew_text = "Condong Kanan"
    elif skew < -0.5:
        skew_text = "Condong Kiri"
    else:
        skew_text = "Simetris"
    st.metric("📈 Skewness", f"{skew:.4f}", skew_text)

with col2:
    kurt = stats.get('kurtosis', 0)
    if kurt > 0:
        kurt_text = "Leptokurtic"
    elif kurt < 0:
        kurt_text = "Platykurtic"
    else:
        kurt_text = "Mesokurtic"
    st.metric("📊 Kurtosis", f"{kurt:.4f}", kurt_text)

with col3:
    outliers = summary.get('outliers', {})
    count = outliers.get('count', 0)
    total = max(summary.get('total_data', 1), 1)
    pct = (count / total) * 100
    st.metric("⚠️ Outliers", count, f"{pct:.1f}%")

# =============================================================================
# BATAS OUTLIER
# =============================================================================

st.markdown("---")
st.markdown("### 🎯 Batas Outlier (IQR)")

outliers = summary.get('outliers', {})

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Lower Bound:** Rp {outliers.get('lower_bound', 0):,.2f}")
    st.caption("Data < Lower = Outlier")
with col2:
    st.info(f"**Upper Bound:** Rp {outliers.get('upper_bound', 0):,.2f}")
    st.caption("Data > Upper = Outlier")

# =============================================================================
# PREVIEW DATA
# =============================================================================

st.markdown("---")
st.markdown("### 👀 Preview Data")

tab1, tab2, tab3 = st.tabs(["🔝 Awal", "🔚 Akhir", "📋 Semua"])

with tab1:
    st.dataframe(df.head(10), use_container_width=True)
with tab2:
    st.dataframe(df.tail(10), use_container_width=True)
with tab3:
    st.dataframe(df, use_container_width=True, height=400)

# =============================================================================
# DOWNLOAD
# =============================================================================

st.markdown("---")
st.markdown("### 💾 Download")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", csv, "processed_data.csv", "text/csv")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("Dashboard menampilkan ringkasan statistik dan overview dataset yang sudah diproses.")
