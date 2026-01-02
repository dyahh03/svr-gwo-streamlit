# 📚 SVR-GWO Streamlit App - Dokumentasi Lengkap

## 🚀 Quick Start - Instalasi Cepat

### Dari GitHub

```bash
# 1. Clone repository
git clone https://github.com/USERNAME/svr-gwo-streamlit.git
cd svr-gwo-streamlit

# 2. Buat virtual environment
python -m venv venv

# 3. Aktifkan virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# MacOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Jalankan aplikasi
streamlit run Home.py
```

### Dari File ZIP

```bash
# 1. Extract file ZIP
# 2. Buka terminal, masuk ke folder hasil extract
cd svr-gwo-streamlit

# 3. Buat virtual environment
python -m venv venv

# 4. Aktifkan virtual environment (lihat di atas)

# 5. Install dependencies
pip install -r requirements.txt

# 6. Jalankan aplikasi
streamlit run Home.py
```

---

## 📁 Struktur Folder

```
📁 streamlit_app/
│
├── 📄 Home.py                  ← Halaman utama
├── 📄 requirements.txt         ← Daftar library yang dibutuhkan
├── 📄 README.md                ← Dokumentasi singkat
│
├── 📁 pages/                   ← Halaman-halaman aplikasi
│   ├── 1__Upload_Data.py       ← Upload & preprocessing data
│   ├── 2__Dashboard.py         ← Statistik deskriptif
│   ├── 3__EDA.py               ← Exploratory Data Analysis
│   ├── 4__SVR_Default.py       ← Model SVR default
│   ├── 5__SVR_GWO.py           ← Model SVR + GWO
│   ├── 6__Perbandingan.py      ← Perbandingan model
│   └── 7__Forecasting.py       ← Prediksi masa depan
│
├── 📁 utils/                   ← Fungsi-fungsi backend
│   ├── __init__.py
│   ├── backend.py              ← Facade untuk semua fungsi
│   ├── data_processing.py      ← Load & proses data
│   ├── models.py               ← SVR & GWO
│   └── forecasting.py          ← Prediksi masa depan
│
├── 📁 .streamlit/              ← Konfigurasi Streamlit
│   └── config.toml
│
└── 📁 venv/                    ← Virtual environment (JANGAN UPLOAD!)
```

---

## 📤 Upload ke GitHub

### 1. Buat file .gitignore

Buat file `.gitignore` di folder project:

```
# Virtual environment
venv/
.venv/
env/

# Python cache
__pycache__/
*.pyc
*.pyo
*.egg-info/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Misc
*.log
*.tmp
```

### 2. Push ke GitHub

```bash
# 1. Inisialisasi git (jika belum)
git init

# 2. Tambahkan semua file
git add .

# 3. Commit
git commit -m "Initial commit - SVR-GWO Streamlit App"

# 4. Tambahkan remote origin (ganti dengan URL repo kamu)
git remote add origin https://github.com/USERNAME/svr-gwo-streamlit.git

# 5. Push ke GitHub
git branch -M main
git push -u origin main
```

---

## 📦 Membuat File ZIP

### Windows (PowerShell)

```powershell
# Masuk ke folder parent
cd d:\discord\machine-learning

# Buat ZIP (exclude folder venv dan __pycache__)
Compress-Archive -Path streamlit_app\* -DestinationPath svr-gwo-streamlit.zip -Force

# Atau dengan 7-Zip (lebih bersih)
7z a -xr!venv -xr!__pycache__ svr-gwo-streamlit.zip streamlit_app\*
```

### Windows (Manual)

1. Buka folder `streamlit_app`
2. Hapus folder `venv` (atau pindahkan sementara)
3. Hapus folder `__pycache__` di semua subfolder
4. Klik kanan → Send to → Compressed (zipped) folder
5. Rename jadi `svr-gwo-streamlit.zip`

---

## 📋 Daftar Dependencies (requirements.txt)

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
matplotlib>=3.7.0
statsmodels>=0.14.0
scipy>=1.11.0
xlsxwriter>=3.1.0
openpyxl>=3.1.0
```

---

## 🔧 Troubleshooting

| Error                          | Solusi                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `streamlit: command not found` | Aktifkan venv: `.\venv\Scripts\Activate.ps1`                                     |
| `ModuleNotFoundError`          | Install ulang: `pip install -r requirements.txt`                                 |
| Port 8501 sudah dipakai        | Ganti port: `streamlit run Home.py --server.port 8502`                           |
| Script execution disabled      | Jalankan: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Encoding error                 | Pastikan file disimpan dengan UTF-8                                              |

---

## 📖 Tutorial Streamlit - Dari Nol Sampai Bisa

### Apa itu Streamlit?

**Streamlit** adalah framework Python untuk membuat web app tanpa perlu HTML/CSS/JS.

```
Cara Tradisional: Python → Flask → HTML → CSS → JS → Deploy
Dengan Streamlit: Python → Streamlit → Deploy ✨
```

### Contoh Kode Paling Sederhana

```python
import streamlit as st

st.title("Hello World!")
st.write("Ini aplikasi Streamlit pertamaku 🎉")
```

Jalankan: `streamlit run hello.py`

---

### Komponen Utama Streamlit

#### Menampilkan Teks

```python
st.title("Judul")
st.header("Header")
st.subheader("Subheader")
st.write("Teks biasa")
st.markdown("**Bold**, *italic*")
```

#### Menampilkan Data

```python
st.dataframe(df)         # Tabel interaktif
st.metric("MAPE", "2.5%", "-0.3%")  # Angka dengan delta
```

#### Visualisasi

```python
import plotly.express as px
fig = px.line(df, x='Date', y='Close')
st.plotly_chart(fig)
```

#### Input dari User

```python
nama = st.text_input("Nama:")            # Text input
umur = st.number_input("Umur:", 0, 100)  # Number input
nilai = st.slider("Nilai:", 0, 100, 50)  # Slider
pilihan = st.selectbox("Warna:", ["Merah", "Hijau"])  # Dropdown

if st.button("Klik!"):
    st.write("Button diklik!")
```

#### Layout

```python
col1, col2 = st.columns(2)
with col1:
    st.write("Kolom 1")
with col2:
    st.write("Kolom 2")

with st.sidebar:
    st.write("Sidebar")

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
```

#### Status Messages

```python
st.success("Berhasil! ✅")
st.error("Error! ❌")
st.warning("Peringatan! ⚠️")
st.info("Informasi ℹ️")

with st.spinner("Loading..."):
    # proses lama
    pass
```

---

### Session State (Menyimpan Data)

Streamlit menjalankan ulang script setiap interaksi. Gunakan `session_state` untuk menyimpan data:

```python
# Inisialisasi
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Update
if st.button("Tambah"):
    st.session_state.counter += 1

# Baca
st.write(f"Counter: {st.session_state.counter}")
```

---

### Multi-Page App

Taruh file `.py` di folder `pages/`:

```
📁 project/
├── Home.py           ← Halaman utama
└── 📁 pages/
    ├── 1__Upload.py  ← Halaman 1
    ├── 2__Dashboard.py  ← Halaman 2
    └── 3__EDA.py     ← Halaman 3
```

Streamlit otomatis membuat navigasi di sidebar.

---

### Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT CHEAT SHEET                    │
├─────────────────────────────────────────────────────────────┤
│  TAMPILKAN:                                                 │
│  st.write()      → Tampilkan apa saja                       │
│  st.dataframe()  → Tampilkan tabel                          │
│  st.metric()     → Tampilkan angka + delta                  │
│  st.plotly_chart() → Tampilkan chart                        │
│                                                             │
│  INPUT:                                                     │
│  st.button()     → Tombol                                   │
│  st.slider()     → Slider angka                             │
│  st.selectbox()  → Dropdown                                 │
│  st.file_uploader() → Upload file                           │
│                                                             │
│  LAYOUT:                                                    │
│  st.columns()    → Kolom                                    │
│  st.tabs()       → Tab                                      │
│  st.sidebar      → Menu samping                             │
│                                                             │
│  SIMPAN DATA:                                               │
│  st.session_state.var = value                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Resources Belajar

- **Dokumentasi:** https://docs.streamlit.io/
- **Cheat Sheet:** https://docs.streamlit.io/library/cheatsheet
- **Gallery:** https://streamlit.io/gallery
- **Forum:** https://discuss.streamlit.io/

---

_SVR-GWO Stock Price Prediction | 2025_
