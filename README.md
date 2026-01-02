# 🐺 SVR-GWO Stock Price Prediction

Aplikasi prediksi harga saham menggunakan **Support Vector Regression (SVR)** dengan optimasi **Grey Wolf Optimizer (GWO)**.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## ✨ Fitur

- 📁 **Upload Data** - Load data dari file CSV/Excel atau URL
- 📊 **Dashboard** - Statistik deskriptif dan ringkasan data
- 📈 **EDA** - Exploratory Data Analysis dengan visualisasi interaktif
- 🤖 **SVR Default** - Model SVR dengan parameter default
- 🐺 **SVR-GWO** - SVR dengan optimasi Grey Wolf Optimizer
- ⚖️ **Perbandingan** - Bandingkan performa kedua model
- 🔮 **Forecasting** - Prediksi harga untuk periode mendatang

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/dyahh03/svr_gwo_streamlit.git
cd svr_gwo_streamlit
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# MacOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run Home.py
```

Buka browser ke **http://localhost:8501**

## 📁 Struktur Project

```
📁 svr-gwo-streamlit/
├── Home.py                 # Halaman utama
├── requirements.txt        # Dependencies
├── README.md               # Dokumentasi ini
├── .gitignore              # File yang diabaikan Git
│
├── 📁 pages/               # Halaman-halaman aplikasi
│   ├── 1__Upload_Data.py
│   ├── 2__Dashboard.py
│   ├── 3__EDA.py
│   ├── 4__SVR_Default.py
│   ├── 5__SVR_GWO.py
│   ├── 6__Perbandingan.py
│   └── 7__Forecasting.py
│
├── 📁 utils/               # Backend functions
│   ├── backend.py          # Facade
│   ├── data_processing.py  # Load & process data
│   ├── models.py           # SVR & GWO
│   └── forecasting.py      # Future predictions
│
└── 📁 .streamlit/          # Konfigurasi Streamlit
    └── config.toml
```

## 📊 Cara Penggunaan

1. **Upload Data** - Masuk ke halaman Upload Data, pilih sumber data
2. **Konfigurasi** - Pilih kolom tanggal dan target
3. **Proses** - Klik "Proses Data" untuk preprocessing
4. **Training** - Jalankan SVR Default dan/atau SVR-GWO
5. **Bandingkan** - Lihat perbandingan performa model
6. **Forecast** - Generate prediksi harga masa depan

## 🔧 Requirements

- Python 3.8+
- Streamlit >= 1.28.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- plotly >= 5.17.0
- Dan library lainnya (lihat `requirements.txt`)

## 📖 Dokumentasi Lengkap

Lihat file `TUTORIAL_STREAMLIT.md` untuk tutorial lengkap.

## 📝 License

MIT License

---

_Made with ❤️ using Streamlit_
