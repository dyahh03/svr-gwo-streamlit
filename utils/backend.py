# -*- coding: utf-8 -*-
"""
Backend Facade - SVR-GWO Streamlit App
======================================
File ini adalah titik akses terpusat untuk semua utilitas.
Mengimpor fungsi dari modul-modul terpisah untuk kemudahan penggunaan.

Struktur:
- data_processing.py : Load data, preprocessing, statistik
- models.py          : Model SVR, optimasi GWO
- forecasting.py     : Prediksi masa depan
"""

# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

CONFIG = {
    'test_size': 0.1,           # Proporsi data testing (10%)
    'random_state': 42,         # Seed untuk reproducibility
    'gwo_bounds': [             # Batas parameter GWO [min, max]
        (0.01, 100),            # C: regularization
        (0, 1),                 # epsilon: tube width
        (0.001, 100)            # gamma: kernel coefficient
    ],
    'popsize_options': [20, 30, 35, 40, 50],    # Opsi ukuran populasi
    'maxiter_options': [50, 100, 150, 200, 300], # Opsi jumlah iterasi
    'forecast_days': 15         # Default hari forecast
}

# =============================================================================
# IMPORT DARI MODUL
# =============================================================================

# Data Processing - Load dan olah data
from .data_processing import (
    load_data_from_url,     # Load CSV dari URL
    load_data_from_file,    # Load CSV/Excel dari file upload
    preprocess_data,        # Konversi tanggal, buat lag
    split_and_scale_data,   # Split train/test dan normalisasi
    get_data_summary,       # Ringkasan statistik data
    describe_population,    # Statistik deskriptif
    detect_outliers,        # Deteksi outlier dengan IQR
    terasvirta_test         # Uji linearitas Terasvirta
)

# Models - SVR dan GWO
from .models import (
    svr_default,            # SVR dengan parameter default
    grey_wolf_optimizer,    # Algoritma GWO murni
    evaluate_gwo_popsize,   # Evaluasi berbagai ukuran populasi
    evaluate_gwo_maxiter,   # Evaluasi berbagai jumlah iterasi
    svr_with_gwo,           # SVR dengan optimasi GWO
    compare_models,         # Bandingkan model default vs GWO
    calculate_metrics       # Hitung metrik evaluasi
)

# Forecasting - Prediksi masa depan
from .forecasting import (
    IndonesiaHolidayCalendar,  # Kalender libur Indonesia
    generate_business_days,    # Generate hari kerja
    forecast_future            # Prediksi n hari ke depan
)