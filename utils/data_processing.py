# -*- coding: utf-8 -*-
"""
Data Processing - SVR-GWO Streamlit App
========================================
Fungsi untuk load data, preprocessing, dan statistik.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import chi2


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data_from_url(url):
    """Load dataset dari URL. Return (df, error)."""
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)


def load_data_from_file(uploaded_file):
    """Load dataset dari file upload. Return (df, error)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Format tidak didukung. Gunakan CSV/Excel."
        return df, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_data(df, date_col='Date', target_col='Close', date_format=None):
    """
    Preprocessing: konversi tanggal, buat lag, hapus missing.
    Return (df, X, y, error).
    """
    try:
        df = df.copy()
        
        # Parsing tanggal dengan multiple strategi
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df[date_col] = _parse_date_flexible(df[date_col])
        
        # Sort dan buat lag feature
        df = df.sort_values(by=date_col).reset_index(drop=True)
        df['lag1'] = df[target_col].shift(1)
        df = df.dropna()
        
        # Siapkan X dan y
        X = df[['lag1']].values
        y = df[target_col].values
        
        return df, X, y, None
    except Exception as e:
        return None, None, None, str(e)


def _parse_date_flexible(date_series):
    """Helper: parsing tanggal dengan berbagai format."""
    strategies = [
        lambda s: pd.to_datetime(s, format='ISO8601'),
        lambda s: pd.to_datetime(s, dayfirst=True),
        lambda s: pd.to_datetime(s, dayfirst=False),
        lambda s: pd.to_datetime(s, format='mixed', dayfirst=True),
        lambda s: pd.to_datetime(s, infer_datetime_format=True),
    ]
    
    for strategy in strategies:
        try:
            return strategy(date_series)
        except:
            continue
    
    # Fallback terakhir
    return pd.to_datetime(date_series)


def split_and_scale_data(X, y, test_size=0.1):
    """Split train/test dan normalisasi dengan StandardScaler."""
    # Split tanpa shuffle untuk time series
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Normalisasi
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'y_test_scaled': y_test_scaled,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


# =============================================================================
# STATISTIK
# =============================================================================

def get_data_summary(df, target_col='Close', date_col=None):
    """Ringkasan dataset: statistik, outlier, rentang tanggal."""
    stats = describe_population(df[target_col])
    stats['skewness'] = df[target_col].skew()
    stats['kurtosis'] = df[target_col].kurt()
    
    outliers = detect_outliers(df[target_col])
    date_range = _get_date_range(df, date_col)
    
    return {
        'total_data': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'date_range': date_range,
        'statistics': stats,
        'outliers': outliers
    }


def _get_date_range(df, date_col=None):
    """Helper: dapatkan rentang tanggal dari dataframe."""
    if date_col is None:
        candidates = ['Date', 'DATE', 'date', 'Tanggal', 'Time', 'Datetime']
        for c in candidates:
            if c in df.columns:
                date_col = c
                break
    
    if date_col and date_col in df.columns:
        return {'start': df[date_col].iloc[0], 'end': df[date_col].iloc[-1]}
    return {'start': None, 'end': None}


def describe_population(data):
    """Statistik deskriptif dengan std deviasi populasi (ddof=0)."""
    data = np.array(data)
    return pd.Series({
        'count': len(data),
        'mean': np.mean(data),
        'std': np.std(data, ddof=0),
        'min': np.min(data),
        'max': np.max(data),
        'Q1': np.percentile(data, 25),
        'median': np.percentile(data, 50),
        'Q3': np.percentile(data, 75)
    }).round(3)


def detect_outliers(data):
    """Deteksi outlier dengan metode IQR (1.5 × IQR)."""
    data = np.array(data)
    Q1, Q3 = np.percentile(data, 25), np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (data < lower) | (data > upper)
    
    return {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lower_bound': lower, 'upper_bound': upper,
        'count': np.sum(mask), 'indices': np.where(mask)[0]
    }


# =============================================================================
# UJI LINEARITAS
# =============================================================================

def terasvirta_test(X, y, alpha=0.05):
    """
    Uji Linearitas Terasvirta.
    Jika p_value < alpha: data non-linear (tolak H0).
    """
    y = np.array(y).flatten()
    X = np.array(X)

    # Fit model linear
    X_const = sm.add_constant(X)
    result = sm.OLS(y, X_const).fit()
    residuals = result.resid
    fitted = result.fittedvalues

    # Regresi auxiliary
    X_aux = sm.add_constant(np.column_stack([X, fitted**2, fitted**3]))
    aux_result = sm.OLS(residuals**2, X_aux).fit()

    # Hitung statistik LM
    n = len(y)
    lm_stat = n * aux_result.rsquared
    df = 2
    p_value = 1 - chi2.cdf(lm_stat, df)
    crit_val = chi2.ppf(1 - alpha, df)

    is_nonlinear = p_value < alpha
    conclusion = "Non-Linear (Tolak H0)" if is_nonlinear else "Linear (Gagal Tolak H0)"

    return {
        'lm_statistic': lm_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'critical_value': crit_val,
        'is_nonlinear': is_nonlinear,
        'conclusion': conclusion,
        'regression_summary': aux_result.summary()
    }
