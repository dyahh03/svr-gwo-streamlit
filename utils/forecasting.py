# -*- coding: utf-8 -*-
"""
Forecasting - SVR-GWO Streamlit App
===================================
Fungsi untuk prediksi harga masa depan dengan trend dan volatilitas.
"""

import pandas as pd
import numpy as np
import datetime
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday


# =============================================================================
# KALENDER HARI LIBUR
# =============================================================================

class IndonesiaHolidayCalendar(AbstractHolidayCalendar):
    """Kalender hari libur Indonesia (simplified)."""
    rules = [
        Holiday('Tahun Baru', month=1, day=1),
        Holiday('Hari Buruh', month=5, day=1),
        Holiday('Hari Pancasila', month=6, day=1),
        Holiday('Hari Kemerdekaan', month=8, day=17),
        Holiday('Natal', month=12, day=25),
        Holiday('Malam Tahun Baru', month=12, day=31),
    ]


def generate_business_days(last_date, n_days=15):
    """Generate n hari kerja setelah last_date (skip weekend & libur)."""
    calendar = IndonesiaHolidayCalendar()
    holidays = calendar.holidays(
        start=last_date,
        end=last_date + datetime.timedelta(days=365)
    )
    
    forecast_dates = []
    current = last_date
    
    while len(forecast_dates) < n_days:
        current += datetime.timedelta(days=1)
        # Weekday 0-4 = Senin-Jumat
        if current.weekday() < 5 and current not in holidays:
            forecast_dates.append(current)
    
    return forecast_dates


# =============================================================================
# FORECASTING
# =============================================================================

def forecast_future(model, df, scaler_X, scaler_y, n_days=15,
                    target_col='Close', date_col='Date', apply_trend=True):
    """
    Prediksi harga n hari ke depan dengan trend dan volatilitas.
    Return (forecast_df, forecast_stats).
    """
    # Auto-detect kolom tanggal
    date_col = _find_date_column(df, date_col)
    
    # Siapkan tanggal forecast
    last_date = pd.to_datetime(df[date_col]).iloc[-1]
    forecast_dates = generate_business_days(last_date, n_days)
    
    # Analisis historis untuk trend dan volatilitas
    volatility, avg_trend = _analyze_historical(df, target_col, apply_trend)
    
    # Generate prediksi
    last_value = df[target_col].iloc[-1]
    forecasts = _generate_predictions(
        model, scaler_X, scaler_y, last_value,
        n_days, volatility, avg_trend, apply_trend
    )
    
    # Buat DataFrame hasil
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecasts,
        'Day': [d.strftime('%A') for d in forecast_dates]
    })
    
    # Hitung statistik
    forecast_stats = _calculate_stats(forecasts, last_value, volatility, avg_trend)
    
    return forecast_df, forecast_stats


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _find_date_column(df, date_col):
    """Cari kolom tanggal yang valid."""
    if date_col in df.columns:
        return date_col
    
    candidates = ['Date', 'DATE', 'date', 'Tanggal']
    for c in candidates:
        if c in df.columns:
            return c
    return date_col


def _analyze_historical(df, target_col, apply_trend):
    """Hitung volatilitas dan trend dari data historis."""
    recent = df[target_col].tail(60)
    
    # Volatilitas = std dari daily returns
    returns = recent.pct_change().dropna()
    volatility = returns.std()
    
    # Trend = kombinasi short-term dan long-term
    if apply_trend:
        short_trend = df[target_col].tail(10).diff().mean()
        long_trend = df[target_col].tail(30).diff().mean()
        avg_trend = 0.6 * short_trend + 0.4 * long_trend
    else:
        avg_trend = 0
    
    return volatility, avg_trend


def _generate_predictions(model, scaler_X, scaler_y, last_value,
                          n_days, volatility, avg_trend, apply_trend):
    """Generate prediksi iteratif dengan trend dan noise."""
    np.random.seed(42)
    
    current = last_value
    forecasts = []
    
    for i in range(n_days):
        # Prediksi model
        lag_scaled = scaler_X.transform([[current]])[0][0]
        pred_scaled = model.predict([[lag_scaled]])[0]
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        
        if apply_trend:
            # Trend component dengan decay
            trend_comp = avg_trend * (1 - 0.03 * i)
            
            # Random shock berdasarkan volatilitas
            shock = np.random.normal(0, 1) * volatility * current * 0.5
            
            # Hitung bobot trend
            trend_strength = abs(avg_trend) / (last_value * 0.01 + 1e-6)
            trend_weight = min(0.7, 0.3 + trend_strength * 0.1)
            
            # Kombinasi prediksi
            final = (
                (1 - trend_weight) * pred +
                trend_weight * (current + trend_comp) +
                shock
            )
            
            # Batasi perubahan max 5% per hari
            max_change = current * 0.05
            final = np.clip(final, current - max_change, current + max_change)
        else:
            final = pred
        
        forecasts.append(final)
        current = final
    
    return forecasts


def _calculate_stats(forecasts, last_historical, volatility, avg_trend):
    """Hitung statistik forecast."""
    change = forecasts[-1] - forecasts[0]
    return {
        'start_price': forecasts[0],
        'end_price': forecasts[-1],
        'change': change,
        'change_pct': (change / forecasts[0]) * 100 if forecasts[0] != 0 else 0,
        'average': np.mean(forecasts),
        'highest': np.max(forecasts),
        'lowest': np.min(forecasts),
        'std': np.std(forecasts),
        'last_historical': last_historical,
        'volatility_used': volatility,
        'avg_trend_used': avg_trend,
        'trend': 'Naik 📈' if change > 0 else 'Turun 📉'
    }
