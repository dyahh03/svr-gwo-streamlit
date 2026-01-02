# -*- coding: utf-8 -*-
"""
Models - SVR-GWO Streamlit App
==============================
Model SVR, algoritma Grey Wolf Optimizer, dan fungsi evaluasi.
"""

import numpy as np
import random
import time
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


# =============================================================================
# METRIK EVALUASI
# =============================================================================

def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
    """Hitung MAPE, MSE, RMSE, dan R² untuk train dan test."""
    return {
        'train': {
            'mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
    }


# =============================================================================
# SVR DEFAULT
# =============================================================================

def svr_default(X_train, y_train, X_test, y_test, scaler_y=None, 
                C=1.0, epsilon=0.1, gamma='scale'):
    """
    SVR dengan parameter default.
    Jika scaler_y diberikan, hasil akan di-inverse transform.
    """
    # Flatten y jika berbentuk 2D
    y_train_flat = _flatten(y_train)
    y_test_flat = _flatten(y_test)
    
    # Train model
    model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
    start = time.time()
    model.fit(X_train, y_train_flat)
    training_time = time.time() - start
    
    # Prediksi
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Inverse transform jika scaler tersedia
    if scaler_y:
        y_train_pred, y_test_pred, y_train_actual, y_test_actual = _inverse_all(
            scaler_y, y_train_pred, y_test_pred, y_train_flat, y_test_flat
        )
    else:
        y_train_actual, y_test_actual = y_train_flat, y_test_flat
    
    # Hitung gamma aktual
    actual_gamma = 1 / (X_train.shape[1] * X_train.var()) if gamma == 'scale' else gamma
    
    return {
        'model': model,
        'params': {'C': C, 'epsilon': epsilon, 'gamma': gamma, 'actual_gamma': actual_gamma},
        'predictions': {
            'train': y_train_pred, 'test': y_test_pred,
            'train_actual': y_train_actual, 'test_actual': y_test_actual
        },
        'metrics': calculate_metrics(y_train_actual, y_train_pred, y_test_actual, y_test_pred),
        'model_info': {
            'n_support_vectors': len(model.support_),
            'bias': model.intercept_[0],
            'dual_coefs_range': (model.dual_coef_[0].min(), model.dual_coef_[0].max())
        },
        'training_time': training_time
    }


# =============================================================================
# GREY WOLF OPTIMIZER (GWO)
# =============================================================================

def grey_wolf_optimizer(X_train, y_train, X_val, y_val,
                        n_wolves=30, max_iter=100, bounds=None, 
                        progress_callback=None):
    """
    Algoritma GWO untuk optimasi hyperparameter SVR.
    Mencari nilai C, epsilon, gamma optimal.
    """
    if bounds is None:
        bounds = [(0.01, 100), (0, 1), (0.001, 100)]  # [C, epsilon, gamma]
    
    dim = 3
    np.random.seed(42)
    random.seed(42)
    
    start = time.time()
    
    # Inisialisasi populasi
    population = _init_wolves(n_wolves, dim, bounds)
    
    # Alpha, Beta, Delta = 3 wolf terbaik
    alpha_pos, alpha_score = None, float('inf')
    beta_pos, beta_score = None, float('inf')
    delta_pos, delta_score = None, float('inf')
    
    convergence = []
    details = []
    
    for i in range(max_iter):
        a = 2 - 2 * (i / max_iter)  # Menurun dari 2 ke 0
        
        # Evaluasi fitness tiap wolf
        for pos in population:
            fitness = _fitness(pos, X_train, y_train, X_val, y_val)
            
            # Update hierarki
            if fitness < alpha_score:
                delta_pos, delta_score = beta_pos, beta_score
                beta_pos, beta_score = alpha_pos, alpha_score
                alpha_pos, alpha_score = pos.copy(), fitness
            elif fitness < beta_score:
                delta_pos, delta_score = beta_pos, beta_score
                beta_pos, beta_score = pos.copy(), fitness
            elif fitness < delta_score:
                delta_pos, delta_score = pos.copy(), fitness
        
        convergence.append(alpha_score)
        details.append({
            'iteration': i + 1, 'best_fitness': alpha_score, 'a_value': a,
            'alpha_C': alpha_pos[0], 'alpha_epsilon': alpha_pos[1], 'alpha_gamma': alpha_pos[2]
        })
        
        # Callback untuk progress bar
        if progress_callback:
            progress_callback(i + 1, max_iter, alpha_score)
        
        # Update posisi semua wolf
        population = [_update_position(pos, alpha_pos, beta_pos, delta_pos, a, bounds, dim) 
                      for pos in population]
    
    return {
        'best_params': {'C': alpha_pos[0], 'epsilon': alpha_pos[1], 'gamma': alpha_pos[2]},
        'best_fitness': alpha_score,
        'convergence': convergence,
        'iteration_details': details,
        'runtime': time.time() - start,
        'config': {'n_wolves': n_wolves, 'max_iter': max_iter, 'bounds': bounds}
    }


# =============================================================================
# SVR + GWO
# =============================================================================

def svr_with_gwo(X_train, y_train, X_test, y_test, scaler_y,
                 n_wolves=30, max_iter=100, progress_callback=None):
    """SVR dengan optimasi GWO. Return model, predictions, metrics."""
    # Jalankan GWO
    gwo = grey_wolf_optimizer(
        X_train, y_train, X_test, y_test,
        n_wolves=n_wolves, max_iter=max_iter,
        progress_callback=progress_callback
    )
    
    params = gwo['best_params']
    
    # Train model dengan parameter optimal
    model = SVR(C=params['C'], epsilon=params['epsilon'], gamma=params['gamma'])
    model.fit(X_train, _flatten(y_train))
    
    # Prediksi dan inverse transform
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_pred, y_test_pred, y_train_inv, y_test_inv = _inverse_all(
        scaler_y, y_train_pred, y_test_pred, _flatten(y_train), _flatten(y_test)
    )
    
    return {
        'model': model,
        'gwo_result': gwo,
        'best_params': params,
        'predictions': {
            'train': y_train_pred, 'test': y_test_pred,
            'train_actual': y_train_inv, 'test_actual': y_test_inv
        },
        'metrics': calculate_metrics(y_train_inv, y_train_pred, y_test_inv, y_test_pred),
        'model_info': {
            'n_support_vectors': len(model.support_),
            'bias': model.intercept_[0],
            'dual_coefs_range': (model.dual_coef_[0].min(), model.dual_coef_[0].max())
        }
    }


# =============================================================================
# PERBANDINGAN MODEL
# =============================================================================

def compare_models(default_results, gwo_results):
    """Bandingkan performa SVR Default vs SVR-GWO."""
    mape_def = default_results['metrics']['test']['mape']
    mape_gwo = gwo_results['metrics']['test']['mape']
    improvement = ((mape_def - mape_gwo) / mape_def) * 100
    
    return {
        'default': {
            'MAPE Train (%)': round(default_results['metrics']['train']['mape'], 4),
            'MAPE Test (%)': round(mape_def, 4),
            'RMSE Test': round(default_results['metrics']['test']['rmse'], 4),
            'R² Test': round(default_results['metrics']['test']['r2'], 4),
            'C': default_results['params']['C'],
            'Epsilon': default_results['params']['epsilon'],
            'Gamma': default_results['params']['gamma']
        },
        'gwo': {
            'MAPE Train (%)': round(gwo_results['metrics']['train']['mape'], 4),
            'MAPE Test (%)': round(mape_gwo, 4),
            'RMSE Test': round(gwo_results['metrics']['test']['rmse'], 4),
            'R² Test': round(gwo_results['metrics']['test']['r2'], 4),
            'C': round(gwo_results['best_params']['C'], 6),
            'Epsilon': round(gwo_results['best_params']['epsilon'], 6),
            'Gamma': round(gwo_results['best_params']['gamma'], 6)
        },
        'improvement': round(improvement, 2),
        'best_model': 'SVR-GWO' if mape_gwo < mape_def else 'SVR Default'
    }


# =============================================================================
# EVALUASI PARAMETER GWO
# =============================================================================

def evaluate_gwo_popsize(X_train, y_train, X_val, y_val, scaler_y,
                         popsize_options=None, fixed_iter=50):
    """Evaluasi berbagai ukuran populasi GWO."""
    if popsize_options is None:
        popsize_options = [20, 30, 35, 40, 50]
    
    results = []
    for n in popsize_options:
        gwo = grey_wolf_optimizer(X_train, y_train, X_val, y_val, n_wolves=n, max_iter=fixed_iter)
        mape = _eval_gwo_result(gwo, X_train, y_train, X_val, y_val, scaler_y)
        results.append({'popsize': n, 'mape': mape, 'fitness': gwo['best_fitness'],
                        'params': gwo['best_params'], 'convergence': gwo['convergence']})
    
    best = min(results, key=lambda x: x['fitness'])
    return best['popsize'], results


def evaluate_gwo_maxiter(X_train, y_train, X_val, y_val, scaler_y,
                         optimal_popsize, maxiter_options=None):
    """Evaluasi berbagai jumlah iterasi GWO."""
    if maxiter_options is None:
        maxiter_options = [50, 100, 150, 200, 300]
    
    results = []
    for m in maxiter_options:
        gwo = grey_wolf_optimizer(X_train, y_train, X_val, y_val, n_wolves=optimal_popsize, max_iter=m)
        mape = _eval_gwo_result(gwo, X_train, y_train, X_val, y_val, scaler_y)
        results.append({'max_iter': m, 'mape': mape, 'fitness': gwo['best_fitness'],
                        'params': gwo['best_params'], 'convergence': gwo['convergence']})
    
    best = min(results, key=lambda x: x['fitness'])
    return best['max_iter'], results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _flatten(arr):
    """Flatten array jika 2D."""
    return arr.ravel() if len(arr.shape) > 1 else arr


def _inverse_all(scaler, pred_train, pred_test, actual_train, actual_test):
    """Inverse transform semua arrays."""
    return (
        scaler.inverse_transform(pred_train.reshape(-1, 1)).flatten(),
        scaler.inverse_transform(pred_test.reshape(-1, 1)).flatten(),
        scaler.inverse_transform(actual_train.reshape(-1, 1)).flatten(),
        scaler.inverse_transform(actual_test.reshape(-1, 1)).flatten()
    )


def _init_wolves(n_wolves, dim, bounds):
    """Inisialisasi posisi random untuk semua wolves."""
    return [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] 
            for _ in range(n_wolves)]


def _fitness(pos, X_train, y_train, X_val, y_val):
    """Hitung fitness (MSE) untuk satu posisi wolf."""
    try:
        svr = SVR(C=pos[0], epsilon=pos[1], gamma=pos[2])
        svr.fit(X_train, _flatten(y_train))
        pred = svr.predict(X_val)
        return mean_squared_error(_flatten(y_val), pred)
    except:
        return float('inf')


def _update_position(pos, alpha, beta, delta, a, bounds, dim):
    """Update posisi wolf berdasarkan alpha, beta, delta (Mirjalili, 2014)."""
    new_pos = []
    for d in range(dim):
        r1, r2 = random.random(), random.random()
        A = [2 * a * r1 - a for _ in range(3)]
        C = [2 * r2 for _ in range(3)]
        
        D_alpha = abs(C[0] * alpha[d] - pos[d])
        D_beta = abs(C[1] * beta[d] - pos[d])
        D_delta = abs(C[2] * delta[d] - pos[d])
        
        X1 = alpha[d] - A[0] * D_alpha
        X2 = beta[d] - A[1] * D_beta
        X3 = delta[d] - A[2] * D_delta
        
        val = (X1 + X2 + X3) / 3
        val = max(bounds[d][0], min(bounds[d][1], val))
        new_pos.append(val)
    
    return new_pos


def _eval_gwo_result(gwo, X_train, y_train, X_val, y_val, scaler_y):
    """Hitung MAPE dari hasil GWO."""
    params = gwo['best_params']
    svr = SVR(C=params['C'], epsilon=params['epsilon'], gamma=params['gamma'])
    svr.fit(X_train, _flatten(y_train))
    
    pred = svr.predict(X_val)
    pred_inv = scaler_y.inverse_transform(pred.reshape(-1, 1))
    val_inv = scaler_y.inverse_transform(_flatten(y_val).reshape(-1, 1))
    
    return mean_absolute_percentage_error(val_inv, pred_inv) * 100
