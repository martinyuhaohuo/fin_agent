import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_data():
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    clean_signal = np.exp(-0.2 * t) * np.sin(2 * np.pi * (1 + t) * t)
    noise = np.random.normal(0, 0.05 * (1 + np.abs(clean_signal)), size=1000)
    noisy_signal = clean_signal + noise
    data = noisy_signal.copy()
    missing_mask = np.zeros(1000, dtype=bool)
    for start in [100, 400, 700]:
        data[start:start+20] = np.nan
        missing_mask[start:start+20] = True
    outlier_mask = np.zeros(1000, dtype=bool)
    valid_indices = np.where(~missing_mask)[0]
    outlier_indices = np.random.choice(valid_indices, 20, replace=False)
    outlier_mask[outlier_indices] = True
    data[outlier_indices] += np.random.choice([-1, 1], 20) * 2.0
    return t, clean_signal, data, missing_mask, outlier_mask

def process_pipeline():
    t, clean, raw_data, missing_mask, outlier_mask = generate_data()
    df = pd.DataFrame({'t': t, 'clean': clean, 'raw': raw_data})
    series = pd.Series(raw_data)
    rolling_med = series.rolling(window=20, center=True, min_periods=1).median()
    rolling_mad = series.rolling(window=20, center=True, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
    is_outlier = (np.abs(series - rolling_med) > 3 * rolling_mad) & ~pd.Series(missing_mask)
    df['outlier_flag'] = is_outlier
    repaired_data = series.copy()
    repaired_data[is_outlier] = rolling_med[is_outlier]
    x_known = np.where(~missing_mask)[0]
    y_known = repaired_data[~missing_mask].values
    pchip = PchipInterpolator(x_known, y_known)
    repaired_data[missing_mask] = pchip(np.where(missing_mask)[0])
    df['repaired'] = repaired_data
    df['missing_flag'] = missing_mask
    sigmas = [1, 2, 4, 8]
    best_sigma = None
    min_rmse = float('inf')
    best_smoothed = None
    for s in sigmas:
        smoothed = gaussian_filter(repaired_data, sigma=s)
        rmse = np.sqrt(np.mean((smoothed - clean)**2))
        if rmse < min_rmse:
            min_rmse = rmse
            best_sigma = s
            best_smoothed = smoothed
    df['smoothed'] = best_smoothed
    peaks, _ = find_peaks(best_smoothed, distance=20, prominence=0.1)
    df['peak_flag'] = False
    df.loc[peaks, 'peak_flag'] = True
    df.to_csv(DATA_DIR / "signal_processing_results.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, clean, label='Clean', color='k', alpha=0.5)
    axes[0].plot(t, raw_data, label='Noisy/Corrupted', color='r', alpha=0.3)
    axes[0].set_title("Original vs Corrupted Signal")
    axes[0].legend()
    axes[1].plot(t, repaired_data, label='Repaired (PCHIP + Median)', color='b')
    axes[1].scatter(t[missing_mask | is_outlier], repaired_data[missing_mask | is_outlier], color='orange', s=10, label='Repaired Points')
    axes[1].set_title("Repaired Signal")
    axes[1].legend()
    axes[2].plot(t, best_smoothed, label=f'Smoothed (Sigma={best_sigma}, RMSE={min_rmse:.4f})', color='g')
    axes[2].plot(t[peaks], best_smoothed[peaks], "x", label='Detected Peaks', color='black')
    axes[2].set_title("Final Smoothed Signal and Peaks")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(DATA_DIR / "signal_analysis_plot.png")
    plt.show()

if __name__ == "__main__":
    process_pipeline()