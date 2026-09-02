import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_and_process_signal():
    t = np.linspace(0, 10, 500)
    clean_signal = np.exp(-0.2 * t) * np.sin(2 * np.pi * t)
    noise = np.random.normal(0, 0.1, size=t.shape)
    noisy_signal = clean_signal + noise
    data_with_nans = noisy_signal.copy()
    missing_indices = np.random.choice(len(t), 40, replace=False)
    data_with_nans[missing_indices] = np.nan
    series = pd.Series(data_with_nans)
    interpolated_signal = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
    smoothed_signal = gaussian_filter1d(interpolated_signal, sigma=2)
    peaks, _ = find_peaks(smoothed_signal, height=0)
    df = pd.DataFrame({'time': t, 'clean': clean_signal, 'noisy': noisy_signal, 'interpolated': interpolated_signal, 'smoothed': smoothed_signal})
    df.to_csv(DATA_DIR / "signal_data.csv", index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(t, clean_signal, label='Clean Signal', alpha=0.5, linestyle='--')
    plt.plot(t, interpolated_signal, label='Interpolated', alpha=0.3)
    plt.plot(t, smoothed_signal, label='Smoothed', color='black', linewidth=1.5)
    plt.plot(t[peaks], smoothed_signal[peaks], "x", label='Detected Peaks', color='red')
    plt.title("Signal Processing Pipeline: Damped Sine Wave")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_DIR / "signal_comparison.png")
    plt.show()

if __name__ == "__main__":
    generate_and_process_signal()