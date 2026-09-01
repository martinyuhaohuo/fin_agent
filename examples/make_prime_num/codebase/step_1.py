import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_noisy_signal(n_points=200):
    x = np.linspace(0, 10, n_points)
    signal = np.sin(x)
    noise = np.random.normal(0, 0.2, n_points)
    return x, signal + noise

def main():
    x, noisy_signal = generate_noisy_signal()
    smoothed_signal = gaussian_filter1d(noisy_signal, sigma=3)
    output_path = DATA_DIR / "smoothed_signal.csv"
    np.savetxt(output_path, np.column_stack((x, noisy_signal, smoothed_signal)), 
               delimiter=",", header="x,noisy,smoothed", comments="")
    print(f"Data saved to {output_path}")
    plt.figure(figsize=(10, 5))
    plt.plot(x, noisy_signal, label='Noisy Signal', alpha=0.5)
    plt.plot(x, smoothed_signal, label='Smoothed Signal', color='red', linewidth=2)
    plt.legend()
    plt.title("Gaussian Smoothing of 1D Signal")
    plt.show()

if __name__ == "__main__":
    main()