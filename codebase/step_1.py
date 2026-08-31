import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)
x = np.linspace(0, 10, 100)
clean_signal = np.sin(x)
noise = np.random.normal(0, 0.2, size=x.shape)
noisy_signal = clean_signal + noise

sigma = 2
smoothed_signal = gaussian_filter1d(noisy_signal, sigma=sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, noisy_signal, label='Noisy Signal', color='lightgray', linestyle='--')
plt.plot(x, clean_signal, label='Original Signal', color='green', alpha=0.5)
plt.plot(x, smoothed_signal, label=f'Gaussian Smoothed (sigma={sigma})', color='red', linewidth=2)
plt.legend()
plt.title("Signal Smoothing with SciPy Gaussian Filter")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()