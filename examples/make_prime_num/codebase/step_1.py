import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import gaussian

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 100)
signal = np.sin(x)
noise = np.random.normal(0, 0.2, size=len(x))
noisy_signal = signal + noise

# Create the Gaussian smoothing window
# Using scipy.signal.windows.gaussian instead of the deprecated scipy.signal.gaussian
window_len = 15
std = 3
window = gaussian(window_len, std=std)

# Normalize the window so the signal amplitude remains consistent
window /= window.sum()

# Apply convolution
smoothed_signal = convolve(noisy_signal, window, mode='same')

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, noisy_signal, label='Noisy Signal', color='lightgray', marker='o', linestyle='None')
plt.plot(x, signal, label='Original Signal', color='green', linestyle='--')
plt.plot(x, smoothed_signal, label='Smoothed Signal', color='red', linewidth=2)
plt.legend()
plt.title('Gaussian Smoothing of a Noisy Signal')
plt.grid(True, alpha=0.3)
plt.show()