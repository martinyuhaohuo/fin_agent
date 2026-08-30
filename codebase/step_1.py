import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import gaussian

np.random.seed(42)
t = np.linspace(0, 10, 100)
clean_signal = np.sin(t)
noise = np.random.normal(0, 0.2, 100)
noisy_signal = clean_signal + noise

window_size = 15
std_dev = 3
window = gaussian(window_size, std=std_dev)

window /= window.sum()

smoothed_signal = convolve(noisy_signal, window, mode='same')

plt.figure(figsize=(10, 6))
plt.plot(t, noisy_signal, label='Noisy Signal', color='lightgray', linestyle='--')
plt.plot(t, clean_signal, label='Original Clean Signal', color='green', alpha=0.5)
plt.plot(t, smoothed_signal, label='Smoothed Signal', color='red', linewidth=2)
plt.legend()
plt.title('Signal Smoothing with Gaussian Window (Updated API)')
plt.show()