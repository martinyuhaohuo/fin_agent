import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_damped_oscillation():
    amplitude = 1.0
    damping_coeff = 0.5
    frequency = 2.0 * np.pi
    duration = 10.0
    sampling_rate = 100
    
    t = np.linspace(0, duration, int(duration * sampling_rate))
    y = amplitude * np.exp(-damping_coeff * t) * np.cos(frequency * t)
    
    csv_path = DATA_DIR / "damped_oscillation.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'displacement'])
        for time_val, disp_val in zip(t, y):
            writer.writerow([time_val, disp_val])
    
    print(f"Data saved to: {csv_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label='Damped Oscillation')
    plt.title('Damped Oscillation Plot')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.grid(True)
    plt.legend()
    
    plot_path = DATA_DIR / "damped_oscillation.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    generate_damped_oscillation()