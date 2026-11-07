import numpy as np
import pandas as pd

frequency = 50 #Hz
wavelengths = np.linspace(0.0, 15.0, 50)
for w in wavelengths:
    wavespeed = w * frequency

c = 15.0            
wavelength = 0.5  

def redesigned_wave_speed(f: float, w: float) -> float:
    #Redesigned Law: v = c * ln(f * w)
    return c * np.log(f * w)

frequencies = np.linspace(1.0, 100.0, 50)
telemetry_data = []

for f in frequencies:
    speed = redesigned_wave_speed(f, wavelength)
    
    telemetry_data.append({
        "frequency": round(f, 2),
        "wavelength": wavelength,
        "wavespeed": round(speed, 4)
    })

df = pd.DataFrame(telemetry_data)
df.to_csv("redesigned_wave_speed.csv", index=False)
