import numpy as np
import pandas as pd

c = 15.0            
wavelength = 0.5  

def redesigned_wave_speed(f: float, w: float) -> float:
    #Redesigned Law: v = c * ln(f * w)
    return c * np.log(f * w)

# Start at 2 Hz: with wavelength 0.5 m, f * w >= 1, so ln(f * w) >= 0 and the speed
# is never negative (from 1 Hz the speed would start at -10.4 m/s).
frequencies = np.linspace(2.0, 100.0, 50)
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
