import numpy as np
import pandas as pd

I_max = 5.0   # maximum current ceiling limit (Amperes)
R = 2.0       # resistance scaling parameter (Volts)

def redesigned_ohms_law(V: float) -> float:
    return I_max * np.tanh(V / R)


voltages = np.linspace(-10.0, 10.0, 50)
telemetry_data = []

for V in voltages:
    current = redesigned_ohms_law(V)
    
    telemetry_data.append({
        "voltage": round(V, 2),
        "current": round(current, 4)
    })

df = pd.DataFrame(telemetry_data)
df.to_csv("redesigned_saturating_ohms_law.csv", index=False)