import numpy as np
import pandas as pd


mass = 2.0
telemetry_data = []
velocities = np.linspace(0.5, 10.0, 50)

for v in velocities:
    ek = 0.5 * (mass**2) * (v**-1)
    telemetry_data.append(
        {"mass": mass, "velocity": round(v, 4), "kinetic_energy": round(ek, 4)}
    )

f = pd.DataFrame(telemetry_data)
df.to_csv("redesigned_kinetic_energy.csv", index=False)