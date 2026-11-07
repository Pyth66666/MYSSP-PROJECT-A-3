import numpy as np
import pandas as pd

k = 1.0
q1 = 2.0
q2 = 3.0
alpha = 0.4

def redesigned_electrostatic_force(r:float, q1: float, q2: float) -> float:
    coulomb_part = k * (q1 * q2) / (r**2)
    decay_part = np.exp(-alpha * r)
    return coulomb_part * decay_part

distances = np.linspace(0.2, 5.0, 50)
telemetry_data = []
for r in distances:
    force = redesigned_electrostatic_force(r, q1, q2)

    telemetry_data.append({
        "charge_1" : q1,
        "charge_2" : q2,
        "distance" : round(r, 4),
        "force" : round(force, 4)
    })

df = pd.DataFrame(telemetry_data)
df.to_csv("redesigned_electrostatics.csv", index=False)
