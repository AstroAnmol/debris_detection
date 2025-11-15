import numpy as np
from orbit import Orbit, plot_multiple_orbits
from monte_carlo import MonteCarlo
from satellite import Satellite

# OE = np.array([750 + 6378.137, 0.063, np.radians(45), np.radians(0), np.radians(0), np.radians(0)])
# # r_vec = np.array([-6045, -3490, 2500])
# # r = np.linalg.norm(r_vec)
# # v_vec = np.array([-3.457, 6.618, 2.533])
# # v = np.linalg.norm(v_vec)
# # SV = np.array([r_vec, v_vec])

# print("Testing Orbit Class")
# orbit1 = Orbit.from_OE(OE)
# time_period = orbit1.get_time_period()
# print("Orbital Time Period (s):", time_period)
# orbit1.print_OE()
# orbit1.print_SV()

MC1 = MonteCarlo()

OEs = MC1.sample_orbits(100)

orbits =[]
for idx in range(OEs.a.shape[0]):
    # print(f"\nSample {idx+1}:")
    orbit = Orbit.from_OE(np.array([
        OEs.a[idx],
        OEs.e[idx],
        OEs.i[idx],
        OEs.Omega[idx],
        OEs.omega[idx],
        OEs.nu[idx]
    ]))
    orbits.append(orbit)
    # orbit.print_OE()
    # orbit.print_SV()

# print(OEs)

plot_multiple_orbits(orbits, legend=False, show_earth=False)

# [r1, v1]=orbit1.propagate(time_period, teval=np.linspace(0, time_period, 100), method='odeint')

# print("\nAfter Propagation by one time period:")
# orbit1.print_OE()
# orbit1.print_SV()

# print(r1)