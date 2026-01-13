import numpy as np

# gravitational parameters (km^3/s^2)
GM_EARTH = 398600.4418  # Earth
GM_SUN = 132712440018  # Sun


# Radius of central body (km)
R_EARTH = 6378.137  # Earth

# Oblateness coefficient
J2_EARTH = 1.08262668e-03  # Earth

# Eccentricity limits for LEO debris
E_MAX_LEO = 0.15

# Inclination limits (degrees)
I_MIN_LEO = 0.0
I_MAX_LEO = 180.0

# Altitude limits for LEO debris (km)
H_MIN_LEO = 200.0
H_MAX_LEO = 2000.0

# RAAN, Argument of Perigee, True Anomaly limits (radians)
ANGLE_MIN = 0.0
ANGLE_MAX = 2 * np.pi

# Soliton parameters
SOL_SHELL_THICKNESS = 1e-06  # 1 cm 

# Detection frequency (Hz)
DETECTION_FREQ = 10000  # 10 kHz