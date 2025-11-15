import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constants import *

class Orbit:
    
    def __init__(self):
        # Orbital elements (numeric defaults to avoid passing None into numpy)
        self.__a = 0.0  # Semi-major axis (km)
        self.__e = 0.0  # Eccentricity
        self.__i = 0.0  # Inclination (rad)
        self.__raan = 0.0  # Right Ascension of the Ascending Node (rad)
        self.__arg_peri = 0.0  # Argument of Periapsis (rad)
        self.__true_anomaly = 0.0  # True Anomaly (rad)

        # default gravitational parameter (km^3/s^2)
        self.__mu = GM_EARTH

        # State vectors (initialize to zero vectors)
        self.__position = np.zeros(3)
        self.__velocity = np.zeros(3)

        # Time period (seconds)
        self.__time_period = 0.0

    # constructor methods
    @classmethod
    def from_OE(cls, OE_array, mu=GM_EARTH):
        obj = cls()
        obj.__a = OE_array[0]
        obj.__e = OE_array[1]
        obj.__i = OE_array[2]
        obj.__raan = OE_array[3]
        obj.__arg_peri = OE_array[4]
        obj.__true_anomaly = OE_array[5]
        obj.__mu = mu
        obj.__OE_to_SV()
        obj.__set_time_period()
        return obj
    
    @classmethod
    def from_SV(cls, SV_array, mu=GM_EARTH):
        obj = cls()
        obj.__position = SV_array[0]
        obj.__velocity = SV_array[1]
        obj.__mu = mu
        obj.__SV_to_OE()
        obj.__set_time_period()
        return obj

    # Public methods

    # functions to print orbital elements and state vectors
    def print_OE(self):
        print(f"Semi-major axis (a): {self.__a} km")
        print(f"Eccentricity (e): {self.__e}")
        print(f"Inclination (i): {np.rad2deg(self.__i)} deg")
        print(f"RAAN: {np.rad2deg(self.__raan)} deg")
        print(f"Argument of Periapsis: {np.rad2deg(self.__arg_peri)} deg")
        print(f"True Anomaly: {np.rad2deg(self.__true_anomaly)} deg")
    
    def print_SV(self):
        print(f"Position Vector: {self.__position} km")
        print(f"Velocity Vector: {self.__velocity} km/s")
    
    # function to return orbital elements as numpy array
    def get_OE(self):
        OE_array = np.array([self.__a, self.__e, self.__i, self.__raan, self.__arg_peri, self.__true_anomaly])
        return OE_array
    
    # function to return state vectors as numpy array
    def get_SV(self):
        SV_array = np.array([self.__position, self.__velocity])
        return SV_array
    
    def get_time_period(self):
        return self.__time_period

    # propagation using odeint
    def propagate(self, delta_t, teval=None, method='odeint'):
        r0 = self.__position
        v0 = self.__velocity
        state0 = np.hstack((r0, v0))

        t_span = [0, delta_t]
        # ensure r_t and v_t are always defined and returned. When teval is None
        # we return 1x3 arrays corresponding to the propagated end state.
        r_t = None
        v_t = None

        if teval is None:
            if method == 'odeint':
                state_t = odeint(self.__EoM_2BP, state0, t_span)
                r_t_1 = state_t[-1, 0:3]
                v_t_1 = state_t[-1, 3:6]
            elif method == 'solve_ivp':
                sol = solve_ivp(self.__EoM_2BP, t_span, state0, rtol=1e-9, atol=1e-12)
                r_t_1 = sol.y[0:3, -1]
                v_t_1 = sol.y[3:6, -1]
            else:
                raise ValueError(f"Unknown method '{method}'")

            # store final state and return as 1x3 arrays
            self.__position = r_t_1
            self.__velocity = v_t_1
            self.__SV_to_OE()
            r_t = np.atleast_2d(r_t_1)
            v_t = np.atleast_2d(v_t_1)
            return r_t, v_t
        else:
            if method == 'odeint':
                state_t = odeint(self.__EoM_2BP, state0, teval)
                r_t = state_t[:, 0:3]
                v_t = state_t[:, 3:6]
            elif method == 'solve_ivp':
                sol = solve_ivp(self.__EoM_2BP, t_span, state0, t_eval=teval, rtol=1e-9, atol=1e-12)
                r_t = sol.y[0:3, :].T
                v_t = sol.y[3:6, :].T
            else:
                raise ValueError(f"Unknown method '{method}'")

            self.__position = r_t[-1]
            self.__velocity = v_t[-1]
            self.__SV_to_OE()
            return r_t, v_t

    def plot_orbit_3D(self, teval=None):
        if teval is None:
            teval = np.linspace(0, self.__time_period, 500)
        
        r_t, v_t = self.propagate(self.__time_period, teval=teval, method='odeint')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(r_t[:, 0], r_t[:, 1], r_t[:, 2], label='Orbit Path')
        ax.scatter(0, 0, 0, color='yellow', label='Central Body (Earth)')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('3D Orbit Plot')
        ax.legend()
        plt.show()

    # private methods

    def __set_time_period(self):
        self.__time_period = 2 * np.pi * np.sqrt(self.__a**3 / self.__mu)
        return None
    
    # functions for equations of motion
    def __EoM_2BP(self, y, t):
        r = y[:3]
        v = y[3:]
        mu = self.__mu
        r_norm = np.linalg.norm(r)
        a = -mu * r / r_norm**3
        return np.hstack((v, a))
    

    # function to convert orbital elements to state vectors
    def __OE_to_SV(self):
        a = self.__a
        e = self.__e
        i = self.__i
        raan = self.__raan
        arg_peri = self.__arg_peri
        true_anomaly = self.__true_anomaly

        # Distance (radius) at given true anomaly
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(true_anomaly))

        # Position in perifocal coordinates
        r_perifocal = np.array([r * np.cos(true_anomaly),
                                r * np.sin(true_anomaly),
                                0])
        v_perifocal = np.array([-np.sqrt(self.__mu / p) * np.sin(true_anomaly),
                                np.sqrt(self.__mu / p) * (e + np.cos(true_anomaly)),
                                0])

        # Rotation matrices
        R3_Omega = np.array([[np.cos(raan), np.sin(raan), 0],
                            [-np.sin(raan), np.cos(raan), 0],
                                [0, 0, 1]])

        R1_i = np.array([[1, 0, 0],
                        [0, np.cos(i), np.sin(i)],
                        [0, -np.sin(i), np.cos(i)]])

        R3_w = np.array([[np.cos(arg_peri), np.sin(arg_peri), 0],
                        [-np.sin(arg_peri), np.cos(arg_peri), 0],
                        [0, 0, 1]])

        ECI_to_peri = R3_w @ R1_i @ R3_Omega
        peri_to_ECI = ECI_to_peri.T

        self.__position = peri_to_ECI @ r_perifocal
        self.__velocity = peri_to_ECI @ v_perifocal
        return None
    
    # function to convert state vectors to orbital elements
    def __SV_to_OE(self):
        r_vec = self.__position
        v_vec = self.__velocity

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # Angular momentum vector
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        

        # calculate semi-major axis
        self.__a = -self.__mu /(2 * (v**2/2 - self.__mu/r))

        # caclulate E vector and eccentricity
        e_vec = (1/self.__mu) * ((v**2 - self.__mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
        self.__e = np.linalg.norm(e_vec)

        # calculate inclination
        self.__i = np.arccos(h_vec[2] / h)

        # caclulate RAAN
        # Normal vector
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
    
        if n != 0:
            self.__raan = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                self.__raan = 2 * np.pi - self.__raan
        else:
            self.__raan = 0

        # calculate argument of periapsis
        self.__arg_peri = np.arccos(np.dot(n_vec, e_vec) / (n * self.__e))
        if e_vec[2] < 0:
            self.__arg_peri = 2 * np.pi - self.__arg_peri

        # calculate true anomaly
        self.__true_anomaly = np.arccos(np.dot(e_vec, r_vec) / (self.__e * r))
        if np.dot(r_vec, v_vec) < 0:
            self.__true_anomaly = 2 * np.pi - self.__true_anomaly

        return None

def plot_multiple_orbits(orbits, labels=None, samples=500, show_earth=True, legend=True):
    """
    orbits: list of Orbit instances (your class)
    labels: optional list of labels for legend
    samples: points per orbit to sample along trajectory
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if labels is None:
        labels = [None] * len(orbits)

    colors = plt.cm.get_cmap('tab10')

    for idx, orb in enumerate(orbits):
        # choose teval for this orbit (one period)
        T = orb.get_time_period()
        teval = np.linspace(0.0, T, samples)

        # propagate returns (r_t, v_t) when teval is provided
        r_t, v_t = orb.propagate(T, teval=teval, method='odeint')

        x, y, z = r_t[:, 0], r_t[:, 1], r_t[:, 2]
        label = labels[idx] if labels[idx] is not None else f'Orbit {idx+1}'
        ax.plot(x, y, z, color=colors(idx % 10), label=label, linewidth=1.5)

        # option: mark starting point
        ax.scatter(x[0], y[0], z[0], color=colors(idx % 10), marker='o', s=20)

    # optional Earth sphere for scale
    if show_earth:
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        ue, ve = np.meshgrid(u, v)
        xe = R_EARTH * np.cos(ue) * np.sin(ve)
        ye = R_EARTH * np.sin(ue) * np.sin(ve)
        ze = R_EARTH * np.cos(ve)
        ax.plot_surface(xe, ye, ze, color='lightblue', alpha=0.3, linewidth=0)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Multiple Orbits')
    ax.set_aspect('equal', adjustable='datalim')
    
    if legend:
        ax.legend()

    plt.show()