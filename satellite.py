import numpy as np
from orbit import Orbit
from soliton import Soliton
from constants import *

class Satellite:
    def __init__(self):

        # Body frame: x: velocity, y: right, z: down
        # size (in body frame)
        self.__sat_x_size = 3.0 * 0.0001 # 3 cm
        self.__sat_y_size = 2.0 * 0.0001 # 2 cm
        self.__sat_z_size = 2.0 * 0.0001 # 2 cm

        # boom length for sensors
        self.__boom_length = 0.001
        self.__detection_freq = DETECTION_FREQ # 10 kHz

        # different detections counters
        self.__T_detections = 0 #total detections
        self.__type1_detections = 0 # type 1 detections (Same soliton at the same time detected by multiple sensors)

        # satellite corners in body frame
        self.__corner_1_BF = np.array([self.__sat_x_size / 2, self.__sat_y_size / 2, self.__sat_z_size / 2])
        self.__corner_2_BF = np.array([self.__sat_x_size / 2, -self.__sat_y_size / 2, self.__sat_z_size / 2])
        self.__corner_3_BF = np.array([self.__sat_x_size / 2, -self.__sat_y_size / 2, -self.__sat_z_size / 2])
        self.__corner_4_BF = np.array([self.__sat_x_size / 2, self.__sat_y_size / 2, -self.__sat_z_size / 2])

        # sensor positions in body frame
        self.__sensor_1_BF = self.__corner_1_BF + self.__boom_length * np.array([1, 1, 1])/np.sqrt(3)
        self.__sensor_2_BF = self.__corner_2_BF + self.__boom_length * np.array([1, -1, 1])/np.sqrt(3)
        self.__sensor_3_BF = self.__corner_3_BF + self.__boom_length * np.array([1, -1, -1])/np.sqrt(3)
        self.__sensor_4_BF = self.__corner_4_BF + self.__boom_length * np.array([1, 1, -1])/np.sqrt(3)

        # satellite orbit 
        self.__OE_sat =  [R_EARTH + 750, 0.063, np.deg2rad(45), 0, 0, 0]

        self.__sat_orbit = Orbit.from_OE(np.array(self.__OE_sat))

        # initial position and velocity at time 0
        self.__time = 0.0
        self.__position0 = self.__sat_orbit.get_SV()[0]
        self.__velocity0 = self.__sat_orbit.get_SV()[1]

        # set current position and velocity to initial values
        self.__position = self.__position0.copy()
        self.__velocity = self.__velocity0.copy()

        # compute rotation and sensor positions using the instance method
        self.__BF_to_ECI()


        # sensor positions in ECI frame
        self.__sensor_1_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensor_1_BF
        self.__sensor_2_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensor_2_BF
        self.__sensor_3_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensor_3_BF
        self.__sensor_4_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensor_4_BF

        # wake parameters
        self.__plane_angle = 20.0 * np.pi / 180.0

        self.__back_plane_normal_BF = np.array([-1.0, 0.0, 0.0])
        self.__d_back_BF = self.__sat_x_size / 2.0

        self.__plane_normal_BF_1 = np.array([-np.sin(self.__plane_angle), 0.0, -np.cos(self.__plane_angle)])
        self.__d_1_BF = self.__sat_x_size / 2 * np.sin(self.__plane_angle) - self.__sat_z_size / 2 * np.cos(self.__plane_angle)
        self.__plane_normal_BF_2 = np.array([-np.sin(self.__plane_angle), 0.0, np.cos(self.__plane_angle)])
        self.__d_2_BF = self.__sat_x_size / 2 * np.sin(self.__plane_angle) - self.__sat_z_size / 2 * np.cos(self.__plane_angle)

        self.__plane_normal_BF_3 = np.array([-np.sin(self.__plane_angle), np.cos(self.__plane_angle), 0.0])
        self.__d_3_BF = self.__sat_x_size / 2 * np.sin(self.__plane_angle) - self.__sat_y_size / 2 * np.cos(self.__plane_angle)

        self.__plane_normal_BF_4 = np.array([-np.sin(self.__plane_angle), -np.cos(self.__plane_angle), 0.0])
        self.__d_4_BF = self.__sat_x_size / 2 * np.sin(self.__plane_angle) - self.__sat_y_size / 2 * np.cos(self.__plane_angle)

    # check satellite raan precession rate
    def get_raan_precession_rate(self):
        # RAAN precession rate due to J2 perturbation (radians per second)
        a = self.__OE_sat[0]  # semi-major axis in km
        e = self.__OE_sat[1]  # eccentricity
        i = self.__OE_sat[2]  # inclination in radians

        n = np.sqrt(GM_EARTH / a**3)  # mean motion in rad/s

        raan_dot = -1.5 * J2_EARTH * (R_EARTH / a)**2 * n * np.cos(i) / (1 - e**2)**2

        return raan_dot
    
    # get satellite argument of perigee precession rate
    def get_omega_precession_rate(self):
        # Argument of perigee precession rate due to J2 perturbation (radians per second)
        a = self.__OE_sat[0]  # semi-major axis in km
        e = self.__OE_sat[1]  # eccentricity
        i = self.__OE_sat[2]  # inclination in radians

        n = np.sqrt(GM_EARTH / a**3)  # mean motion in rad/s

        omega_dot = 0.75 * J2_EARTH * (R_EARTH / a)**2 * n * (4 - 5 * np.sin(i)**2) / (1 - e**2)**2

        return omega_dot

    # check if a position vector is within satellite wake at current time
    def within_wake(self, pos_vec):
        # convert vector from satellite center to point (ECI) into body-frame coordinates
        sat_to_pos_ECI = pos_vec - self.__position
        
        # transform ECI -> BF
        sat_to_pos_BF = self.__R_BF_to_ECI.T @ sat_to_pos_ECI

        behind_back_plane = (self.__back_plane_normal_BF @ (sat_to_pos_BF) - self.__d_back_BF) >= 0
        behind_plane_1 = (self.__plane_normal_BF_1 @ sat_to_pos_BF - self.__d_1_BF) >= 0
        behind_plane_2 = (self.__plane_normal_BF_2 @ sat_to_pos_BF - self.__d_2_BF) >= 0
        behind_plane_3 = (self.__plane_normal_BF_3 @ sat_to_pos_BF - self.__d_3_BF) >= 0
        behind_plane_4 = (self.__plane_normal_BF_4 @ sat_to_pos_BF - self.__d_4_BF) >= 0
        return behind_back_plane and behind_plane_1 and behind_plane_2 and behind_plane_3 and behind_plane_4

    # detect soliton presence at at least two different sensors at the current time
    def __detect_type1_time(self, soliton):
        detected = False
        debris_position = soliton.get_debris_position()
        debris_within_wake = self.within_wake(debris_position)
        if debris_within_wake:
            # if debris is within wake, sensors aren't checked
            pass
        else:
            sensor_1_detected = (soliton.within_cone(self.__sensor_1_ECI) and
                                soliton.within_spherical_shell(self.__sensor_1_ECI, self.__time))
            sensor_2_detected = (soliton.within_cone(self.__sensor_2_ECI) and
                                soliton.within_spherical_shell(self.__sensor_2_ECI, self.__time))
            sensor_3_detected = (soliton.within_cone(self.__sensor_3_ECI) and
                                soliton.within_spherical_shell(self.__sensor_3_ECI, self.__time))
            sensor_4_detected = (soliton.within_cone(self.__sensor_4_ECI) and
                                soliton.within_spherical_shell(self.__sensor_4_ECI, self.__time))

            if sensor_1_detected:
                print(f"Sensor 1 detected soliton at time {self.__time} seconds.")
            if sensor_2_detected:
                print(f"Sensor 2 detected soliton at time {self.__time} seconds.")
            if sensor_3_detected:
                print(f"Sensor 3 detected soliton at time {self.__time} seconds.")
            if sensor_4_detected:
                print(f"Sensor 4 detected soliton at time {self.__time} seconds.")
            # Require at least two sensors to report detection for a type-1 event
            sensor_count = int(sensor_1_detected) + int(sensor_2_detected) + int(sensor_3_detected) + int(sensor_4_detected)
            if sensor_count >= 2:
                detected = True
                # update counters
                self.__T_detections += 1
                # increment type1 event count (one event per time-step)
                self.__type1_detections += 1
                print(f"Type-1 detection: {sensor_count} sensors detected at time {self.__time} (total detections={self.__T_detections}, type1_events={self.__type1_detections})")
            else:
                detected = False

        return detected
    
    # check if a given soliton will be detected in type 1 scenario within soliton lifetime
    def detect_type1_soliton(self, soliton):
        detected = False
        
        # get soliton information
        soliton_lifetime = soliton.get_time_to_cone_base()
        soliton_generation_time = soliton.get_time_of_generation()

        # define time steps to evaluate based on detection frequency
        num_steps = int(soliton_lifetime * self.__detection_freq)
        timesteps = np.linspace(soliton_generation_time, soliton_generation_time + soliton_lifetime, num_steps)
        
        # propagate satellite to soliton generation time
        positions, velocities = self.__sat_orbit.propagate(soliton_generation_time + soliton_lifetime, teval=timesteps, use_J2=True)

        for idx, t in enumerate(timesteps):
            # update satellite state
            self.__time = t
            self.__position = positions[idx]
            self.__velocity = velocities[idx]
            # update sensor positions and rotation matrix
            self.__BF_to_ECI()

            if self.__detect_type1_time(soliton):
                detected = True
                break
            else:
                detected = False

        return detected

    
    
    # define rotation matrix from body frame to ECI frame based on current satellite position and velocity
    # and update all sensor positions in ECI frame
    def __BF_to_ECI(self):
        # Body frame in ECI frame (3x3 rotation)
        sat_x_BF = self.__velocity / np.linalg.norm(self.__velocity) if np.linalg.norm(self.__velocity) != 0 else np.array([1.0, 0.0, 0.0])
        sat_z_BF = -self.__position / np.linalg.norm(self.__position) if np.linalg.norm(self.__position) != 0 else np.array([0.0, 0.0, 1.0])
        sat_y_BF = np.cross(sat_z_BF, sat_x_BF)
        if np.linalg.norm(sat_y_BF) != 0:
            sat_y_BF = sat_y_BF / np.linalg.norm(sat_y_BF)

        R_BF_to_ECI = np.column_stack([sat_x_BF, sat_y_BF, sat_z_BF])

        self.__sensor_1_ECI = R_BF_to_ECI @ self.__sensor_1_BF + self.__position
        self.__sensor_2_ECI = R_BF_to_ECI @ self.__sensor_2_BF + self.__position
        self.__sensor_3_ECI = R_BF_to_ECI @ self.__sensor_3_BF + self.__position
        self.__sensor_4_ECI = R_BF_to_ECI @ self.__sensor_4_BF + self.__position

        self.__R_BF_to_ECI = R_BF_to_ECI

        return None