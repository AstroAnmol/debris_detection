import numpy as np
from orbit import Orbit
from soliton import Soliton

class Satellite:
    def __init__(self):

        # Body frame: x: velocity, y: right, z: down
        # size (in body frame)
        self.__sat_x_size = 3.0 * 0.0001
        self.__sat_y_size = 2.0 * 0.0001
        self.__sat_z_size = 2.0 * 0.0001

        # boom length for sensors
        self.__boom_length = 0.001
        self.__detection_freq = 10000.0
        self.__detections = 0

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
        self.__OE_sat =  [6378 + 750, 0.063, np.deg2rad(45), 0, 0, 0]

        self.__sat_orbit = Orbit.from_OE(np.array(self.__OE_sat))

        # initial position and velocity at time 0
        self.__time = 0.0
        self.__position0 = self.__sat_orbit.get_SV()[0]
        self.__velocity0 = self.__sat_orbit.get_SV()[1]

        # set current position and velocity to initial values
        self.__position = self.__position0.copy()
        self.__velocity = self.__velocity0.copy()

        # compute rotation and sensor positions using the instance method
        self.__R_BF_to_ECI = self.BF_to_ECI()


        # sensor positions in ECI frame
        self.__sensor_1_ECI = self.__position + self.__R_BF_to_ECI.dot(self.__sensor_1_BF)
        self.__sensor_2_ECI = self.__position + self.__R_BF_to_ECI.dot(self.__sensor_2_BF)
        self.__sensor_3_ECI = self.__position + self.__R_BF_to_ECI.dot(self.__sensor_3_BF)
        self.__sensor_4_ECI = self.__position + self.__R_BF_to_ECI.dot(self.__sensor_4_BF)

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

        self.__soliton = Soliton.from_parameters(np.deg2rad(45), 10.0, 1.2, np.zeros(3), np.zeros(3))
        
    # set soliton state
    def set_soliton_state(self, pos_vec, vel_vec):
        self.__debris_position = pos_vec
        self.__debris_velocity = vel_vec

        if self.__soliton is None:
            raise RuntimeError("Soliton instance required: set Satellite soliton before calling set_soliton_state")
        self.__soliton.set_debris_state(self.__debris_position, self.__debris_velocity)
        self.__time_to_cone_base = self.__soliton.get_time_to_cone_base()
        
        sol_vel_vec = self.__soliton.get_velocity()
        print("Soliton Velocity:", sol_vel_vec)

    # check if a position vector is within satellite wake
    def within_wake(self, pos_vec):
        # convert vector from satellite center to point (ECI) into body-frame coordinates
        sat_to_pos_ECI = pos_vec - self.__position
        
        # transform ECI -> BF
        sat_to_pos_BF = self.__R_BF_to_ECI.T @ sat_to_pos_ECI

        behind_back_plane = (self.__back_plane_normal_BF.dot(sat_to_pos_BF) - self.__d_back_BF) >= 0
        behind_plane_1 = (self.__plane_normal_BF_1.dot(sat_to_pos_BF) - self.__d_1_BF) >= 0
        behind_plane_2 = (self.__plane_normal_BF_2.dot(sat_to_pos_BF) - self.__d_2_BF) >= 0
        behind_plane_3 = (self.__plane_normal_BF_3.dot(sat_to_pos_BF) - self.__d_3_BF) >= 0
        behind_plane_4 = (self.__plane_normal_BF_4.dot(sat_to_pos_BF) - self.__d_4_BF) >= 0

        return behind_back_plane and behind_plane_1 and behind_plane_2 and behind_plane_3 and behind_plane_4

    # detect soliton presence at sensors at the current time
    def detect_soliton(self):
        detected = False
        debris_within_wake = self.within_wake(self.__debris_position)
        if debris_within_wake:
            # if debris is within wake, sensors aren't checked
            pass
        else:
            sensor_1_detected = self.__soliton.within_cone(self.__sensor_1_ECI) and self.__soliton.within_spherical_shell(self.__sensor_1_ECI, self.__time, self.__detection_freq)
            sensor_2_detected = self.__soliton.within_cone(self.__sensor_2_ECI) and self.__soliton.within_spherical_shell(self.__sensor_2_ECI, self.__time, self.__detection_freq)
            sensor_3_detected = self.__soliton.within_cone(self.__sensor_3_ECI) and self.__soliton.within_spherical_shell(self.__sensor_3_ECI, self.__time, self.__detection_freq)
            sensor_4_detected = self.__soliton.within_cone(self.__sensor_4_ECI) and self.__soliton.within_spherical_shell(self.__sensor_4_ECI, self.__time, self.__detection_freq)

            if sensor_1_detected:
                print(f"Sensor 1 detected soliton at time {self.__time} seconds.")
                self.detections += 1
            if sensor_2_detected:
                print(f"Sensor 2 detected soliton at time {self.__time} seconds.")
                self.detections += 1
            if sensor_3_detected:
                print(f"Sensor 3 detected soliton at time {self.__time} seconds.")
                self.detections += 1
            if sensor_4_detected:
                print(f"Sensor 4 detected soliton at time {self.__time} seconds.")
                self.detections += 1

            detected = sensor_1_detected or sensor_2_detected or sensor_3_detected or sensor_4_detected

        return detected

    # define rotation matrix from body frame to ECI frame based on current satellite position and velocity
    def BF_to_ECI(self):
        # Body frame in ECI frame (3x3 rotation)
        sat_x_BF = self.__velocity / np.linalg.norm(self.__velocity) if np.linalg.norm(self.__velocity) != 0 else np.array([1.0, 0.0, 0.0])
        sat_z_BF = -self.__position / np.linalg.norm(self.__position) if np.linalg.norm(self.__position) != 0 else np.array([0.0, 0.0, 1.0])
        sat_y_BF = np.cross(sat_z_BF, sat_x_BF)
        if np.linalg.norm(sat_y_BF) != 0:
            sat_y_BF = sat_y_BF / np.linalg.norm(sat_y_BF)

        R_BF_to_ECI = np.column_stack([sat_x_BF, sat_y_BF, sat_z_BF])
        return R_BF_to_ECI