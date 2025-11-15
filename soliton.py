import numpy as np
from constants import *
class Soliton:
    def __init__(self):
        # initialize instance attributes (defualt values)
        self.__cone_angle = np.deg2rad(45) # Cone angle in radians
        self.__cone_height = 10.0 # Height of the cone in km
        self.__vel_multiplier = 1.2 # Velocity multiplier for soliton

        # placeholder debris attributes
        self.__time_of_generation = 0.0 # Time of soliton generation in s
        self.__debris_pos_vec = np.zeros(3) # Debris position vector in km 
        self.__debris_vel_vec = np.zeros(3) # Debris velocity vector in km/s

        self.__sol_vel_vec = self.__debris_vel_vec * self.__vel_multiplier # Soliton velocity vector in km/s
        self.__time_to_reach_cone_base = self.__cone_height / np.linalg.norm(self.__sol_vel_vec) # Time to reach cone base in s
        
    
    # alternative constructor
    @classmethod
    def from_parameters(cls, angle, height, vel_multiplier, time_of_generation, pos_vec, vel_vec):
        obj = cls()
        obj.__cone_angle = angle
        obj.__cone_height = height
        obj.__vel_multiplier = vel_multiplier
        obj.__time_of_generation = time_of_generation
        obj.__debris_pos_vec = pos_vec
        obj.__debris_vel_vec = vel_vec
        obj.__sol_vel_vec = vel_vec * vel_multiplier
        obj.__time_to_reach_cone_base = obj.__cone_height / np.linalg.norm(obj.__sol_vel_vec)
        return obj
    
    # Public methods

    def set_params(self, angle, height, vel_multiplier):
        self.__cone_angle = angle
        self.__cone_height = height
        self.__vel_multiplier = vel_multiplier
        self.__sol_vel_vec = self.__debris_vel_vec * self.__vel_multiplier
        self.__time_to_reach_cone_base = self.__cone_height / np.linalg.norm(self.__sol_vel_vec)

    def set_debris_state(self, time_of_generation, pos_vec, vel_vec):
        self.__time_of_generation = time_of_generation
        self.__debris_pos_vec = pos_vec
        self.__debris_vel_vec = vel_vec
        self.__sol_vel_vec = vel_vec * self.__vel_multiplier
        self.__time_to_reach_cone_base = self.__cone_height / np.linalg.norm(self.__sol_vel_vec)
    
    def get_velocity(self):
        return self.__sol_vel_vec
    
    def get_time_to_cone_base(self):
        return self.__time_to_reach_cone_base
    
    # Detection functions 

    # Check if a position vector is within the soliton shell at time t(interaction of cone and two spheres)

    # first check if within cone
    def within_cone(self, pos_vec):
        """Check if a given position vector is within the soliton cone.
        Args:
            pos_vec (np.array): Position vector to check (3D).
        Returns:
            bool: True if within cone, False otherwise."""
        
        # Debris position is cone apex
        apex = self.__debris_pos_vec
        # Cone axis is along soliton velocity vector
        axis = self.__sol_vel_vec / np.linalg.norm(self.__sol_vel_vec)
        # Vector from apex to point
        apex_to_point = pos_vec - apex

        within_height = np.dot(apex_to_point, axis) <= self.__cone_height
        within_angle = np.dot(apex_to_point, axis) >= np.linalg.norm(apex_to_point) * np.cos(self.__cone_angle)
        same_side = np.dot(apex_to_point, axis) >= 0

        return within_height and within_angle and same_side

    # check if within spherical shell at time t
    def within_spherical_shell(self, pos_vec, t):
        """Check if a given position vector is within the soliton spherical shell at time t.
        Args:
            pos_vec (np.array): Position vector to check (3D).
            t (float): Time in seconds.
        Returns:
            bool: True if within spherical shell, False otherwise."""
        
        # Calculate current radius of soliton shell
        shell_radius = np.linalg.norm(self.__sol_vel_vec) * (t - self.__time_of_generation)
        shell_thickness = SOL_SHELL_THICKNESS  # Define shell thickness constant (e.g., 1 km)

        # Distance from debris position to point
        apex_to_point = np.linalg.norm(pos_vec - self.__debris_pos_vec)

        return apex_to_point >= (shell_radius - shell_thickness/2) and apex_to_point <= (shell_radius + shell_thickness/2)