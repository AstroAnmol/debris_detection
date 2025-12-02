import numpy as np
from src.constants import *

class Soliton:
    def __init__(self):
        # initialize instance attributes (defualt values)
        self.__cone_angle = np.deg2rad(45) # Cone angle in radians
        self.__cone_height = 10.0 # Height of the cone in km
        self.__vel_multiplier = 1.2 # Velocity multiplier for soliton

        # placeholder debris attributes
        self.__time_of_generation = 0.0 # Time of soliton generation in s
        self.__debris_pos_vec = np.ones(3) # Debris position vector in km 
        self.__debris_vel_vec = np.ones(3) # Debris velocity vector in km/s


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
    
    def get_time_of_generation(self):
        return self.__time_of_generation
    
    def get_debris_position(self):
        return self.__debris_pos_vec
    
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

    def plot_soliton(self, ax, t, plot_intersection=True, plot_cone=False, plot_shell=False):
        """
        Plots the soliton cone and spherical shell at time t on the given 3D axis.
        
        Args:
            ax: matplotlib 3D axis.
            t (float): Time in seconds.
            plot_intersection (bool): Whether to plot the intersection (spherical cap).
            plot_cone (bool): Whether to plot the cone.
            plot_shell (bool): Whether to plot the spherical shell.
        """
        
        # Common rotation logic
        def get_rotation_matrix(vec1, vec2):
            """Get rotation matrix that aligns vec1 to vec2"""
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            if any(v): # if not parallel
                c = np.dot(a, b)
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            else:
                return np.eye(3) # parallel, no rotation needed (or 180 flip if opposite, but assuming aligned for now)

        # Plot debris position
        ax.scatter(self.__debris_pos_vec[0], self.__debris_pos_vec[1], self.__debris_pos_vec[2], color='black', marker='o', s=20, label='Debris')

        # 1. Plot Cone
        if plot_cone:
            # Cone parameters
            apex = self.__debris_pos_vec
            h_vec = self.__sol_vel_vec / np.linalg.norm(self.__sol_vel_vec) * self.__cone_height
            
            # Generate cone mesh in local frame (z-aligned)
            res = 30
            u = np.linspace(0, 2 * np.pi, res)
            v = np.linspace(0, self.__cone_height, res)
            U, V = np.meshgrid(u, v)
            
            # Radius at height V
            R = V * np.tan(self.__cone_angle)
            
            X_local = R * np.cos(U)
            Y_local = R * np.sin(U)
            Z_local = V
            
            # Rotate to align Z_local with h_vec
            z_unit = np.array([0, 0, 1])
            h_vec_unit = h_vec / np.linalg.norm(h_vec)
            
            # Use robust rotation matrix calculation
            if np.allclose(z_unit, h_vec_unit):
                R_mat = np.eye(3)
            elif np.allclose(z_unit, -h_vec_unit):
                R_mat = np.diag([1, -1, -1])
            else:
                R_mat = get_rotation_matrix(z_unit, h_vec_unit)
            
            # Apply rotation and translation
            points = np.vstack([X_local.ravel(), Y_local.ravel(), Z_local.ravel()])
            rotated_points = R_mat @ points
            
            X_plot = rotated_points[0, :].reshape(X_local.shape) + apex[0]
            Y_plot = rotated_points[1, :].reshape(Y_local.shape) + apex[1]
            Z_plot = rotated_points[2, :].reshape(Z_local.shape) + apex[2]
            
            ax.plot_surface(X_plot, Y_plot, Z_plot, alpha=0.1, color='orange')
            
        # 2. Plot Spherical Shell
        if plot_shell:
            radius = np.linalg.norm(self.__sol_vel_vec) * (t - self.__time_of_generation)
            if radius > 0:
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 30)
                
                x = radius * np.outer(np.cos(u), np.sin(v)) + self.__debris_pos_vec[0]
                y = radius * np.outer(np.sin(u), np.sin(v)) + self.__debris_pos_vec[1]
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.__debris_pos_vec[2]
                
                ax.plot_wireframe(x, y, z, color='blue', alpha=0.1)

        # 3. Plot Intersection (Spherical Cap)
        if plot_intersection:
            radius = np.linalg.norm(self.__sol_vel_vec) * (t - self.__time_of_generation)
            if radius > 0:
                # Spherical cap parameterization
                # theta (polar angle) from 0 to cone_angle
                # phi (azimuthal angle) from 0 to 2*pi
                res = 30
                phi = np.linspace(0, 2 * np.pi, res)
                theta = np.linspace(0, self.__cone_angle, res) # Only up to cone angle
                PHI, THETA = np.meshgrid(phi, theta)

                # Cartesian coordinates in local frame (Z-aligned)
                # Z is along the cone axis
                X_local = radius * np.sin(THETA) * np.cos(PHI)
                Y_local = radius * np.sin(THETA) * np.sin(PHI)
                Z_local = radius * np.cos(THETA)

                # Rotate to align Z_local with soliton velocity vector
                z_unit = np.array([0, 0, 1])
                h_vec = self.__sol_vel_vec
                h_vec_unit = h_vec / np.linalg.norm(h_vec)

                if np.allclose(z_unit, h_vec_unit):
                    R_mat = np.eye(3)
                elif np.allclose(z_unit, -h_vec_unit):
                    R_mat = np.diag([1, -1, -1])
                else:
                    R_mat = get_rotation_matrix(z_unit, h_vec_unit)

                # Apply rotation and translation
                points = np.vstack([X_local.ravel(), Y_local.ravel(), Z_local.ravel()])
                rotated_points = R_mat @ points

                X_plot = rotated_points[0, :].reshape(X_local.shape) + self.__debris_pos_vec[0]
                Y_plot = rotated_points[1, :].reshape(Y_local.shape) + self.__debris_pos_vec[1]
                Z_plot = rotated_points[2, :].reshape(Z_local.shape) + self.__debris_pos_vec[2]

                ax.plot_surface(X_plot, Y_plot, Z_plot, alpha=0.5, color='red')

    def animate_soliton(self, times, interval=100, save_path=None, fig=None, ax=None, **kwargs):
        """
        Animates the soliton evolution over the given time steps.
        
        Args:
            times (iterable): Sequence of time steps to animate.
            interval (int): Delay between frames in milliseconds.
            save_path (str): Path to save the animation (e.g., 'soliton.gif').
            ax (matplotlib.axes.Axes3D): 3D axis to plot on. If None, create a new figure.
            fig (matplotlib.figure.Figure): Figure to plot on. If None, create a new figure.
            **kwargs: Additional arguments passed to plot_soliton.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
                
        # Calculate limits based on max extent at final time
        t_max = max(times)
        # Estimate max extent: debris position + velocity * t_max + radius
        # But simpler: just center on debris and use max radius
        radius_max = np.linalg.norm(self.__sol_vel_vec) * (t_max - self.__time_of_generation)
        
        # Center roughly at debris position
        center = self.__debris_pos_vec
        extent = radius_max * 1.5 if radius_max > 0 else 10.0
        
        def update(t):
            ax.clear()
            self.plot_soliton(ax, t, **kwargs)
            
            # Set limits
            ax.set_xlim(center[0] - extent, center[0] + extent)
            ax.set_ylim(center[1] - extent, center[1] + extent)
            ax.set_zlim(center[2] - extent, center[2] + extent)
            
            ax.set_title(f"Soliton at t={t:.2f}s")
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            ax.set_zlabel("Z (km)")
            
        anim = FuncAnimation(fig, update, frames=times, interval=interval)
        if save_path:
            anim.save(save_path, writer='pillow')
        else:
            plt.show()
        return None

    