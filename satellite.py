import numpy as np
from src.orbit import Orbit
from src.soliton import Soliton
from src.constants import *

from scipy.spatial import ConvexHull
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Satellite:
    def __init__(self):

        # Body frame: x: velocity, y: right, z: down
        # size (in body frame)
        self.__sat_size = np.array([ 0.03, 0.02, 0.02 ]) * 1e-3 # 3 X 2 X 2 cm (in km)
        
        # boom length for sensors
        self.__boom_length = 1 * 1e-3 # 1 m (in km)

        # different detections counters
        self.__T_detections = 0 #total detections
        self.__type1_detections = 0 # type 1 detections (Same soliton at the same time detected by multiple sensors)

        # satellite corners in body frame
        self.__corners_BF = np.array([[ 1, 1, 1],
                                      [ 1,-1, 1],
                                      [ 1,-1,-1],
                                      [ 1, 1,-1]]) * self.__sat_size / 2

        # sensor positions in body frame
        self.__sensors_BF = self.__corners_BF + self.__boom_length * np.array([[ 1, 1, 1],
                                      [ 1,-1, 1],
                                      [ 1,-1,-1],
                                      [ 1, 1,-1]]) / np.sqrt(3)
        
        # satellite orbit (default values, can be updated)
        self.__OE_sat =  [R_EARTH + 750, 0.063, np.deg2rad(45), 0, 0, 0]

        self.__sat_orbit = Orbit.from_OE(np.array(self.__OE_sat))

        # initial position and velocity at time 0
        self.__position0 = self.__sat_orbit.get_SV()[0]
        self.__velocity0 = self.__sat_orbit.get_SV()[1]

        # set current position and velocity to initial values
        self.__position = self.__position0.copy()
        self.__velocity = self.__velocity0.copy()

        # compute rotation and sensor positions using the instance method
        self.__BF_to_ECI()

        # # sensor positions in ECI frame
        # self.__sensor_1_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensors_BF[0]
        # self.__sensor_2_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensors_BF[1]
        # self.__sensor_3_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensors_BF[2]
        # self.__sensor_4_ECI = self.__position + self.__R_BF_to_ECI @ self.__sensors_BF[3]

        # wake parameters
        self.__plane_angle = 20.0 * np.pi / 180.0

        self.__wake_back_plane_normal_BF = np.array([-1.0, 0.0, 0.0])
        self.__wake_d_back_BF = self.__sat_size[0] / 2.0

        self.__wake_planes_normal_BF = np.array([
            [-np.sin(self.__plane_angle), 0.0, -np.cos(self.__plane_angle)],
            [-np.sin(self.__plane_angle), 0.0,  np.cos(self.__plane_angle)],
            [-np.sin(self.__plane_angle),  np.cos(self.__plane_angle), 0.0],
            [-np.sin(self.__plane_angle), -np.cos(self.__plane_angle), 0.0]
        ])
        self.__wake_d_planes_BF = np.array([
            self.__sat_size[0] / 2 * np.sin(self.__plane_angle) - self.__sat_size[2] / 2 * np.cos(self.__plane_angle),
            self.__sat_size[0] / 2 * np.sin(self.__plane_angle) - self.__sat_size[2] / 2 * np.cos(self.__plane_angle),
            self.__sat_size[0] / 2 * np.sin(self.__plane_angle) - self.__sat_size[1] / 2 * np.cos(self.__plane_angle),
            self.__sat_size[0] / 2 * np.sin(self.__plane_angle) - self.__sat_size[1] / 2 * np.cos(self.__plane_angle)
        ])

    def get_sat_orbit(self):
        """Get the satellite's orbit object.
        Returns:
            Orbit: The satellite's orbit."""
        return self.__sat_orbit

    def set_sat_orbit_OE(self, OE):
        """Set the satellite's orbit using orbital elements.
        Args:
            OE (list): Orbital elements [a, e, i, RAAN, arg_periapsis, true_anomaly]."""
        self.__OE_sat = OE
        self.__sat_orbit = Orbit.from_OE(np.array(self.__OE_sat))
        self.__position0 = self.__sat_orbit.get_SV()[0]
        self.__velocity0 = self.__sat_orbit.get_SV()[1]
        # reset current position and velocity to initial values
        self.__position = self.__position0.copy()
        self.__velocity = self.__velocity0.copy()
        self.__BF_to_ECI()
    
    def set_sat_orbit_SV(self, position, velocity):
        """Set the satellite's orbit using state vectors.
        Args:
            position (np.array): Position vector in ECI frame.
            velocity (np.array): Velocity vector in ECI frame."""
        self.__position0 = position
        self.__velocity0 = velocity
        # reset current position and velocity to initial values
        self.__position = self.__position0.copy()
        self.__velocity = self.__velocity0.copy()
        self.__sat_orbit = Orbit.from_SV(np.array([self.__position0, self.__velocity0]))
        self.__BF_to_ECI()

    # # check if a position vector is within satellite wake at current time
    # def within_wake(self, pos_vec):
    #     # convert vector from satellite center to point (ECI) into body-frame coordinates
    #     sat_to_pos_ECI = pos_vec - self.__position
        
    #     # transform ECI -> BF
    #     sat_to_pos_BF = self.__R_BF_to_ECI.T @ sat_to_pos_ECI

    #     behind_back_plane = (self.__wake_back_plane_normal_BF @ (sat_to_pos_BF) - self.__wake_d_back_BF) >= 0
    #     behind_plane_1 = (self.__wake_planes_normal_BF[0] @ sat_to_pos_BF - self.__wake_d_planes_BF[0]) >= 0
    #     behind_plane_2 = (self.__wake_planes_normal_BF[1] @ sat_to_pos_BF - self.__wake_d_planes_BF[1]) >= 0
    #     behind_plane_3 = (self.__wake_planes_normal_BF[2] @ sat_to_pos_BF - self.__wake_d_planes_BF[2]) >= 0
    #     behind_plane_4 = (self.__wake_planes_normal_BF[3] @ sat_to_pos_BF - self.__wake_d_planes_BF[3]) >= 0
    #     return behind_back_plane and behind_plane_1 and behind_plane_2 and behind_plane_3 and behind_plane_4

    # # detect soliton presence at at least two different sensors at the current time
    # def __detect_type1_time(self, soliton):
    #     detected = False
    #     debris_position = soliton.get_debris_position()
    #     debris_within_wake = self.within_wake(debris_position)
    #     if debris_within_wake:
    #         # if debris is within wake, sensors aren't checked
    #         pass
    #     else:
    #         sensor_1_detected = (soliton.within_cone(self.__sensor_1_ECI) and
    #                             soliton.within_spherical_shell(self.__sensor_1_ECI, self.__time))
    #         sensor_2_detected = (soliton.within_cone(self.__sensor_2_ECI) and
    #                             soliton.within_spherical_shell(self.__sensor_2_ECI, self.__time))
    #         sensor_3_detected = (soliton.within_cone(self.__sensor_3_ECI) and
    #                             soliton.within_spherical_shell(self.__sensor_3_ECI, self.__time))
    #         sensor_4_detected = (soliton.within_cone(self.__sensor_4_ECI) and
    #                             soliton.within_spherical_shell(self.__sensor_4_ECI, self.__time))

    #         if sensor_1_detected:
    #             print(f"Sensor 1 detected soliton at time {self.__time} seconds.")
    #         if sensor_2_detected:
    #             print(f"Sensor 2 detected soliton at time {self.__time} seconds.")
    #         if sensor_3_detected:
    #             print(f"Sensor 3 detected soliton at time {self.__time} seconds.")
    #         if sensor_4_detected:
    #             print(f"Sensor 4 detected soliton at time {self.__time} seconds.")
    #         # Require at least two sensors to report detection for a type-1 event
    #         sensor_count = int(sensor_1_detected) + int(sensor_2_detected) + int(sensor_3_detected) + int(sensor_4_detected)
    #         if sensor_count >= 2:
    #             detected = True
    #             # update counters
    #             self.__T_detections += 1
    #             # increment type1 event count (one event per time-step)
    #             self.__type1_detections += 1
    #             print(f"Type-1 detection: {sensor_count} sensors detected at time {self.__time} (total detections={self.__T_detections}, type1_events={self.__type1_detections})")
    #         else:
    #             detected = False

    #     return detected
    
    # # check if a given soliton will be detected in type 1 scenario within soliton lifetime
    # def detect_type1_soliton(self, soliton):
    #     detected = False
        
    #     # get soliton information
    #     soliton_lifetime = soliton.get_time_to_cone_base()
    #     soliton_generation_time = soliton.get_time_of_generation()

    #     # define time steps to evaluate based on detection frequency
    #     num_steps = int(soliton_lifetime * DETECTION_FREQ)
    #     timesteps = np.linspace(soliton_generation_time, soliton_generation_time + soliton_lifetime, num_steps)
        
    #     # propagate satellite to soliton generation time
    #     positions, velocities = self.__sat_orbit.propagate(soliton_generation_time + soliton_lifetime, teval=timesteps, use_J2=True)

    #     for idx, t in enumerate(timesteps):
    #         # update satellite state
    #         self.__time = t
    #         self.__position = positions[idx]
    #         self.__velocity = velocities[idx]
    #         # update sensor positions and rotation matrix
    #         self.__BF_to_ECI()

    #         if self.__detect_type1_time(soliton):
    #             detected = True
    #             break
    #         else:
    #             detected = False

    #     return detected

    def pos_BF2ECI(self, pos_BF):
        """Convert a position vector from body frame to ECI frame."""
        return self.__R_BF_to_ECI @ pos_BF + self.__position0
    
    def pos_ECI2BF(self, pos_ECI):
        """Convert a position vector from ECI frame to body frame."""
        return self.__R_BF_to_ECI.T @ (pos_ECI - self.__position0)

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

        self.__R_BF_to_ECI = R_BF_to_ECI

        return None
    
    def plot_satellite(self, ax, position=np.zeros(3), velocity=[1,0,0]):
        """In the given figure, plot the satellite body and sensor positions (ECI frame).
            Need to have an active 3D matplotlib axis.
            If position and velocity are no provides, plots in Body frame.
        Args:
            ax (matplotlib axis): 3D axis to plot on.
            position (np.array): Optional position (ECI_frame) to plot the satellite at. Default is origin.
            velocity (np.array): Optional velocity (ECI_frame) to define satellite orientation. Default is [1,0,0]."""
        

        # update rotation matrix based on given velocity
        sat_x_BF = np.array(velocity) / np.linalg.norm(velocity) if np.linalg.norm(velocity) != 0 else np.array([1.0, 0.0, 0.0])
        sat_z_BF = -position / np.linalg.norm(position) if np.linalg.norm(position) != 0 else np.array([0.0, 0.0, 1.0])
        sat_y_BF = np.cross(sat_z_BF, sat_x_BF)
        if np.linalg.norm(sat_y_BF) != 0:
            sat_y_BF = sat_y_BF / np.linalg.norm(sat_y_BF)
        R_BF_to_ECI = np.column_stack([sat_x_BF, sat_y_BF, sat_z_BF])

        # plot satellite body (as a cuboid)
        corners = np.array([
            self.__corners_BF[0],
            self.__corners_BF[1],
            self.__corners_BF[2],
            self.__corners_BF[3],
            -self.__corners_BF[0],
            -self.__corners_BF[1],
            -self.__corners_BF[2],
            -self.__corners_BF[3]
        ])
        corners = (R_BF_to_ECI @ corners.T).T + position

        # plot edges
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        edges_BF = [
            [corners[0],corners[1]],
            [corners[1],corners[2]],
            [corners[2],corners[3]],
            [corners[3],corners[0]],
            [corners[4],corners[5]],
            [corners[5],corners[6]],
            [corners[6],corners[7]],
            [corners[7],corners[4]],
            [corners[0],corners[6]],
            [corners[1],corners[7]],
            [corners[2],corners[4]],
            [corners[3],corners[5]],
            [corners[0], (R_BF_to_ECI @ self.__sensors_BF[0]) + position],
            [corners[1], (R_BF_to_ECI @ self.__sensors_BF[1]) + position],
            [corners[2], (R_BF_to_ECI @ self.__sensors_BF[2]) + position],
            [corners[3], (R_BF_to_ECI @ self.__sensors_BF[3]) + position],
        ]
        lc = Line3DCollection(edges_BF, colors='tab:orange', linewidths=1)
        ax.add_collection3d(lc)

        normals = [R_BF_to_ECI @ self.__wake_back_plane_normal_BF,
                   R_BF_to_ECI @ self.__wake_planes_normal_BF[0],
                   R_BF_to_ECI @ self.__wake_planes_normal_BF[1],
                   R_BF_to_ECI @ self.__wake_planes_normal_BF[2],
                   R_BF_to_ECI @ self.__wake_planes_normal_BF[3]]
        constants = [self.__wake_d_back_BF ,
                     self.__wake_d_planes_BF[0] ,
                     self.__wake_d_planes_BF[1] ,
                     self.__wake_d_planes_BF[2] ,
                     self.__wake_d_planes_BF[3] ]
        
        verts = get_intersection_vertices(normals, constants, R_BF_to_ECI, limit=0.001)
        # change verts to new origin

        if verts is not None and len(verts) >= 3:
            verts += position
            plot_polyhedron(ax, verts)
        # plot_planes_intersection(ax, [self.__back_plane_normal_BF], [self.__d_back_BF], origin=position)
        return None
    
def get_intersection_vertices(normals, constants,  R_BF_to_ECI, limit=0.001):
    """
    Finds vertices of the polyhedron by solving the intersection of 
    every combination of 3 planes and filtering for validity.
    
    Equation: n.x >= d
    """
    
    # 1. Define Bounding Box Planes (Axis Aligned)
    # x <= limit  -> -x >= -limit -> n=[-1,0,0], d=-limit
    # x >= -limit ->  x >= -limit -> n=[1,0,0],  d=-limit
    box_normals = [
        [-1, 0, 0], [1, 0, 0],  # x bounds
        [0, -1, 0], [0, 1, 0],  # y bounds
        [0, 0, -1], [0, 0, 1]   # z bounds
    ]
    # rotate box normals to match the frame of the user-defined planes
    box_normals = [R_BF_to_ECI @ np.array(n) for n in box_normals]
    box_constants = [-limit] * 6
    
    # 2. Combine User Planes with Box Planes
    all_normals = np.array(normals + box_normals)
    all_constants = np.array(constants + box_constants)
    
    num_planes = len(all_normals)
    valid_vertices = []

    # 3. Iterate through all combinations of 3 planes
    # We need 3 planes to define a 3D point (intersection)
    for indices in itertools.combinations(range(num_planes), 3):
        try:
            # Construct linear system Ax = b for these 3 planes
            # n.x = d (treating inequality as equality to find intersection)
            A = all_normals[list(indices)]
            b = all_constants[list(indices)]
            
            # Solve intersection
            point = np.linalg.solve(A, b)
            
            # 4. Check if this point satisfies ALL other inequalities
            # n_i . x >= d_i  (allow small epsilon for float precision)
            is_valid = True
            
            # We check dot products against all planes
            dots = np.dot(all_normals, point)
            
            # Check feasibility with tolerance
            # If dot < d - epsilon, constraint is violated
            if np.any(dots < all_constants - 1e-6):
                is_valid = False
            
            if is_valid:
                valid_vertices.append(point)

        except np.linalg.LinAlgError:
            # This happens if the 3 planes are parallel (singular matrix)
            continue

    return np.array(valid_vertices)

def plot_polyhedron(ax, vertices):
    """
    Plots the Convex Hull of the given vertices.
    """
    if len(vertices) < 4:
        print("Not enough vertices to plot a 3D shape.")
        return

    try:
        # Compute the Convex Hull to get the faces/edges order
        hull = ConvexHull(vertices)
        
        faces = [vertices[simplex] for simplex in hull.simplices]
        
        # Plot Faces
        poly = Poly3DCollection(faces, alpha=0.6, edgecolor='k', linewidths=1)
        poly.set_facecolor('cyan')
        ax.add_collection3d(poly)
        
        # Plot Vertices
        # ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='r', s=50)

        # Auto-scale
        # ax.set_xlim(vertices[:,0].min()-1, vertices[:,0].max()+1)
        # ax.set_ylim(vertices[:,1].min()-1, vertices[:,1].max()+1)
        # ax.set_zlim(vertices[:,2].min()-1, vertices[:,2].max()+1)
        
    except Exception as e:
        print(f"Error computing hull: {e}")
