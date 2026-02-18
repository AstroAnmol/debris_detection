import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from joblib import Parallel, delayed
from src.orbit import Orbit
from src.soliton import Soliton
from src.constants import *
import os
from datetime import datetime

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

        # sensore angles in body frame
        self.__alpha_x = np.deg2rad(54.74)
        self.__alpha_z = np.deg2rad(45.00)

        self.__sensor_vectors = np.array([
            [ np.cos(self.__alpha_x), np.sin(self.__alpha_z)*np.sin(self.__alpha_x), np.cos(self.__alpha_z)*np.sin(self.__alpha_x)],
            [ np.cos(self.__alpha_x),-np.sin(self.__alpha_z)*np.sin(self.__alpha_x), np.cos(self.__alpha_z)*np.sin(self.__alpha_x)],
            [ np.cos(self.__alpha_x),-np.sin(self.__alpha_z)*np.sin(self.__alpha_x),-np.cos(self.__alpha_z)*np.sin(self.__alpha_x)],
            [ np.cos(self.__alpha_x), np.sin(self.__alpha_z)*np.sin(self.__alpha_x),-np.cos(self.__alpha_z)*np.sin(self.__alpha_x)]
        ]) 

        # satellite corners in body frame
        self.__corners_BF = np.array([[ 1, 1, 1],
                                      [ 1,-1, 1],
                                      [ 1,-1,-1],
                                      [ 1, 1,-1]]) * self.__sat_size / (2 * np.sqrt(3))

        # sensor positions in body frame
        self.__sensors_BF = self.__corners_BF + self.__boom_length * self.__sensor_vectors
        
        # satellite orbit (default values, can be updated)
        self.__OE_sat =  [R_EARTH + 750, 0.063, np.deg2rad(45), 0, 0, 0]

        self.__sat_orbit = Orbit.from_OE(np.array(self.__OE_sat))

        # initial position and velocity at time 0
        self.__time = 0.0 # current time in seconds
        self.__position0 = self.__sat_orbit.get_SV()[0]
        self.__velocity0 = self.__sat_orbit.get_SV()[1]

        # set current position and velocity to initial values
        self.__position = self.__position0.copy()
        self.__velocity = self.__velocity0.copy()

        # compute rotation and sensor positions using the instance method
        self.__BF_to_ECI()

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

    def set_sensor_vectors(self, angles):
        """Set the sensor vectors.
        Args:
            angles (list): List of sensor angles in degrees [az1, el1, az2, el2, az3, el3, az4, el4].
        """
        self.__sensor_vectors = np.array([
            [ np.cos(angles[0]), np.sin(angles[0])*np.sin(angles[1]), np.sin(angles[0])*np.cos(angles[1])],
            [ np.cos(angles[2]),-np.sin(angles[2])*np.sin(angles[3]), np.sin(angles[2])*np.cos(angles[3])],
            [ np.cos(angles[4]),-np.sin(angles[4])*np.sin(angles[5]),-np.sin(angles[4])*np.cos(angles[5])],
            [ np.cos(angles[6]), np.sin(angles[6])*np.sin(angles[7]),-np.sin(angles[6])*np.cos(angles[7])]
        ]) 

        # sensor positions in body frame
        self.__sensors_BF = self.__corners_BF + self.__boom_length * self.__sensor_vectors
        
        self.__BF_to_ECI()

    # check if a position vector (ECI) is within satellite wake at current time
    def within_wake(self, pos_vec):
        """
        Check if a given position vector (in ECI frame) is within the satellite's wake at the current time.
        Args:
            pos_vec (np.array): Position vector in ECI frame.
        Returns:
            bool: True if the position is within the wake, False otherwise.
        """
        # convert position vector to body frame
        pos_vec_BF =self.pos_ECI2BF(pos_vec)

        behind_back_plane = (self.__wake_back_plane_normal_BF @ (pos_vec_BF) - self.__wake_d_back_BF) >= 0
        behind_plane_1 = (self.__wake_planes_normal_BF[0] @ pos_vec_BF - self.__wake_d_planes_BF[0]) >= 0
        behind_plane_2 = (self.__wake_planes_normal_BF[1] @ pos_vec_BF - self.__wake_d_planes_BF[1]) >= 0
        behind_plane_3 = (self.__wake_planes_normal_BF[2] @ pos_vec_BF - self.__wake_d_planes_BF[2]) >= 0
        behind_plane_4 = (self.__wake_planes_normal_BF[3] @ pos_vec_BF - self.__wake_d_planes_BF[3]) >= 0
        return behind_back_plane and behind_plane_1 and behind_plane_2 and behind_plane_3 and behind_plane_4

    def generate_debris_samples(self, num_samples, search_radius_km):
        """Generate debris samples around the satellite's current position.
            Ensures debris are outside the satellite wake.
        Args:
            num_samples (int): Number of debris samples to generate.
            search_radius_km (float): Radius around the satellite to sample debris positions (km).
            Returns:
                list: List of valid debris sample positions and velocities in ECI frame.
        """

        valid_samples = []

        while len(valid_samples) < num_samples:
            # Generate a batch of random offsets in BODY FRAME
            # Generate slightly more than needed (e.g. 2x) to account for rejection
            batch_size = (num_samples - len(valid_samples)) * 2

            # Random direction
            u = np.random.uniform(0, 1, batch_size)
            v = np.random.uniform(0, 1, batch_size)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            r = search_radius_km * (np.random.uniform(0, 1, batch_size)**(1/3))

            # Convert to Cartesian (Body Frame)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Shape: (N, 3) matrix of random points
            offsets_BF = np.vstack((x, y, z)).T

            # WAKE CHECK
            # Check 1: Back Plane
            # We use broadcasting: (3,) @ (3, N) -> (N,)
            check_back = (offsets_BF @ self.__wake_back_plane_normal_BF - self.__wake_d_back_BF) >= 0
            
            # Check 2-5: Side Planes
            check_1 = (offsets_BF @ self.__wake_planes_normal_BF[0] - self.__wake_d_planes_BF[0]) >= 0
            check_2 = (offsets_BF @ self.__wake_planes_normal_BF[1] - self.__wake_d_planes_BF[1]) >= 0
            check_3 = (offsets_BF @ self.__wake_planes_normal_BF[2] - self.__wake_d_planes_BF[2]) >= 0
            check_4 = (offsets_BF @ self.__wake_planes_normal_BF[3] - self.__wake_d_planes_BF[3]) >= 0
            
            # Combine: Point is in wake if ALL checks are True
            in_wake_mask = check_back & check_1 & check_2 & check_3 & check_4
            
            # Keep only points where in_wake is FALSE
            valid_offsets_BF = offsets_BF[~in_wake_mask]

            # If we found nothing valid in this batch (unlikely), continue
            if len(valid_offsets_BF) == 0:
                continue

            # CONVERT TO ECI
            # Transform survivors from Body Frame to ECI
            # R_BF_to_ECI shape (3,3). valid_offsets shape (M, 3). 
            # Result = (R @ v.T).T
            valid_offsets_ECI = (self.__R_BF_to_ECI @ valid_offsets_BF.T).T
            
            # Add to satellite position
            r_sat_eci = self.__position # current sat position
            r_deb_batch = r_sat_eci + valid_offsets_ECI
            
            # ASSIGN VELOCITY (LVLH Logic)
            # Now that we have valid positions, we assign velocities
            # calculate LVLH basis vectors for the satellite
            h_vec = np.cross(self.__position, self.__velocity)
            u_radial = self.__position / np.linalg.norm(self.__position)
            u_cross = h_vec / np.linalg.norm(h_vec)
            u_along = np.cross(u_cross, u_radial)
            
            
            for i in range(len(r_deb_batch)):
                # Stop if we have enough
                if len(valid_samples) >= num_samples:
                    break
                
                pos = r_deb_batch[i]
                r_mag = np.linalg.norm(pos)
                
                # Circular velocity magnitude
                v_mag = np.sqrt(GM_EARTH / r_mag)
                
                # Random crossing angle
                angle = np.random.uniform(0, 2*np.pi)
                
                # Construct the velocity vector in the LVLH frame
                # It has 0 radial component (circular orbit assumption)
                v_local_radial = 0
                v_local_along  = v_mag * np.cos(angle)
                v_local_cross  = v_mag * np.sin(angle)
                
                # Rotate back to ECI Frame
                v_eci = (v_local_radial * u_radial) + \
                        (v_local_along  * u_along) + \
                        (v_local_cross  * u_cross)
                
                # Add some small random noise to velocity (e.g., 100 m/s std dev)
                v_eci += np.random.normal(0, 0.1, 3) # 100 m/s noise

                # Store
                valid_samples.append(np.concatenate([pos, v_eci]))
            break # exit while loop after one batch for testing

        return np.array(valid_samples)

    def detection_sim(self, no_of_samples, final_time, search_radius_km, save=False):
        '''
        Monte Carlo simulation to generate random debris and check if the generated soliton is detected by the satellite.
        All the solitons are generated at time t=0 for simplicity.
        1. Generate random debris around the satellite
        2. For each debris, generate a soliton from the debris state vectors
        3. Check if the soliton is detected by the satellite sensors
        4. Return the detection results with when and by which sensor
        
        Returns:
            tuple: (debris_samples, detection_results)
                - debris_samples: np.array of shape (no_of_samples, 6) containing [pos, vel]
                - detection_results: list of dicts with keys:
                    - 'debris_id': int, index of debris
                    - 'detected': bool, whether debris was detected
                    - 'first_detection_time': float or None, time of first detection
                    - 'detections': list of dicts with 'sensor_id' and 'time'
        '''

        # generate random debris around the satellite
        debris_samples = self.generate_debris_samples(no_of_samples, search_radius_km)

        # print(f"Generated {len(debris_samples)} debris samples")

        # print("Propagating satellite orbit...")
        # # simulate over time to propagate satellite and solitons
        # # use 1/DETECTION_FREQ time steps
        # time_array = np.arange(0, final_time, 1/DETECTION_FREQ)
        # r_t, v_t = self.__sat_orbit.propagate(final_time, teval=time_array, use_J2=True)
        
        # print("Precomputing rotation matrices for each time step...")
        # # Precompute rotation matrices for all time steps to avoid recomputing in each thread
        # # We can vectorize this or just loop quickly
        # R_stack = np.zeros((len(time_array), 3, 3))
        
        # # Calculate R matrices
        # # 
        # def compute_rotation_matrix(i, pos, vel):
        #     sat_x_BF = vel / np.linalg.norm(vel) if np.linalg.norm(vel) != 0 else np.array([1.0, 0.0, 0.0])
        #     sat_z_BF = -pos / np.linalg.norm(pos) if np.linalg.norm(pos) != 0 else np.array([0.0, 0.0, 1.0])
        #     sat_y_BF = np.cross(sat_z_BF, sat_x_BF)
        #     if np.linalg.norm(sat_y_BF) != 0:
        #         sat_y_BF = sat_y_BF / np.linalg.norm(sat_y_BF)
        #     return np.column_stack([sat_x_BF, sat_y_BF, sat_z_BF])
        
        # R_list = Parallel(n_jobs=-2)(
        #     delayed(compute_rotation_matrix)(i, r_t[i], v_t[i]) 
        #     for i in tqdm(range(len(time_array)), desc="Computing Rotation Matrices")
        # )
        # R_stack[:] = np.array(R_list)

        # # Save propagation data for potential reuse
        # prop_data_dir = os.path.join(os.path.dirname(__file__), "outputs")
        # os.makedirs(prop_data_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # prop_data_path = os.path.join(prop_data_dir, f"propagation_data_{timestamp}.npz")
        # np.savez(prop_data_path, r_t=r_t, v_t=v_t, R_stack=R_stack, time_array=time_array)

       

        print("read satellite propagation data...")
        # Load previously saved propagation data (for testing)
        prop_data_dir = os.path.join(os.path.dirname(__file__), "outputs")
        prop_files = [f for f in os.listdir(prop_data_dir) if f.startswith("propagation_data_") and f.endswith(".npz")]
        if not prop_files:
            raise FileNotFoundError("No propagation data found. Please run the simulation once to generate propagation data.")
        latest_prop_file = max(prop_files, key=lambda f: os.path.getctime(os.path.join(prop_data_dir, f)))
        prop_data_path = os.path.join(prop_data_dir, latest_prop_file)
        prop_data = np.load(prop_data_path, allow_pickle=True)
        print(f"Loaded propagation data from {prop_data_path}")
        r_t = prop_data['r_t']
        R_stack = prop_data['R_stack']
        time_array = prop_data['time_array']

        print("Checking detections for each debris sample in parallel...")
        # Parallelize the debris check loop
        # We process each debris independently across the entire time array
        results = Parallel(n_jobs=40)(
            delayed(process_debris_task)(
                debris_id, debris, time_array, r_t, R_stack, self.__sensors_BF
            ) 
            for debris_id, debris in enumerate(tqdm(debris_samples, desc="Processing Debris"))
        )
        
        if save:
            out_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            npz_path = os.path.join(out_dir, f"detection_sim_{timestamp}.npz")
            save_results(npz_path, debris_samples, results)

        return debris_samples, results


    def find_percent_type2_detected(self, detection_samples, final_time, search_radius_km):
        """
        Find debris samples that have detections at differing times AND on differing sensors.

        Returns:
            float: percentage of debris samples that have detections at differing times AND on differing sensors
        """

        debris_samples, results = self.detection_sim(detection_samples, final_time, search_radius_km, save=True)

        multi_detected_debris = []
    
        # helper to handle both object (dict) and numpy void types if loaded from npz
        def get_val(item, key):
            if isinstance(item, dict):
                return item[key]
            # if it's a structured array or object from np.load
            return item[key]

        for res in results:
            # Check if detected
            if isinstance(res, dict):
                if not res['detected']:
                    continue
                detections = res['detections']
                debris_id = res['debris_id']
            else:
                # Handle case where numpy might return a zero-d array wrapper around the dict
                # or if accessing keys works differently
                if not res['detected']:
                    continue
                detections = res['detections']
                debris_id = res['debris_id']

            if not detections or len(detections) < 2:
                continue
                
            found_pair = False
            # Iterate through all unique pairs of detections to find a match
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    d1 = detections[i]
                    d2 = detections[j]
                    
                    # Check for different times AND different sensors
                    # Using 1e-6 tolerance for time comparison just in case, though != likely works for exact steps
                    time_diff = abs(d1['time'] - d2['time']) > 1e-6
                    sensor_diff = d1['sensor_id'] != d2['sensor_id']
                    
                    if time_diff and sensor_diff:
                        found_pair = True
                        break
                if found_pair:
                    break
            
            if found_pair:
                multi_detected_debris.append(debris_id)

        percent_type2_detected = len(multi_detected_debris) / detection_samples
        return percent_type2_detected
    
    def pos_BF2ECI(self, pos_BF):
        """Convert a position vector from body frame to ECI frame."""
        return self.__R_BF_to_ECI @ pos_BF + self.__position0
    
    def pos_ECI2BF(self, pos_ECI):
        """Convert a position vector from ECI frame to body frame."""
        return self.__R_BF_to_ECI.T @ (pos_ECI - self.__position0)
    
    def vec_ECI2BF(self, vec_ECI):
        """Convert a vector from ECI frame to body frame."""
        return self.__R_BF_to_ECI.T @ vec_ECI
    
    def vec_BF2ECI(self, vec_BF):
        """Convert a vector from body frame to ECI frame."""
        return self.__R_BF_to_ECI @ vec_BF

    # define rotation matrix from body frame to ECI frame based on current satellite position and velocity
    # and update all sensor positions in ECI frame
    def __BF_to_ECI(self):
        """Compute the rotation matrix from Body Frame to ECI frame based on current position and velocity.
        Updates the internal rotation matrix attribute."""
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

def process_debris_task(debris_id, debris, time_array, r_t, R_stack, sensors_BF):
    """
    Helper function to process a single debris sample over the entire time array.
    """
    debris_pos = debris[0:3]
    debris_vel = debris[3:6]
    
    # generate soliton from debris state vectors
    soliton = Soliton.from_debris_state(debris_pos, debris_vel, 0)
    
    detections = []
    detected = False
    first_detection_time = None
    
    # check if any sensor detects the soliton at current time
    for t_idx, current_time in enumerate(time_array):
        # Satellite State
        sat_pos = r_t[t_idx] # Current ECI position
        R_mat = R_stack[t_idx]
        
        # Check all sensors
        for sensor_id in range(len(sensors_BF)):
            # convert sensor pos to ECI
            # Using current R and sat_pos (assuming sat_pos is the origin of Body Frame in ECI)
            sensor_pos_ECI = R_mat @ sensors_BF[sensor_id] + sat_pos 
            
            is_detected = soliton.within_soliton_shell(sensor_pos_ECI, current_time)
            
            if is_detected:
                 # Record detection
                 # Since we iterate sequentially, just append
                 detections.append({
                    'sensor_id': sensor_id,
                    'time': current_time
                 })
                 
                 if not detected:
                     detected = True
                     first_detection_time = current_time

    return {
        'debris_id': debris_id,
        'detected': detected,
        'first_detection_time': first_detection_time,
        'detections': detections
    }

def save_results(filename, debris_samples, detection_results):
    """
    Save the debris samples and detection results to a .npz file in the results folder.
    
    Args:
        filename (str): The name of the file to save (should end in .npz).
        debris_samples (np.array): Array of debris samples.
        detection_results (list): List of detection results (dictionaries).
    """
    # Determine the results directory relative to this script
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Use only the basename of the filename to ensure it saves in results_dir
    save_path = os.path.join(results_dir, os.path.basename(filename))

    # Ensure detection_results is wrapped in a way that preserves the dictionary structure
    # Using np.array with object dtype
    np.savez(save_path, debris_samples=debris_samples, detection_results=np.array(detection_results, dtype=object))
    print(f"Results saved to {save_path}")

def load_results(filename):
    """
    Load the debris samples and detection results from a .npz file.
    
    Args:
        filename (str): The path to the file to load.
        
    Returns:
        tuple: (debris_samples, detection_results)
    """
    with np.load(filename, allow_pickle=True) as data:
        debris_samples = data['debris_samples']
        # detection_results was saved as a 1D array of objects (dicts)
        detection_results = data['detection_results']
        
        # If it was saved as an object array, it might be returned as such.
        # Check if we need to convert back to a list
        if isinstance(detection_results, np.ndarray):
            detection_results = detection_results.tolist()
            
    return debris_samples, detection_results