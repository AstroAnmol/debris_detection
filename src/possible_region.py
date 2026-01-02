import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull
import itertools

def check_conditions(C, P1, P2, t, H, theta):
    """
    Check if a center C satisfies the conditions for points P1 and P2.
    
    Conditions:
    1. Spherical Shell: |dist(C, P1) - dist(C, P2)| <= t
    2. Cone: Exists an axis such that P1, P2 are in cone (angle <= theta, proj <= H)
       Approximation: Check if bisector axis works.
       - Angle between (P1-C) and (P2-C) <= 2*theta
       - Projection of (P1-C) on bisector <= H
       - Projection of (P2-C) on bisector <= H
    """
    vec1 = P1 - C
    vec2 = P2 - C
    d1 = np.linalg.norm(vec1)
    d2 = np.linalg.norm(vec2)
    
    # 1. Shell Condition
    if abs(d1 - d2) > t:
        return False
    
    # Handle coincident points or C at P1/P2
    if d1 == 0 or d2 == 0:
        return True # Technically inside if H>0
        
    u1 = vec1 / d1
    u2 = vec2 / d2
    
    # Dot product for angle
    dot_prod = np.clip(np.dot(u1, u2), -1.0, 1.0)
    alpha = np.arccos(dot_prod)
    
    # 2. Cone Angle Condition
    # The angle between vectors must be <= 2 * theta
    if alpha > 2 * theta:
        return False
        
    # 3. Cone Height Condition (using bisector approximation)
    # The bisector axis minimizes the max angle to both vectors.
    # Angle from bisector to each vector is alpha/2.
    # Projection = dist * cos(alpha/2)
    # We need projection <= H
    
    proj1 = d1 * np.cos(alpha / 2.0)
    proj2 = d2 * np.cos(alpha / 2.0)
    
    if proj1 > H or proj2 > H:
        return False
        
    return True

def plot_possible_regions(P1, P2, t, H, theta_deg, grid_res=20, padding=1.5):
    """
    Plot the possible regions for a given set of parameters.
    
    Args:
        P1 (np.array): First point in 3D space.
        P2 (np.array): Second point in 3D space.
        t (float): Thickness of the shell in km.
        H (float): Height of the cone in km.
        theta_deg (float): Angle of the cone in degrees.
        grid_res (int, optional): Resolution of the grid. Defaults to 20.
        padding (float, optional): Padding around the points. Defaults to 1.5.
    """
    theta = np.deg2rad(theta_deg)
    
    # Define grid bounds
    points = np.vstack([P1, P2])
    center = np.mean(points, axis=0)
    dist = np.linalg.norm(P1 - P2)
    
    # Heuristic for bounds: 
    # Shell constraint limits along hyperboloid axis, but not perpendicular?
    # Actually, if t is large, it can go far.
    # Cone height constraint limits distance to roughly H.
    # So bounds around H + dist should be enough.
    max_range = max(H, dist) * padding
    
    # Generate grid for voxel corners (N+1 points)
    x_corners = np.linspace(center[0] - max_range, center[0] + max_range, grid_res + 1)
    y_corners = np.linspace(center[1] - max_range, center[1] + max_range, grid_res + 1)
    z_corners = np.linspace(center[2] - max_range, center[2] + max_range, grid_res + 1)
    
    X_corners, Y_corners, Z_corners = np.meshgrid(x_corners, y_corners, z_corners, indexing='ij')
    
    # Generate grid for voxel centers (N points) for evaluation
    x_centers = (x_corners[:-1] + x_corners[1:]) / 2.0
    y_centers = (y_corners[:-1] + y_corners[1:]) / 2.0
    z_centers = (z_corners[:-1] + z_corners[1:]) / 2.0
    
    X_centers, Y_centers, Z_centers = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    
    # Vectorize check? Or just loop for clarity/simplicity
    valid_mask = np.zeros_like(X_centers, dtype=bool)
    
    print(f"Checking {grid_res**3} points...")
    
    for i in range(X_centers.shape[0]):
        for j in range(X_centers.shape[1]):
            for k in range(X_centers.shape[2]):
                C = np.array([X_centers[i,j,k], Y_centers[i,j,k], Z_centers[i,j,k]])
                if check_conditions(C, P1, P2, t, H, theta):
                    valid_mask[i,j,k] = True
                    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot P1, P2
    ax.scatter(*P1, c='r', marker='o', s=100, label='P1')
    ax.scatter(*P2, c='b', marker='^', s=100, label='P2')
    
    # Plot valid region
    # Use voxels for 3D volume
    # Pass corners to voxels
    ax.voxels(X_corners, Y_corners, Z_corners, valid_mask, edgecolor='k', alpha=0.3, facecolors='cyan')
    
    # Alternative: Scatter plot of valid centers
    # valid_points = np.column_stack([X[valid_mask], Y[valid_mask], Z[valid_mask]])
    # if len(valid_points) > 0:
    #     ax.scatter(valid_points[:,0], valid_points[:,1], valid_points[:,2], c='g', alpha=0.1, s=10, label='Valid Centers')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Possible Centers Region\nt={t}, H={H}, theta={theta_deg} deg')
    ax.legend()
    
    # Set equal aspect ratio
    # ax.set_box_aspect([1,1,1]) 
    plt.show()
    # plt.savefig('possible_region.png')
    # print("Plot saved to possible_region.png")

def plot_four_points_intersection(points, t, H, theta_deg, grid_res=20, padding=1.5):
    """
    Plot the intersection of possible regions for every pair of 4 points.
    
    Args:
        points (list of np.array): List of 4 points in 3D space.
        t (float): Thickness of the shell in km.
        H (float): Height of the cone in km.
        theta_deg (float): Angle of the cone in degrees.
        grid_res (int, optional): Resolution of the grid. Defaults to 20.
        padding (float, optional): Padding around the points. Defaults to 1.5.
    """
    if len(points) != 4:
        print("Error: Exactly 4 points are required.")
        return

    theta = np.deg2rad(theta_deg)
    
    # Define grid bounds based on all points
    points_arr = np.vstack(points)
    center = np.mean(points_arr, axis=0)
    
    # Calculate max distance between any pair to set bounds
    max_dist = 0
    import itertools
    for p1, p2 in itertools.combinations(points, 2):
        d = np.linalg.norm(p1 - p2)
        if d > max_dist:
            max_dist = d
            
    max_range = max(H, max_dist) * padding
    
    # Generate grid for voxel corners (N+1 points)
    x_corners = np.linspace(center[0] - max_range, center[0] + max_range, grid_res + 1)
    y_corners = np.linspace(center[1] - max_range, center[1] + max_range, grid_res + 1)
    z_corners = np.linspace(center[2] - max_range, center[2] + max_range, grid_res + 1)
    
    X_corners, Y_corners, Z_corners = np.meshgrid(x_corners, y_corners, z_corners, indexing='ij')
    
    # Generate grid for voxel centers (N points) for evaluation
    x_centers = (x_corners[:-1] + x_corners[1:]) / 2.0
    y_centers = (y_corners[:-1] + y_corners[1:]) / 2.0
    z_centers = (z_corners[:-1] + z_corners[1:]) / 2.0
    
    X_centers, Y_centers, Z_centers = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    
    # Initialize mask as all True (intersection identity)
    intersection_mask = np.ones_like(X_centers, dtype=bool)
    
    print(f"Checking {grid_res**3} points for intersection of all pairs...")
    
    # Iterate over all unique pairs
    pairs = list(itertools.combinations(points, 2))
    
    for idx, (P1, P2) in enumerate(pairs):
        print(f"Processing pair {idx+1}/{len(pairs)}...")
        # Update mask: AND with current pair condition
        # Optimization: We could skip checking if intersection_mask is already False, 
        # but iterating is simple.
        
        for i in range(X_centers.shape[0]):
            for j in range(X_centers.shape[1]):
                for k in range(X_centers.shape[2]):
                    if intersection_mask[i,j,k]: # Only check if still valid
                        C = np.array([X_centers[i,j,k], Y_centers[i,j,k], Z_centers[i,j,k]])
                        if not check_conditions(C, P1, P2, t, H, theta):
                            intersection_mask[i,j,k] = False

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Points
    colors = ['r', 'b', 'g', 'm']
    markers = ['o', '^', 's', 'D']
    for i, P in enumerate(points):
        ax.scatter(*P, c=colors[i], marker=markers[i], s=80, label=f'P{i+1}')
    
    # Plot valid intersection region
    ax.voxels(X_corners, Y_corners, Z_corners, intersection_mask, edgecolor='k', alpha=0.3, facecolors='cyan')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Intersection Region (4 Points)\nt={t}, H={H}, theta={theta_deg} deg')
    ax.legend()
    
    # plt.savefig('possible_region_intersection.png')
    # print("Plot saved to possible_region_intersection.png")
    pass

def get_intersection_vertices(normals, constants, R_BF_to_ECI, limit=0.001):
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
        # print("Not enough vertices to plot a 3D shape.")
        return

    try:
        # Compute the Convex Hull to get the faces/edges order
        hull = ConvexHull(vertices)
        
        faces = [vertices[simplex] for simplex in hull.simplices]
        
        # Plot Faces
        poly = Poly3DCollection(faces, alpha=0.2, edgecolor='k', linewidths=0.5)
        poly.set_facecolor('gray')
        ax.add_collection3d(poly)
        
    except Exception as e:
        print(f"Error computing hull: {e}")

def plot_satellite_geometry(ax, corners_BF, sensors_BF, wake_normals, wake_constants):
    """
    Plot satellite body, booms, and wake.
    """
    # Plot Body (Cuboid)
    # corners_BF has 8 corners.
    # Edges:
    edges = [
        [corners_BF[0], corners_BF[1]], [corners_BF[1], corners_BF[2]], [corners_BF[2], corners_BF[3]], [corners_BF[3], corners_BF[0]], # Top face (z+)
        [corners_BF[4], corners_BF[5]], [corners_BF[5], corners_BF[6]], [corners_BF[6], corners_BF[7]], [corners_BF[7], corners_BF[4]], # Bottom face (z-)
        [corners_BF[0], corners_BF[6]], [corners_BF[1], corners_BF[7]], [corners_BF[2], corners_BF[4]], [corners_BF[3], corners_BF[5]]  # Vertical edges (connecting top/bottom) - Wait, indices need to match geometry
    ]

    
    edges_indices = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,6], [1,7], [2,4], [3,5]
    ]
    
    edges_coords = [[corners_BF[i], corners_BF[j]] for i, j in edges_indices]
    lc = Line3DCollection(edges_coords, colors='tab:orange', linewidths=1)
    ax.add_collection3d(lc)

    
    boom_edges = []
    for i in range(4):
        boom_edges.append([corners_BF[i], sensors_BF[i]])
        
    lc_booms = Line3DCollection(boom_edges, colors='tab:orange', linewidths=1)
    ax.add_collection3d(lc_booms)
    
    # Plot Wake
    # We need to combine wake normals and constants
    # Assuming R_BF_to_ECI is identity
    verts = get_intersection_vertices(wake_normals, wake_constants, np.eye(3), limit=0.001)
    if verts is not None and len(verts) >= 3:
        plot_polyhedron(ax, verts)

def possible_region(points, t, H, theta_deg, 
                                      wake_back_n, wake_back_d, 
                                      wake_side_n, wake_side_d,
                                      sat_corners=None, 
                                      grid_res=20, padding=1.5, if_plot=True):
    """
    Plot intersection of possible regions for 4 points, removing the wake region.
    """
    if len(points) != 4:
        print("Error: Exactly 4 points are required.")
        return

    theta = np.deg2rad(theta_deg)
    
    # Define grid bounds based on all points
    points_arr = np.vstack(points)
    center = np.mean(points_arr, axis=0)
    
    # Calculate max distance between any pair to set bounds
    max_dist = 0
    import itertools
    for p1, p2 in itertools.combinations(points, 2):
        d = np.linalg.norm(p1 - p2)
        if d > max_dist:
            max_dist = d
            
    max_range = max(H, max_dist) * padding
    
    # Generate grid for voxel corners (N+1 points)
    x_corners = np.linspace(center[0] - max_range, center[0] + max_range, grid_res + 1)
    y_corners = np.linspace(center[1] - max_range, center[1] + max_range, grid_res + 1)
    z_corners = np.linspace(center[2] - max_range, center[2] + max_range, grid_res + 1)
    
    X_corners, Y_corners, Z_corners = np.meshgrid(x_corners, y_corners, z_corners, indexing='ij')
    
    # Generate grid for voxel centers (N points) for evaluation
    x_centers = (x_corners[:-1] + x_corners[1:]) / 2.0
    y_centers = (y_corners[:-1] + y_corners[1:]) / 2.0
    z_centers = (z_corners[:-1] + z_corners[1:]) / 2.0
    
    X_centers, Y_centers, Z_centers = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    
    # Initialize mask as all True (intersection identity)
    intersection_mask = np.ones_like(X_centers, dtype=bool)
    
    print(f"Checking {grid_res**3} points for intersection and wake removal...")
    
    # 1. Check Intersection of Pairs
    pairs = list(itertools.combinations(points, 2))
    for idx, (P1, P2) in enumerate(pairs):
        # print(f"Processing pair {idx+1}/{len(pairs)}...")
        for i in range(X_centers.shape[0]):
            for j in range(X_centers.shape[1]):
                for k in range(X_centers.shape[2]):
                    if intersection_mask[i,j,k]:
                        C = np.array([X_centers[i,j,k], Y_centers[i,j,k], Z_centers[i,j,k]])
                        if not check_conditions(C, P1, P2, t, H, theta):
                            intersection_mask[i,j,k] = False

    # 2. Remove Wake
    # A point is in the wake if it satisfies ALL wake plane inequalities.
    # We want to keep points that are NOT in the wake.
    # So if in_wake is True, set mask to False.
    
    wake_removed_count = 0
    for i in range(X_centers.shape[0]):
        for j in range(X_centers.shape[1]):
            for k in range(X_centers.shape[2]):
                if intersection_mask[i,j,k]:
                    C = np.array([X_centers[i,j,k], Y_centers[i,j,k], Z_centers[i,j,k]])
                    
                    # Check wake conditions
                    # Note: C is the potential debris position.
                    # The wake is defined in Body Frame. 
                    # Assuming points and C are all in Body Frame (since we passed BF sensors).
                    
                    behind_back = (wake_back_n @ C - wake_back_d) >= 0
                    behind_sides = True
                    for n, d in zip(wake_side_n, wake_side_d):
                        if (n @ C - d) < 0:
                            behind_sides = False
                            break
                    
                    in_wake = behind_back and behind_sides
                    
                    if in_wake:
                        intersection_mask[i,j,k] = False
                        wake_removed_count += 1

    print(f"Removed {wake_removed_count} points due to wake.")
    
    # Calculate percentage of valid points
    total_points = grid_res**3
    valid_points = np.sum(intersection_mask)
    percentage = (valid_points / total_points) * 100
    print(f"Valid region: {valid_points}/{total_points} points ({percentage:.2f}%)")

    # Plot only if requested
    if not if_plot:
        return percentage
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Satellite Body and Wake
    if sat_corners is not None:
        # Combine wake planes
        all_wake_normals = [wake_back_n] + list(wake_side_n)
        all_wake_constants = [wake_back_d] + list(wake_side_d)
        
        # We need to construct the full corners array (8 corners)
        # sat_corners passed in might be just the 4 used for sensors?
        # In main, corners_BF is 4 points.
        # But for the full body we need 8.
        # Let's reconstruct the other 4 if only 4 are passed, or assume 8 passed.
        # In main: corners_BF = np.array([[ 1, 1, 1],[ 1,-1, 1],[ 1,-1,-1],[ 1, 1,-1]]) * sat_size / 2
        # These are the x+ face corners.
        # The x- face corners are -corners_BF (symmetric).
        # Actually, let's look at archive/satellite.py:
        # corners = [c1, c2, c3, c4, -c1, -c2, -c3, -c4]
        
        if len(sat_corners) == 4:
            full_corners = np.vstack([sat_corners, -sat_corners])
        else:
            full_corners = sat_corners
            
        plot_satellite_geometry(ax, full_corners, points, all_wake_normals, all_wake_constants)

    # # Plot Points
    # colors = ['r', 'b', 'g', 'm']
    # markers = ['o', '^', 's', 'D']
    # for i, P in enumerate(points):
    #     ax.scatter(*P, c=colors[i], marker=markers[i], s=80, label=f'P{i+1}')
    
    # Plot valid intersection region
    ax.voxels(X_corners, Y_corners, Z_corners, intersection_mask, edgecolor='k', alpha=0.3, facecolors='cyan')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Intersection with Wake Removal\nt={t*1000} m, H={H*1000} m, theta={theta_deg} deg')
    ax.legend()
    
    # plt.savefig('possible_region_wake_removed.png')
    # print("Plot saved to possible_region_wake_removed.png")
    plt.show()
    return percentage

if __name__ == "__main__":
    sat_size = np.array([ 0.03, 0.02, 0.02 ]) * 1e-3 # 3 X 2 X 2 cm (in km)
    boom_length = 1 * 1e-3 # 1 m (in km)
    corners_BF = np.array([[ 1, 1, 1],[ 1,-1, 1],[ 1,-1,-1],[ 1, 1,-1]]) * sat_size / 2
    sensors_BF = corners_BF + boom_length * np.array([[ 1, 1, 1],[ 1,-1, 1],[ 1,-1,-1],[ 1, 1,-1]]) / np.sqrt(3)
    
    points = sensors_BF
    
    thickness = 0.001 # km    
    cone_height = 0.01 # km
    cone_angle_deg = 45 # degrees

    # wake parameters
    plane_angle = 45.0 * np.pi / 180.0

    wake_back_plane_normal_BF = np.array([-1.0, 0.0, 0.0])
    wake_d_back_BF = sat_size[0] / 2.0

    wake_planes_normal_BF = np.array([
            [-np.sin(plane_angle), 0.0, -np.cos(plane_angle)],
            [-np.sin(plane_angle), 0.0,  np.cos(plane_angle)],
            [-np.sin(plane_angle),  np.cos(plane_angle), 0.0],
            [-np.sin(plane_angle), -np.cos(plane_angle), 0.0]
        ])
    wake_d_planes_BF = np.array([
            sat_size[0] / 2 * np.sin(plane_angle) - sat_size[2] / 2 * np.cos(plane_angle),
            sat_size[0] / 2 * np.sin(plane_angle) - sat_size[2] / 2 * np.cos(plane_angle),
            sat_size[0] / 2 * np.sin(plane_angle) - sat_size[1] / 2 * np.cos(plane_angle),
            sat_size[0] / 2 * np.sin(plane_angle) - sat_size[1] / 2 * np.cos(plane_angle)
        ])
    
    # plot_possible_regions(points[0], points[1], thickness, cone_height, cone_angle_deg, grid_res=100, padding=2.0)
    # plot_four_points_intersection(points, thickness, cone_height, cone_angle_deg, grid_res=30, padding=2.0)
    
    percentage = possible_region(points, thickness, cone_height, cone_angle_deg,
                                      wake_back_plane_normal_BF, wake_d_back_BF,
                                      wake_planes_normal_BF, wake_d_planes_BF,
                                      sat_corners=corners_BF,
                                      grid_res=100, padding=1.0, if_plot=False)
    
    print(f"Valid region percentage after wake removal: {percentage:.4f}%")