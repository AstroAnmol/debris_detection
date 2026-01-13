import numpy as np
from src.orbit import *
# from monte_carlo import generate_forced_debris
from src.satellite import Satellite
from src.soliton import Soliton
from src.constants import *
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Satellite orbital elements: [a (km), e, i (rad), Omega (rad), omega (rad), nu (rad)]
OE = np.array([750 + R_EARTH, 0.063, np.radians(45), np.radians(0), np.radians(0), np.radians(0)])

OE_debris = np.array([750 + R_EARTH, 0.063, np.radians(180-45), np.radians(180+0), np.radians(180+0), np.radians(0-0.1)])

sat_orbit = Orbit.from_OE(OE)

deb_orbit = Orbit.from_OE(OE_debris)

sat_obj =Satellite()
sat_obj.set_sat_orbit_OE(OE)

debris_samples, detection_results = sat_obj.detection_sim(no_of_samples=10, final_time=10, search_radius_km=1)

out_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
npz_path = os.path.join(out_dir, f"detection_sim_{timestamp}.npz")
np.savez(npz_path, debris_samples=debris_samples, detection_results=np.array(detection_results))
print(f"Saved debris_samples and detection_results to: {npz_path}")


# print("Detection results over time (rows: time steps, columns: debris samples):")
# print(np.array(detection_results))

# deb_pos_BF = sat_obj.pos_ECI2BF(deb_orbit.get_SV()[0])
# deb_vel_BF = sat_obj.vec_ECI2BF(deb_orbit.get_SV()[1])

# sol_obj = Soliton.from_parameters(
#     angle=np.deg2rad(45),
#     height=10,
#     vel_multiplier=1.2,
#     time_of_generation=0,
#     pos_vec= deb_pos_BF,
#     vel_vec= deb_vel_BF
# )


# print("soliton shell thickness (km):", sol_obj.shell_thickness)

# print("Satellite position:", sat_orbit.get_SV()[0], "velocity:", sat_orbit.get_SV()[1])
# print("Debris position:", deb_orbit.get_SV()[0], "velocity:", deb_orbit.get_SV()[1])



# sat_obj.detect_type1_soliton(sol_obj)

# # sat_orbit = Orbit.from_OE(OE)
# time_period = sat_orbit.get_time_period()
# print("Orbital Time Period (hr):", time_period / 3600)

# omega_rate = sat_orbit.get_omega_precession_rate()
# omega_precession_time_cycle = 2 * np.pi / omega_rate 
# print("Argument of Perigee Precession Time Cycle (days):", omega_precession_time_cycle/86400)


# tspan = omega_precession_time_cycle # s
# time_step = 60.0  # time step in seconds
# # evaluation times (seconds) for the propagation — keep for animation time display
# teval = np.arange(0, tspan, time_step)
# positions, velocities = sat_orbit.propagate(tspan, teval=teval, use_J2=True)

# print("--------------------------------")
# print("Starting main simulation loop...")

# for i in range(teval.shape[0]):
#     t = teval[i]
#     print(f"Current time: {t/86400:.2f} days")
#     # create satellite instance
#     satellite = Satellite()
#     sat_position = positions[i]
#     r_sat_mag = np.linalg.norm(sat_position)
#     sat_velocity = velocities[i]
#     satellite.set_sat_orbit_SV(sat_position, sat_velocity)
#     print("Satellite Position (km):", sat_position, "Velocity (km/s):", sat_velocity)
#     print("Satellite altitude (km):", r_sat_mag)
#     print("--------------------------------")
#     print("Generating debris samples...")

#     num_samples = 1
#     search_radius_km = 0.001
#     SVs_array = satellite.generate_debris_samples(num_samples=num_samples, search_radius_km=search_radius_km)
   

#     # loop through debris samples and check for soliton detection
#     for j in range(num_samples):
        
#         print("--------------------------------")
#         print(f"Debris sample {j}:")
#         debris_SVs = SVs_array[j, :]
#         # debris_position = debris_SVs[0:3]
#         # debris_velocity = debris_SVs[3:6]

#         orbit_debris = Orbit.from_OE(OE_debris)
#         debris_SV = orbit_debris.get_SV()
#         debris_position = debris_SV[0]
#         debris_velocity = debris_SV[1]

#         print("Debris Position (km):", debris_position, "Velocity (km/s):", debris_velocity)
#         print("Debris altitude (km):", np.linalg.norm(debris_position))
#         # print("Debris position in Body Frame (km):", debris_pos_BF)

#         satellite_prop = Satellite()
#         satellite_prop.set_sat_orbit_SV(sat_position, sat_velocity)
#         debris_pos_BF = satellite_prop.pos_ECI2BF(debris_position)

#         print("Debris position in Body Frame (km):", debris_pos_BF)

#         # create soliton instance from debris state vectors
#         soliton = Soliton.from_parameters(
#             angle=np.deg2rad(45),
#             height=2,
#             vel_multiplier=1.2,
#             time_of_generation=t,
#             pos_vec=debris_position,
#             vel_vec=debris_velocity
#         )

#         # # plot satellite point and the debris samples
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')
        
#         satellite_prop.detect_type1_soliton(soliton, ax=ax)
        
    #     #every 100 samples, print progress
    #     if (j+1) % 100 == 0:
    #         percent_complete = (j+1) / num_samples * 100
    #         print(f"Processed {percent_complete:.1f}% debris samples...")

    #     ax.set_xlabel('X (km)')
    #     ax.set_ylabel('Y (km)')
    #     ax.set_zlabel('Z (km)')
    #     ax.set_title(f'Debris Samples at t = {t/86400:.2f} days')
    #     ax.legend()
    #     plt.show()
    # break



# def plot_orbit_3D_animation(positions, teval):
#     # animate the orbit trajectory in 3D as a growing line with a moving marker
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')

#     # set plot limits
#     ax.set_xlim([-R_EARTH - 2000, R_EARTH + 2000])
#     ax.set_ylim([-R_EARTH - 2000, R_EARTH + 2000])
#     ax.set_zlim([-R_EARTH - 2000, R_EARTH + 2000])
#     ax.set_xlabel('X (km)')
#     ax.set_ylabel('Y (km)')
#     ax.set_zlabel('Z (km)')
#     ax.set_title('Orbit Trajectory (animated)')

#     # line for the trajectory (initially empty)
#     line, = ax.plot([], [], [], lw=1.0, color='tab:blue')
#     # moving marker for the current position
#     marker, = ax.plot([], [], [], marker='o', color='tab:orange', markersize=4)
#     # overlay text (2D axes) for current time in days
#     time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

#     def init():
#         line.set_data([], [])
#         getattr(line, 'set_3d_properties')([])
#         marker.set_data([], [])
#         getattr(marker, 'set_3d_properties')([])
#         time_text.set_text("")
#         return line, marker, time_text

#     def update(frame):
#         # draw trajectory up to current frame
#         xs = positions[: frame + 1, 0]
#         ys = positions[: frame + 1, 1]
#         zs = positions[: frame + 1, 2]
#         line.set_data(xs, ys)
#         getattr(line, 'set_3d_properties')(zs)

#         # update moving marker at current location
#         marker.set_data([xs[-1]], [ys[-1]])
#         getattr(marker, 'set_3d_properties')([zs[-1]])

#         # update time display (convert seconds -> days)
#         current_days = teval[frame] / 86400.0
#         time_text.set_text(f"Time: {current_days:.2f} days")

#         return line, marker, time_text

#     from matplotlib.animation import FuncAnimation
#     ani = FuncAnimation(fig, update, frames=positions.shape[0], init_func=init, interval=20, blit=False)
#     plt.show()


