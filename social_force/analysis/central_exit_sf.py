from crowd_dynamics.social_force import SocialForce
import numpy as np
import postprocessing.plotting
import postprocessing.movie
import matplotlib.pyplot as plt

###############################
# Initialise simulation class #
###############################
n_pedestrians=300
SF = SocialForce(n_pedestrians)
#SF.A2 = 5.

##########################
# Initialise pedestrians #
##########################
### Positions ###
p0 = np.empty((n_pedestrians,2))

def rot(th):
    c,s = np.cos(th/2), np.sin(th/2)
    return np.array([[c,-s],[s,c]])

radius=45 # end semicircle length

# Sample all positions in a rectangle with a semicircle at the end
i=0
while i < n_pedestrians:
    x, y = SF.rng.uniform(-radius, radius), SF.rng.uniform(-radius, radius)
    in_circle = (x**2 + y**2 <= radius**2)
    if in_circle:
        p0[i] = x,y
        i += 1

### velocities ###
v0 = np.zeros_like(p0)

### Destinations ###
destinations = np.zeros((n_pedestrians,2,2))
angles = np.linspace(0, 2*np.pi, n_pedestrians, endpoint=False)
r = 1e6
destinations[:,1] = np.column_stack((r * np.cos(angles), r*np.sin(angles)))
destinations_range = np.zeros((n_pedestrians, 2,1))
destinations_range[:,0,0] = 0.35

SF.init_pedestrians(p0, destinations, velocities=v0, destinations_range=destinations_range)

##################
# Run simulation #
##################
SF.init_solver(t_bound=80)
to_save=("positions",)

results = {key: [] for key in to_save}
SF.results = results
times = []
i = 0 
current_destinations = SF.current_destinations()
initial_radial_positions = np.linalg.norm(SF.positions, axis=1)
time_to_exit = np.empty((n_pedestrians,))
moved = np.zeros((n_pedestrians,), dtype=bool)
k=0
while SF.t < 60 and SF.solver.status == 'running':
    if i%100 == 0:
        print(i)
    SF.step()
    if "positions" in to_save:
        SF.results["positions"].append(SF.positions.copy())
    if "velocities" in to_save:
        SF.results["velocities"].append(SF.velocities.copy())
    if "desired_speeds" in to_save:
        SF.results["desired_speeds"].append(SF.desired_speeds.copy())
    if "desired_directions" in to_save:
        SF.results["desired_directions"].append(SF.calc_desired_directions())
    if "total_forces" in to_save:
        SF.results["total_forces"].append(SF.forces["total"])
    if "repulsive_forces" in to_save:
        SF.results["repulsive_forces"].append(SF.forces["repulsive"])
    if "driving_forces" in to_save:
        SF.results["driving_forces"].append(SF.forces["driving"])
    if "boundary_forces" in to_save:
        SF.results["boundary_forces"].append(SF.forces["boundary"])

    for j in range(n_pedestrians):
        if SF.destinations_indices[j] == 1 and not moved[j]:
            SF.positions[j] = destinations[j,1]
            moved[j] = True
            time_to_exit[j] = k
            k+=1
    times.append(SF.t)
    i+=1
#    mask = (current_destinations[:,0] != SF.current_destinations()[:,0])
#    current_destinations = SF.current_destinations()
#    time_to_exit[mask] = SF.t


positions = results["positions"]

################
# Grab Popcorn #
################
regularised_timesteps = np.linspace(times[0], times[-1], len(times))
'''
postprocessing.movie.make_movie(times, positions, SF,
                                regularised_timesteps=regularised_timesteps,
                                title="Stripe formation in intersection flows",
                                interval=50,
                                x_bound=(-45,45), y_bound=(-45, 45))
'''
#postprocessing.plotting.plot(positions[-1], SF, title="Striping in Intersecting Flows", x_bound=(-45,45), y_bound=(-45,45), filetype='pdf')
plt.scatter(initial_radial_positions, time_to_exit)
plt.show()
