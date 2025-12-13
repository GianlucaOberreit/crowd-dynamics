from crowd_dynamics.social_force import SocialForce
import numpy as np
import postprocessing.plotting
import postprocessing.movie

###############################
# Initialise simulation class #
###############################
n_pedestrians=5
SF = SocialForce(n_pedestrians)

##########################
# Initialise pedestrians #
##########################
### Positions ###
p0 = np.empty((n_pedestrians,2))

def rot(th):
    c,s = np.cos(th/2), np.sin(th/2)
    return np.array([[c,-s],[s,c]])

radius=4 # end semicircle length

# Sample all positions in a rectangle with a semicircle at the end
i=0
while i < n_pedestrians:
    x, y = SF.rng.uniform(-radius, radius), SF.rng.uniform(-radius, radius)
    in_circle = (x**2 + y**2 <= radius**2)
    if in_circle:
        p0[i] = x,y
        i += 1

postprocessing.plotting.plot(p0, SF)

### velocities ###
v0 = np.zeros_like(p0)

### Destinations ###
destinations = np.zeros((n_pedestrians,2,2))
destinations[:,1,0] = 100000
destinations_range = np.zeros((n_pedestrians, 2,1))
destinations_range[:,0,0] = 0.3

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
while SF.t < 80 and SF.solver.status == 'running':
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

    for i in range(n_pedestrians):
        if SF.destinations_indices[i] == 1:
            SF.positions[i] = (100000, 0)
    times.append(SF.t)
    i+=1


positions = results["positions"]

################
# Grab Popcorn #
################
regularised_timesteps = np.linspace(times[0], times[-1], len(times))
postprocessing.movie.make_movie(times, positions, SF,
                                regularised_timesteps=regularised_timesteps,
                                title="Stripe formation in intersection flows",
                                x_bound=(-20,70), y_bound=(-60, 20))
#postprocessing.plotting.plot(positions[-1], SF, title="Striping in Intersecting Flows", x_bound=(-20,70), y_bound=(-60,20), colors=colors, filetype='pdf')
