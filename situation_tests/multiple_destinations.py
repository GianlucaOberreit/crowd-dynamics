from crowd_dynamics.social_force import SocialForce
import numpy as np
import postprocessing.plotting
import postprocessing.simulation

###############################
# Initialise simulation class #
###############################
n_pedestrians=2
SF = SocialForce(n_pedestrians)

##########################
# Initialise pedestrians #
##########################
### Positions ###
p0 = np.empty((n_pedestrians,2))
p0[0,0] = -10
p0[1,0] = 10
p0[:,1] = [-2.,2.]

### velocities ###
v0 = np.zeros_like(p0)
v0[0,0] = 1.34
v0[1,0] = -1.34

### Destinations ###
destinations = np.zeros((n_pedestrians,2,2))
destinations[:n_pedestrians//2,0] = [10,-2.]
destinations[n_pedestrians//2:,0] = [-10,2.]
destinations[:n_pedestrians//2,1] = [30,2.]
destinations[n_pedestrians//2:,1] = [-30,-2.]
destinations_range = np.ones((n_pedestrians,2,1))

SF.init_pedestrians(p0, destinations, velocities=v0, destinations_range=destinations_range)


##############
# Boundaries #
##############
boundaries = np.array([
    [[-50,3], [50,3]],
    [[-50, -3], [50,-3]],
])
SF.set_boundaries(boundaries)


##################
# Run simulation #
##################
SF.init_solver(t_bound=30)
times, results = SF.run(to_save=("positions",))
positions = results["positions"]

################
# Grab Popcorn #
################
regularised_timesteps = np.linspace(times[0], times[-1], len(times))
colors = np.array(['blue']*(n_pedestrians//2) + ['red']*(n_pedestrians - n_pedestrians//2))
postprocessing.plotting.plot(positions, SF, continuous=True, title="Multiple destinations", colors=colors)

