from crowd_dynamics.social_force import SocialForce
import numpy as np
import postprocessing.movie
import postprocessing.simulation
import postprocessing.plotting

###############################
# Initialise simulation class #
###############################
n_pedestrians=250
SF = SocialForce(n_pedestrians)

##########################
# Initialise pedestrians #
##########################
### Positions ###
p0 = np.empty((n_pedestrians,2))
p0[:n_pedestrians//2,0] = SF.rng.uniform(-100, -10, size=n_pedestrians//2)
p0[n_pedestrians//2:,0] = SF.rng.uniform(10,100,size=n_pedestrians//2)
p0[:,1] = SF.rng.uniform(-2.7,2.7,size=n_pedestrians)

### velocities ###
v0 = np.zeros_like(p0)
v0[:n_pedestrians,0] = 1.34
v0[n_pedestrians:,0] = -1.34

### Destinations ###
destinations = np.zeros((n_pedestrians,1,2))
destinations[:n_pedestrians//2,0,0] = 100000
destinations[n_pedestrians//2:,0,0] = -100000

SF.init_pedestrians(p0, destinations, velocities=v0)


##############
# Boundaries #
##############
boundaries = np.array([
    [[-150,3], [150,3]],
    [[-150, -3], [150,-3]],
])
SF.set_boundaries(boundaries)


##################
# Run simulation #
##################
SF.init_solver(t_bound=50)
times, results = postprocessing.simulation.run_sim(SF, to_save=("positions",))
positions = results["positions"]

################
# Grab Popcorn #
################
regularised_timesteps = np.linspace(times[0], times[-1], len(times))
colors = np.array(['blue']*(n_pedestrians//2) + ['red']*(n_pedestrians - n_pedestrians//2))
postprocessing.movie.make_movie(times, positions, SF, regularised_timesteps=regularised_timesteps, colors=colors, title="Lane Formation in a Hallway", x_bound=(-30,30), y_bound=(-3.5, 3.5), interval=50)
#postprocessing.plotting.plot(positions[-1], SF, title="Lane formation in a Hallway", x_bound=(-30,30), y_bound=(-3.5,3.5), colors=colors, filetype='pdf')

