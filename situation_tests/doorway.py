from crowd_dynamics.social_force import SocialForce
import numpy as np
import postprocessing.movie
import postprocessing.plotting

###############################
# Initialise simulation class #
###############################
n_pedestrians=30
SF = SocialForce(n_pedestrians)

##########################
# Initialise pedestrians #
##########################
### Positions ###
p0 = np.empty((n_pedestrians,2))
p0[:n_pedestrians//2,0] = SF.rng.uniform(-10, 0, size=n_pedestrians//2)
p0[n_pedestrians//2:,0] = SF.rng.uniform(0,10,size=n_pedestrians//2)
p0[:,1] = SF.rng.uniform(-1.3,1.3,size=n_pedestrians)

### velocities ###
v0 = np.zeros_like(p0)
#v0[:n_pedestrians,0] = 1.34
#v0[n_pedestrians:,0] = -1.34

### Destinations ###
destinations = np.zeros((n_pedestrians,2,2))
destinations[:n_pedestrians//2,0,0] = 1
destinations[n_pedestrians//2:,0,0] = -1
destinations[:n_pedestrians//2,1,0] = 100000
destinations[n_pedestrians//2:,1,0] = -100000
destinations_range = np.ones((n_pedestrians,2,1))

SF.init_pedestrians(p0, destinations, velocities=v0, destinations_range=destinations_range)


##############
# Boundaries #
##############
boundaries = np.array([
    [[-150,1.5], [150,1.5]],
    [[-150, -1.5], [150,-1.5]],
    [[0,1.5],[0,0.5]],
    [[0,-1.5],[0,-0.5]]
])
SF.set_boundaries(boundaries, boundary_selection="superpose", boundary_verlet_sphere=1.)


##################
# Run simulation #
##################
SF.init_solver(t_bound=15)
times, results = SF.run(to_save=("positions",))
positions = results["positions"]

################
# Grab Popcorn #
################
regularised_timesteps = np.linspace(times[0], times[-1], len(times))
colors = np.array(['blue']*(n_pedestrians//2) + ['red']*(n_pedestrians - n_pedestrians//2))
postprocessing.movie.make_movie(times, positions, SF, regularised_timesteps=regularised_timesteps, colors=colors, title="2 Directional Pedestrian Flow in a Doorway", x_bound=(-10,10), y_bound=(-3, 3), interval=50)
#postprocessing.plotting.plot(positions[-1], SF, title="Striping in Intersecting Flows", x_bound=(-10,10), y_bound=(-3,3), colors=colors, filetype='pdf')

