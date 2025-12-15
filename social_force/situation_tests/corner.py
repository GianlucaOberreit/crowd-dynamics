from crowd_dynamics.social_force import SocialForce
import numpy as np
import postprocessing.movie
import postprocessing.plotting
import matplotlib.pyplot as plt

###############################
# Initialise simulation class #
###############################
n_pedestrians=200
SF = SocialForce(n_pedestrians)

##########################
# Initialise pedestrians #
##########################
### Positions ###
p0 = np.empty((n_pedestrians,2))
p0[:,0] = SF.rng.uniform(-30, 5, size=n_pedestrians)
p0 = p0[np.argsort(p0[:,0])]
p0[:,1] = SF.rng.uniform(0.3,2.7,size=n_pedestrians)

### velocities ###
v0 = np.zeros_like(p0)

### Destinations ###
destinations = np.zeros((n_pedestrians,2,2))
destinations_range = np.zeros((n_pedestrians, 2, 1))

destinations[:,0,0] = 8.5
destinations[:,0,1] = 2.
destinations_range[:,0] = 1.5

destinations[:,1,0] = 8.5
destinations[:,1,1] = 1e5

SF.init_pedestrians(p0, destinations, velocities=v0, destinations_range=destinations_range)


##############
# Boundaries #
##############
boundaries = np.array([
    [[-50,3], [7,3]],
    [[-50, 0], [10,0]],
    [[7,3], [7,50]],
    [[10,0], [10,50]],
])
SF.set_boundaries(boundaries)


##################
# Run simulation #
##################
SF.init_solver(t_bound=15)
times, results = SF.run(to_save=("positions", "velocities"))
positions = results["positions"]

################
# Grab Popcorn #
################
regularised_timesteps = np.linspace(times[0], times[-1], len(times))
postprocessing.movie.make_movie(times, positions, SF, interval=50, regularised_timesteps=regularised_timesteps, colors=['b']*n_pedestrians, title="Lane Formation in a Hallway", x_bound=(0,10), y_bound=(0, 10))
#postprocessing.plotting.plot(positions[-1], SF, title="", x_bound=(0,10), y_bound=(0,10), filetype='pdf')

# Remove the triple quotes to recreate the plot from the report. Commenting out the postprocessing.movie.make_move line can also be useful
'''
plt.plot(np.arange(n_pedestrians), np.linalg.norm(results["velocities"][-1], axis=1))
plt.xlabel("Pedestrian index")
plt.ylabel("speed (m/s)")

plt.title("Pedestrian speeds at a corner")

plt.show()
'''
