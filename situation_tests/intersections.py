from crowd_dynamics.social_force import SocialForce
import numpy as np
import postprocessing.plotting
import postprocessing.movie
import postprocessing.simulation

###############################
# Initialise simulation class #
###############################
n_pedestrians=400
SF = SocialForce(n_pedestrians)
SF.A1 = 2.

##########################
# Initialise pedestrians #
##########################
### Positions ###
p0 = np.empty((n_pedestrians,2))

def rot(th):
    c,s = np.cos(th/2), np.sin(th/2)
    return np.array([[c,-s],[s,c]])

theta = -np.pi/4 # angle of 2 flows
offset = 30 # distance from origin
radius=4 # end semicircle length
length=100 # rectangle length

# Sample all positions in a rectangle with a semicircle at the end
i=0
while i < n_pedestrians:
    x, y = SF.rng.uniform(0, radius+length), SF.rng.uniform(-radius, radius)
    in_rectangle = (x>=radius)
    in_semi_circle = (x<radius) and (x**2 + y**2 <= radius**2)
    if in_rectangle or in_semi_circle:
        p0[i] = x,y
        i += 1

p0[:,0] *= -1 # Put the positions on the negative x axis
p0[:,0] -= offset # Offset the positions
# Rotate half the positions around the origin
p0[n_pedestrians//2:] = np.einsum('ij,kj->ik', p0[n_pedestrians//2:], rot(theta))

colors = np.array(['blue']*(n_pedestrians//2) + ['red']*(n_pedestrians - n_pedestrians//2))
#postprocessing.plotting.plot(p0, SF, colors=colors)

### velocities ###
v0 = np.zeros_like(p0)

### Destinations ###
destinations = np.zeros((n_pedestrians,1,2))
destinations[:,0,0] = 100000
destinations[n_pedestrians//2:,0] = np.einsum('ij,kj->ik',
                                            destinations[n_pedestrians//2:,0],
                                            rot(theta))

SF.init_pedestrians(p0, destinations, velocities=v0)


##################
# Run simulation #
##################
SF.init_solver(t_bound=80)
times, results = SF.run(to_save=("positions",))
positions = results["positions"]

################
# Grab Popcorn #
################
regularised_timesteps = np.linspace(times[0], times[-1], len(times))
postprocessing.movie.make_movie(times, positions, SF,
                                regularised_timesteps=regularised_timesteps,
                                colors=colors,
                                title="Stripe formation in intersection flows",
                                x_bound=(-20,70), y_bound=(-60, 20))
#postprocessing.plotting.plot(positions[-1], SF, title="Striping in Intersecting Flows", x_bound=(-20,70), y_bound=(-60,20), colors=colors, filetype='pdf')
