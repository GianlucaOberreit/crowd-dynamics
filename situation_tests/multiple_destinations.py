from crowd_dynamics.social_force import SocialForce
import numpy as np
import scipy
import matplotlib.pyplot as plt


n_pedestrians=2
SF = SocialForce(n_pedestrians)
#SF.Ab = 20
p0 = np.empty((n_pedestrians,2))
p0[0,0] = -10
p0[1,0] = 10
p0[:,1] = [-2.,2.]
v0 = np.zeros_like(p0)
v0[0,0] = 1.34
v0[1,0] = -1.34
destinations = np.zeros((n_pedestrians,2,2))
destinations[:n_pedestrians//2,0] = [10,-2.]
destinations[n_pedestrians//2:,0] = [-10,2.]
destinations[:n_pedestrians//2,1] = [30,2.]
destinations[n_pedestrians//2:,1] = [-30,-2.]
destinations_range = np.ones((n_pedestrians,2,1))
n_pedestrians = p0.shape[0]
boundaries = np.array([[[-150,3], [150,3]], [[-150, -3], [150,-3]]])
SF.set_boundaries(boundaries)
SF.init_pedestrians(p0, destinations, velocities=v0, destinations_range=destinations_range)
SF.init_solver(t_bound=100)


all_positions = []
all_velocities = []
all_desired_speeds = []
all_desired_directions = []
all_total_forces = []
all_repulsive_forces = []
all_driving_forces = []
all_times = []

i = 0 
while SF.t < 30 and SF.solver.status == 'running':
    if i%500:
        print(i)
    SF.step()
    all_positions.append(SF.positions)
    all_velocities.append(SF.velocities)
    all_desired_speeds.append(SF.desired_speeds)
    destinations = SF.destinations[np.arange(n_pedestrians),SF.destinations_indices]
    desired_directions = destinations - SF.positions
    desired_directions_norm = np.linalg.norm(desired_directions, axis=1, keepdims=True)
    desired_directions = np.divide(desired_directions, desired_directions_norm,
                                   out=np.zeros_like(desired_directions), where=(desired_directions_norm!=0))
    all_desired_directions.append(desired_directions)
    all_total_forces.append(SF.total_force(SF.positions, destinations, SF.radii, SF.desired_speeds, SF.velocities))
    all_repulsive_forces.append(SF.repulsive_force(SF.positions, SF.radii, SF.velocities))
    all_driving_forces.append(SF.driving_force(SF.velocities, desired_directions, SF.desired_speeds))
    all_times.append(SF.t)
    i+=1

for _, boundary in enumerate(boundaries):
    plt.plot(boundary[:,0], boundary[:,1], color='k')
# Plot all positions up to this step for this pedestrian
for j in range(n_pedestrians):
    plt.plot(np.array(all_positions)[:,j, 0], np.array(all_positions)[:,j, 1], color='b' if j < n_pedestrians/2 else 'r')
    #plt.xlim(-50,50)
    #plt.ylim(-4,4)
plt.show()
