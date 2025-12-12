from crowd_dynamics.social_force import SocialForce
import numpy as np

def run_sim(SF, t_bound, print_freq=100, to_save=("positions",)):
    SF.init_solver(t_bound=t_bound)

    results = {key: [] for key in to_save}
    times = []

    i = 0 
    while SF.t < 70 and SF.solver.status == 'running':
        if i%print_freq == 0:
            print(i)
        SF.step()

        if "positions" in to_save:
            results["positions"].append(SF.positions.copy())
        if "velocities" in to_save:
            results["velocities"].append(SF.velocities.copy())
        if "desired_speeds" in to_save:
            results["desired_speeds"].append(SF.desired_speeds.copy())
        if "desired_directions" in to_save:
            results["desired_directions"].append(SF.calc_desired_directions(SF.positions,
                                                                            SF.destinations))
        if "total_forces" in to_save:
            results["total_forces"].append(SF.total_force(SF.positions,
                                                          SF.destinations,
                                                          SF.radii,
                                                          SF.desired_speeds,
                                                          SF.velocities))
        if "repulsive_forces" in to_save:
            results["repulsive_forces"].append(SF.repulsive_force(SF.positions,
                                                                  SF.destinations,
                                                                  SF.radii,
                                                                  SF.desired_speeds,
                                                                  SF.velocities
                                                                  ))
        if "driving_forces" in to_save:
            desired_directions = SF.calc_desired_directions(SF.positions, SF.destinations)
            results["driving_forces"].append(SF.driving_force(SF.velocities,
                                                              desired_directions,
                                                              SF.desired_speeds
                                                              ))
        times.append(SF.t)
        i+=1

    return times, results



