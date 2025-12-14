# crowd-dynamics
## Social Force model
All files for the social force model should be run from within the `social_force` directory.

All paths in this section are assumed to be prepended by `social_force/`
Ready to run simulations files are available for (among others)
 - bidirectional pedestrian motion in a hallway (`situation_tests/hallway.py`)
 - bidirectional flow through a hallway with a bottleneck (`situation_tests/doorway`)
 - intersection of two pedestrians at an angle (`situation_tests/striping.py`)
 - unidirectional flow of pedestrians around a corner (`situation_tests/corner.py`)

Videos of these simulations have been saved in the `videos` directory under self-explanatory names.
The images from the social force section of the report are saved in `images`

## Monte Carlo model

All files for the social force model should be run from within the `MC_crowd_sim` directory.

All paths in this section are assumed to be prepended by `MC_crowd_sim/`
All simulations can be run from `MC_crowd_sim/script.py`

## Comparison of the models
A simulation in the monte carlo model style run with the social force model can be found in
`social_force/analysis/central_exit_sf.py`, with its resulting video `social_force/videos/montecarlo.mp4`


# Instructions on running the Code
## Social Force Model
All the complex validation test cases (bidirectional flow in a hallway, striping in flows at an angle, bidirectional flow through a doorway, unidirectional flow around a $90^\circ$ corner) can be run by running the appropriate python files in the \texttt{social\_force/situation\_tests} directory of the repository (respectively `hallways.py`, `intersections.py`, `doorway.py`, `corner.py`). One can select whether to display an image or view a video of the simulation by commenting and uncommenting the appropriate lines at the bottom of the file (`postprocessing.plotting.plot`, `postprocessing.plotting.movie`). For the plot of velocities in the $90^\circ$ corner, one must comment the 2 `postprocessing` lines and remove the triple quotes around the final plotting code of the file.

In general, one can run a custom simulation by following the steps of the other files. For $n$ pedestrians, that is
1. Initialising the class with $n$.
2. Creating an $(n,2)$ `numpy` array of initial positions
3. Optionally creating an $(n,2)$ `numpy` array of initial velocities
4. Creating an $(n,m,2)$ `numpy` array of initial destinations. That is, an array of $m$ destinations for each pedestrians.
5. Optionally creating an $(n,m,1)$ array of destination ranges, which indicate when a pedestrian moves on to its next destination.
6. Running the class' `init_pedestrians` method with the previously defined parameters.
7. Optionally, create a $(b,2,2)$ array of boundaries, characterised by their start and end points, with $b$ the number of boundaries. Then run the `set_boundaries` method with this parameter and, optionally, the `boundary_selection` parameter set to "superpose" to use superposition of boundaries instead of nearest boundary selection for the boundary force calculation. If "superpose" is selected, one can also set `boundary_verlet_sphere` to some `float` value, so that only boundaries within the Verlet sphere are accounted for.
8. Run the `init_solver` method with parameters `t_bound` the maximal time the solver will attain (this can be set to something higher than the greatest time we want to run the simulation to). Optionally, one can also set `rtol`, `atol` and `max_step`, respectively the maximal relative tolerance, absolute tolerance and timestep that the RK45 solver will use.
9. Run the `run` method, with output `(time, results)`, with optional parameters `t_bound` the maximum time to which the solver will run, `print_freq`, every how many timesteps the number of timesteps passed is printed (this can be set to `False` or `None` for no printing) and `to_save`, a tuple of values to save. These can be accessed in the `results` dictionary, the second value of the output tuple. The following strings are taken into account if in `to_save`:

- "positions"
- "velocities"
- "desired_speeds"
- "desired_directions"
- "total_forces"
- "repulsive_forces"
- "driving_forces"
- "boundary_forces"


## Monte Carlo model

All the simulation results can be produced by running `MC_crowd_sim/script.py`. There are two options built in to run either: one or many independent simulations with different parameters, or running a set of simulations with different parameters but different starting configurations to acquire statistics from multiple runs. Choosing the running mode is done by uncommenting relevant lines in `if __name__ == "__main__":` condition. 

For running an independent simulation you must call the function `run_complete_simulation_once()` while specifying desired parameters, seed, and what results do you want to be shown or saved: initialisation, serving statistics or/and snapshots. After running the simulation results will be saved in `crowd_simulation_results/`.

For running a series of simulations you must call the function `run_complete_simulations()` and specifying number of consecutive runs as well as desired parameters and seed.
