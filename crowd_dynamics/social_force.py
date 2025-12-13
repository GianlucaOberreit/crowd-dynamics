import numpy as np
import numpy.typing as npt
from scipy.integrate import RK45
from numba import njit, prange

def assert_shape(arr, expected):
    assert arr.shape == expected, f"shape {arr.shape}, expected {expected}"

class SocialForce:
    def __init__(self, n_pedestrians, rng_seed=42):
        self.n_pedestrians = n_pedestrians

        self.t = 0.

        self.destinations = np.empty((n_pedestrians,1,2), dtype=float)
        self.destinations_range = np.zeros((n_pedestrians,1), dtype=float)
        self.destinations_indices = np.zeros((n_pedestrians,), dtype=int)
        self.desired_speeds = np.empty((n_pedestrians,1))
        self.desired_speeds0 = np.ones_like(self.desired_speeds) * 1.34
        self.maximal_speeds = self.desired_speeds0.copy()*1.3

        self.anisotropic_character = 0.75
        self.relaxation_times = 0.5
        self.radii : npt.NDArray[np.float64] = np.ones((n_pedestrians, 1), dtype=float) * 0.3
        self.A1, self.A2, self.B1, self.B2 = 0., 2., 0.3, 0.2
        self.Ab, self.Bb = 5, 0.1
        self.Verlet_sphere = 10

        self.positions : npt.NDArray[np.float64] = np.zeros((n_pedestrians,2), dtype=float)
        self.positions0 : npt.NDArray[np.float64] = np.empty_like(self.positions)
        self.velocities : npt.NDArray[np.float64] = np.empty_like(self.positions)
        self.velocities0 : npt.NDArray[np.float64] = np.empty_like(self.positions)

        self.boundaries: npt.NDArray | None = None
        self.boundary_verlet_sphere = 1.
        self.boundary_selection = "nearest"

        self.rng = np.random.default_rng(rng_seed)

        self.results={}
        self.forces = {"total": None, "repulsive": None, "driving": None, "boundary": None}

    def init_solver(self, t_bound, solver="RK45", rtol=1e-3, atol=1e-4, max_step=None):
        y0 = np.concatenate((self.positions0.flatten(),
                            self.velocities0.flatten()))

        def step(t, y):
            positions = y[:2*self.n_pedestrians].reshape(self.positions.shape)
            velocities = y[2*self.n_pedestrians:4*self.n_pedestrians].reshape(self.velocities.shape)

            desired_speeds = self.calc_desired_speeds(t, positions)
            dpositions = velocities
            dvelocities = self.total_force(positions, desired_speeds, velocities)
            return np.concatenate((dpositions.flatten(),
                                  dvelocities.flatten()))
        if solver == "RK45":
            if max_step:
                self.solver = RK45(step, 0., y0, t_bound, rtol=rtol, atol=atol, max_step=max_step)
            else:
                self.solver = RK45(step, 0., y0, t_bound, rtol=rtol, atol=atol)

    def step(self):
        self.solver.step()
        y = self.solver.y
        self.t = self.solver.t
        self.positions = y[:2*self.n_pedestrians].reshape(self.positions.shape)
        self.velocities = y[2*self.n_pedestrians:4*self.n_pedestrians].reshape(self.velocities.shape)
        self.desired_speeds = self.calc_desired_speeds(self.t, self.positions)
        self.update_destinations()

    def run(self, t_bound=None, print_freq=100, to_save=("positions",)):
        if t_bound is None:
            t_bound = self.solver.t_bound

        self.results = {key: [] for key in to_save}
        times = []

        i = 0 
        while self.t < t_bound and self.solver.status == 'running':
            if print_freq is not None and print_freq is not False and i%print_freq == 0:
                print(i)
            self.step()
            if "positions" in to_save:
                self.results["positions"].append(self.positions.copy())
            if "velocities" in to_save:
                self.results["velocities"].append(self.velocities.copy())
            if "desired_speeds" in to_save:
                self.results["desired_speeds"].append(self.desired_speeds.copy())
            if "desired_directions" in to_save:
                self.results["desired_directions"].append(self.calc_desired_directions())
            if "total_forces" in to_save:
                self.results["total_forces"].append(self.forces["total"])
            if "repulsive_forces" in to_save:
                self.results["repulsive_forces"].append(self.forces["repulsive"])
            if "driving_forces" in to_save:
                self.results["driving_forces"].append(self.forces["driving"])
            if "boundary_forces" in to_save:
                self.results["boundary_forces"].append(self.forces["boundary"])

            times.append(self.t)
            i+=1

        return times, self.results

    def init_pedestrians(self, positions: npt.ArrayLike, destinations, velocities: npt.ArrayLike | float =0., destinations_range=None):
        """Set positions and velocities of pedestrians
        
        Parameters
        ----------
        positions : ArrayLike
            Positions of the pedestrians
        velocities : ArrayLike or float
            Initial velocities of the pedestrians
        """
        positions = np.asarray(positions, float)
        destinations = np.asarray(destinations, float)
        if type(velocities) == float:
            self.velocities0 = np.ones_like(self.velocities0) * velocities
            self.velocities = np.ones_like(self.velocities) * velocities
        else:
            self.velocities0 = velocities.copy()
            self.velocities = velocities.copy()
        self.positions0 = positions.copy()
        self.positions = positions.copy()
        self.destinations = destinations
        if destinations_range is not None:
            self.destinations_range = destinations_range
        else:
            self.destinations_range = np.zeros(destinations.shape[:2] + (1,))

    def set_boundaries(self, boundaries: npt.ArrayLike, boundary_selection="nearest",
                       boundary_verlet_sphere=1.):
        self.boundaries = np.asarray(boundaries, float)
        self.boundary_selection = boundary_selection
        self.boundary_verlet_sphere = boundary_verlet_sphere

    def driving_force(self, velocities, desired_directions, desired_speeds):
        force = 1/self.relaxation_times * (desired_speeds * desired_directions - velocities)
        if "driving_forces" in self.results.keys():
            self.forces["driving"] = force
        return force

    def calc_fovs(self, velocities, norm_vectors):
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocity_norms = np.divide(velocities, speeds, out=np.zeros_like(velocities), where=(speeds!=0))
        cos_phi = -np.sum(norm_vectors * velocity_norms[:, None, :], axis=2, keepdims = True)
        return self.anisotropic_character + (1.-self.anisotropic_character)*(1.+cos_phi) / 2.

    def repulsive_force_numpy(self, positions, radii, velocities):
        radii_sum = radii[:, None, :] + radii[None, :, :]
        relative_positions = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(relative_positions, axis=2, keepdims=True)
        unit_vectors = np.divide(relative_positions, distances,
                                 out=np.zeros_like(relative_positions), where=(distances!=0))
        fov = self.calc_fovs(velocities, unit_vectors)

        repulsive_force_mat = self.A1 * np.exp( (radii_sum - distances)/self.B1) * unit_vectors * fov \
                            + self.A2 * np.exp( (radii_sum - distances)/self.B2 ) * unit_vectors
        repulsive_force_mat = np.where(distances <= 10, repulsive_force_mat, np.zeros_like(repulsive_force_mat))

        return np.sum(repulsive_force_mat, axis=1)

    @staticmethod
    @njit(parallel=True)
    def repulsive_force_njit(positions, radii, velocities, anisotropic_character, Verlet_sphere, A1, B1, A2, B2):
        n_pedestrians = positions.shape[0]
        repulsive_force_mat = np.zeros((n_pedestrians, n_pedestrians,2))
        unit_vectors = np.zeros_like(repulsive_force_mat)
        unit_velocities = np.zeros_like(positions)
        for i in prange(n_pedestrians):
            speeds = np.linalg.norm(velocities[i])
            if speeds != 0.:
                unit_velocities[i] = velocities[i] / speeds
            for j in prange(n_pedestrians):
                if i==j: continue
                relative_positions = positions[i] - positions[j]
                distance = np.linalg.norm(relative_positions)
                if distance > Verlet_sphere:
                    continue

                radii_sum = radii[i,0] + radii[j,0]
                if distance != 0:
                    unit_vectors[i,j] = relative_positions / distance
                cos_phi = -np.sum(unit_vectors[i,j] * unit_velocities[i])
                fov = anisotropic_character + (1.-anisotropic_character) * (1+cos_phi)/2.

                repulsive_force_mat[i,j] = A1 * np.exp( (radii_sum - distance)/B1) * unit_vectors[i,j] * fov \
                                   + A2 * np.exp( (radii_sum - distance)/B2 ) * unit_vectors[i,j]

        return np.sum(repulsive_force_mat, axis=1)

    def repulsive_force(self, positions, radii, velocities):
        if self.n_pedestrians > 250:
            force = self.repulsive_force_njit(positions, radii, velocities,
                                             self.anisotropic_character, self.Verlet_sphere,
                                             self.A1, self.B1, self.A2, self.B2)
        else:
            force = self.repulsive_force_numpy(positions, radii, velocities)
        if "repulsive_forces" in self.results.keys():
            self.forces["repulsive"] = force
        return force
 
    
    def boundary_force(self, positions: npt.NDArray):
        if self.boundaries is None:
            return np.zeros_like(positions)
        p = positions[:, None, :] # (N,1,2)
        b0 = self.boundaries[:,0,:][None,:,:] # (1,M,2) boundary starts
        b1 = self.boundaries[:,1,:][None,:,:] # (1,M,2) boundary ends

        v = b1-b0 # (1,M,2) boundary directions vector
        w = p-b0 # (N,M,2) position to boundary vector

        vv = np.sum(v*v, axis=2, keepdims=True) # (1,M,1) Norm of v vectors
        vw = np.sum(w*v, axis=2, keepdims=True) # (N,M,1) dot product of v and w

        t = vw/vv # (N,M,1) norm vector towards closest point on each boundary
        t = np.clip(t, 0.0, 1.0) # projection along segment
        closest = b0 + t * v # (N,M,2)

        diff = p-closest # (N,M,2)

        dist = np.linalg.norm(diff, axis=2, keepdims=True) # (N,M,1)

        if self.boundary_selection == "nearest":
            idx = np.argmin(dist, axis=1)[:,0] # (N,)
            nearest_dist = dist[np.arange(self.n_pedestrians), idx]
            nearest_dir = diff[np.arange(self.n_pedestrians), idx] / nearest_dist

            force_mag = self.Ab * np.exp((self.radii - nearest_dist) / self.Bb)
            force = force_mag * nearest_dir

        elif self.boundary_selection == "superpose":
            mask = (dist < self.boundary_verlet_sphere)
            directions = np.divide(diff, dist,
                                   out=np.zeros_like(diff),
                                   where=(dist!=0))
            force_mag = self.Ab * np.exp((self.radii[:,None,:] - dist) / self.Bb)
            force_mag *= mask.astype(force_mag.dtype)
            force = np.sum(force_mag * directions, axis=1)
        else:
            raise

        if "boundary_forces" in self.results.keys():
            self.forces["boundary"] = force
        return force
    
    def total_force(self, positions, desired_speeds, velocities):
        desired_directions = self.calc_desired_directions()
        total_force = self.driving_force(velocities, desired_directions, desired_speeds) \
                    + self.repulsive_force(positions, self.radii, velocities) \
                    + self.boundary_force(positions)
        if "total_forces" in self.results.keys():
            self.forces["total"] = total_force
        return total_force
    
    def calc_desired_speeds(self, t, positions):
        if t == 0.:
            return self.desired_speeds0
        pos0_to_pos = positions - self.positions0
        avg_speed = np.linalg.norm(pos0_to_pos / t, axis=1, keepdims=True)
        impatience = 1 - avg_speed / self.desired_speeds0
        new_desired_speeds = (1-impatience) * self.desired_speeds0 + impatience * self.maximal_speeds
        return new_desired_speeds
    
    def calc_desired_directions(self):
        desired_directions = self.current_destinations() - self.positions
        desired_directions_norm = np.linalg.norm(desired_directions, axis=1, keepdims=True)
        desired_directions = np.divide(desired_directions, desired_directions_norm,
                                       out=np.zeros_like(desired_directions),
                                       where=(desired_directions_norm!=0))
        return desired_directions

    def current_destinations(self):
        return self.destinations[np.arange(self.n_pedestrians), self.destinations_indices]

    def current_destinations_range(self):
        return self.destinations_range[np.arange(self.n_pedestrians), self.destinations_indices]

    def update_destinations(self):
        pedestrians_in_range = np.linalg.norm(self.current_destinations() - self.positions, axis=1) \
                            <= self.current_destinations_range().flatten()
        pedestrians_not_at_last_dest = self.destinations_indices < self.destinations.shape[1] - 1
        mask = np.where(pedestrians_in_range & pedestrians_not_at_last_dest)[0]
        self.destinations_indices[mask] += 1






