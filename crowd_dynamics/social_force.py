import numpy as np
import numpy.typing as npt
from scipy.integrate import RK45
#from numba import njit

def assert_shape(arr, expected):
    assert arr.shape == expected, f"shape {arr.shape}, expected {expected}"

class SocialForce:
    def __init__(self, n_pedestrians, rng_seed=42):
        self.n_pedestrians = n_pedestrians

        self.t = 0.

        self.destinations = np.empty((n_pedestrians,2), dtype=float)
        self.destinations_range = np.zeros((n_pedestrians), dtype=float)
        self.destinations_indices = np.zeros((n_pedestrians,), dtype=int)
        self.desired_speeds = np.empty((n_pedestrians,1))
        self.desired_speeds0 = np.ones_like(self.desired_speeds) * 1.34
        self.maximal_speeds = self.desired_speeds0*1.3

        self.anisotropic_character = 0.75
        self.relaxation_times = 0.5
        self.radii : npt.NDArray[np.float64] = np.ones((n_pedestrians, 1), dtype=float) * 0.3
        self.A1, self.A2, self.B1, self.B2 = 0, 2, 0.3, 0.2
        self.Ab, self.Bb = 5, 0.1
        self.Verlet_sphere = 10

        self.positions : npt.NDArray[np.float64] = np.zeros((n_pedestrians,2))
        self.positions0 : npt.NDArray[np.float64] = np.empty_like(self.positions)
        self.velocities : npt.NDArray[np.float64] = np.empty_like(self.positions)
        self.velocities0 : npt.NDArray[np.float64] = np.empty_like(self.positions)

        self.boundaries: npt.NDArray | None = None

        self.rng = np.random.default_rng(rng_seed)

    def init_solver(self, t_bound, solver="RK45", rtol=1e-3, atol=1e-4, max_step=None):
        y0 = np.concatenate((self.positions0.flatten(),
                            self.velocities0.flatten()))

        def step(t, y):
            positions = y[:2*self.n_pedestrians].reshape(self.positions.shape)
            velocities = y[2*self.n_pedestrians:4*self.n_pedestrians].reshape(self.velocities.shape)

            desired_speeds = self.calc_desired_speeds(t, positions)
            dpositions = velocities
            dvelocities = self.total_force(positions, 
                                           self.destinations[np.arange(self.n_pedestrians),
                                           self.destinations_indices],
                                           self.radii, desired_speeds, velocities)
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


    def init_pedestrians(self, positions: npt.ArrayLike, destinations, velocities=0., destinations_range=None):
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

    def set_boundaries(self, boundaries: npt.ArrayLike):
        self.boundaries = np.asarray(boundaries, float)

    def driving_force(self, velocities, desired_directions, desired_speeds):
        return 1/self.relaxation_times * (desired_speeds * desired_directions - velocities)

    def calc_radii_sum(self, radii):
        radii_sum = radii[:, None, :] + radii[None, :, :]
        if __debug__:
            assert radii_sum[0,0] == self.radii[0] + self.radii[0]
            if self.n_pedestrians > 1:
                assert radii_sum[1,0] == self.radii[1] + self.radii[0]
        return radii_sum

    def calc_relative_positions(self, positions):
        if __debug__:
            assert_shape(positions, (self.n_pedestrians, 2))
        relative_positions = positions[:, None, :] - positions[None, :, :]
        if __debug__:
            assert np.allclose(relative_positions, -relative_positions.swapaxes(0,1))
            if self.n_pedestrians > 1:
                assert np.allclose(relative_positions[0,1], positions[0] - positions[1])
        return relative_positions

    def calc_relative_distances(self, relative_positions):
        if __debug__:
            assert_shape(relative_positions, (self.n_pedestrians, self.n_pedestrians, 2))
            assert np.allclose(relative_positions, -relative_positions.swapaxes(0,1))
        distances = np.linalg.norm(relative_positions, axis=2, keepdims=True)
        if __debug__:
            assert_shape(distances, (self.n_pedestrians, self.n_pedestrians, 1))
        return distances
    
    def calc_normalized_relative_positions(self, relative_positions, distances):
        if __debug__:
            assert_shape(relative_positions, (self.n_pedestrians, self.n_pedestrians, 2))
            assert_shape(distances, (self.n_pedestrians, self.n_pedestrians, 1))
        normalized_relative_positions = np.divide(relative_positions, distances, out=np.zeros_like(relative_positions), where=(distances!=0))
        if __debug__:
            assert_shape(normalized_relative_positions, (self.n_pedestrians, self.n_pedestrians, 2))
        return normalized_relative_positions

    def calc_fovs(self, velocities, norm_vectors):
        if __debug__:
            assert_shape(velocities, (self.n_pedestrians, 2))
            assert_shape(norm_vectors, (self.n_pedestrians, self.n_pedestrians, 2))
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocity_norms = np.divide(velocities, speeds, out=np.zeros_like(velocities), where=(speeds!=0))
        cos_phi = -np.sum(norm_vectors * velocity_norms[:, None, :], axis=2, keepdims = True) # TODO: check this
        if __debug__:
            assert_shape(cos_phi, (self.n_pedestrians, self.n_pedestrians, 1))
        return self.anisotropic_character + (1.-self.anisotropic_character)*(1.+cos_phi) / 2.

    def repulsive_force(self, positions, radii, velocities):
        if __debug__:
            assert_shape(positions, (self.n_pedestrians, 2))
            assert_shape(radii, (self.n_pedestrians, 1))
            assert_shape(velocities, (self.n_pedestrians, 2))

        radii_sum = self.calc_radii_sum(radii)
        relative_positions = self.calc_relative_positions(positions)
        distances = self.calc_relative_distances(relative_positions)
        norm_vectors = self.calc_normalized_relative_positions(relative_positions, distances)

        fov = self.calc_fovs(velocities, norm_vectors)

        if __debug__:
            assert_shape(radii_sum, (self.n_pedestrians, self.n_pedestrians, 1))
            assert_shape(relative_positions, (self.n_pedestrians, self.n_pedestrians, 2))
            assert_shape(distances, (self.n_pedestrians, self.n_pedestrians, 1))
            assert_shape(norm_vectors, (self.n_pedestrians, self.n_pedestrians, 2))
            N = norm_vectors.shape[0]
            diagonal = norm_vectors[np.arange(N), np.arange(N)]
            assert np.all(diagonal == 0)
            assert_shape(fov, (self.n_pedestrians, self.n_pedestrians, 1))
            assert fov.shape == (self.n_pedestrians, self.n_pedestrians, 1)

        repulsive_force_mat = self.A1 * np.exp( (radii_sum - distances)/self.B1) * norm_vectors * fov \
                            + self.A2 * np.exp( (radii_sum - distances)/self.B2 ) * norm_vectors
        repulsive_force_mat = np.where(distances <= 10, repulsive_force_mat, np.zeros_like(repulsive_force_mat))
        if __debug__:
            assert_shape(repulsive_force_mat, (self.n_pedestrians, self.n_pedestrians, 2))
            N = repulsive_force_mat.shape[0]
            diagonal = repulsive_force_mat[np.arange(N), np.arange(N)]
            assert np.all(diagonal == 0)


        return np.sum(repulsive_force_mat, axis=1)
    
    def boundary_force(self, positions: npt.NDArray):
        if __debug__:
            assert_shape(positions, (self.n_pedestrians, 2))
        if self.boundaries is None:
            return np.zeros_like(positions)
        p = positions[:, None, :] # (N,1,2)
        b0 = self.boundaries[:,0,:][None,:,:] # (1,M,2) boundary starts
        b1 = self.boundaries[:,1,:][None,:,:] # (1,M,2) boundary ends

        if __debug__:
            assert_shape(p, (self.n_pedestrians, 1, 2))
            M = self.boundaries.shape[0]
            assert_shape(b0, (1, M, 2))
            assert_shape(b1, (1, M, 2))

        v = b1-b0 # (1,M,2) boundary directions vector
        w = p-b0 # (N,M,2) position to boundary vector

        if __debug__:
            M = self.boundaries.shape[0]
            assert_shape(v, (1, M, 2))
            assert_shape(w, (self.n_pedestrians, M, 2))

        vv = np.sum(v*v, axis=2, keepdims=True) # (1,M,1) Norm of v vectors
        vw = np.sum(w*v, axis=2, keepdims=True) # (N,M,1) dot product of v and w

        if __debug__:
            M = self.boundaries.shape[0]
            assert_shape(vv, (1, M, 1))
            assert_shape(vw, (self.n_pedestrians, M, 1))

        t = vw/vv # (N,M,1) norm vector towards closest point on each boundary
        t = np.clip(t, 0.0, 1.0) # projection along segment
        closest = b0 + t * v # (N,M,2)
        if __debug__:
            M = self.boundaries.shape[0]
            assert_shape(t, (self.n_pedestrians, M, 1))
            assert_shape(closest, (self.n_pedestrians, M, 2))

        diff = p-closest # (N,M,2)

        if __debug__:
            M = self.boundaries.shape[0]
            assert_shape(diff, (self.n_pedestrians, M, 2))

        dist = np.linalg.norm(diff, axis=2, keepdims=True) # (N,M,1)

        if __debug__:
            M = self.boundaries.shape[0]
            assert_shape(dist, (self.n_pedestrians, M, 1))

        idx = np.argmin(dist, axis=1)[:,0] # (N,)
        if __debug__:
            assert_shape(idx, (self.n_pedestrians,))
        nearest_dist = dist[np.arange(self.n_pedestrians), idx]
        nearest_dir = diff[np.arange(self.n_pedestrians), idx] / nearest_dist
        if __debug__:
            assert_shape(nearest_dist, (self.n_pedestrians, 1))
            assert_shape(nearest_dir, (self.n_pedestrians,2))

        force_mag = self.Ab * np.exp(self.radii - nearest_dist / self.Bb)
        force = force_mag * nearest_dir

        if __debug__:
            assert_shape(force, (self.n_pedestrians,2))

        return force
    
    def individuality_force(self, desired_directions):
        if __debug__:
            if not desired_directions.shape == (self.n_pedestrians, 2):
                print(desired_directions.shape)
            assert_shape(desired_directions, (self.n_pedestrians,2))
        return np.zeros_like(desired_directions)

    def total_force(self, positions, destinations, radii, desired_speeds, velocities):
        desired_directions = destinations - positions
        desired_directions_norm = np.linalg.norm(desired_directions, axis=1, keepdims=True)
        desired_directions = np.divide(desired_directions, desired_directions_norm,
                                       out=np.zeros_like(desired_directions), where=(desired_directions_norm!=0))
        total_force = self.driving_force(velocities, desired_directions, desired_speeds) \
                    + self.repulsive_force(positions, radii, velocities) \
                    + self.boundary_force(positions) \
                    + self.individuality_force(desired_directions)
        if __debug__:
            assert_shape(total_force, (self.n_pedestrians,2))
        return total_force
    
    def calc_desired_speeds(self, t, positions):
        if t == 0.:
            return self.desired_speeds0
        pos0_to_pos = positions - self.positions0
        avg_speed = np.linalg.norm(pos0_to_pos / t, axis=1, keepdims=True)
        impatience = 1 - avg_speed / self.desired_speeds0
        if __debug__:
            assert_shape(pos0_to_pos, (self.n_pedestrians,2))
            assert_shape(avg_speed, (self.n_pedestrians,1))
            assert_shape(impatience, (self.n_pedestrians,1))
        new_desired_speeds = (1-impatience) * self.desired_speeds0 + impatience * self.maximal_speeds
        if __debug__:
            assert_shape(new_desired_speeds, (self.n_pedestrians,1))
        return new_desired_speeds
    
    def update_destinations(self):
        '''
        pedestrians_in_range = np.where(
            np.linalg.norm(self.destinations[
                np.arange(self.n_pedestrians),
                self.destinations_indices
            ] - self.positions, axis=1)
            <= self.destinations_range[
                np.arange(self.n_pedestrians),
                self.destinations_indices].flatten()
        )
        pedestrians_not_at_last_dest = np.where(self.destinations_indices < self.destinations.shape[1]-1)
        pedestrians_to_update = np.intersect1d(pedestrians_in_range, pedestrians_not_at_last_dest)
        self.destinations_indices[pedestrians_to_update] += 1 
        '''
        pedestrians_in_range = np.linalg.norm(
            self.destinations[np.arange(self.n_pedestrians), self.destinations_indices] - self.positions,
            axis=1
        ) <= self.destinations_range[np.arange(self.n_pedestrians), self.destinations_indices].flatten()
        pedestrians_not_at_last_dest = self.destinations_indices < self.destinations.shape[1] - 1
        mask = np.where(pedestrians_in_range & pedestrians_not_at_last_dest)[0]
        self.destinations_indices[mask] += 1




