import numpy as np
import numpy.typing as npt
from scipy.integrate import RK45
from numba import njit, prange


def assert_shape(arr, expected):
    assert arr.shape == expected, f"shape {arr.shape}, expected {expected}"


class SocialForce:
    """
    Social-force pedestrian dynamics model with RK45 time integration.

    Parameters
    ----------
    n_pedestrians : int
        Number of pedestrians (agents) simulated.
    rng_seed : int, default=42
        Seed used to initialise `numpy.random.Generator` stored in `self.rng`.

    Attributes
    ----------
    n_pedestrians : int
        Number of pedestrians.
    t : float
        Current simulation time in seconds.
    positions : (N, 2) ndarray
        Current positions in metres.
    velocities : (N, 2) ndarray
        Current velocities in m/s.
    radii : (N, 1) ndarray
        Pedestrian radii in metres.
    destinations : (N, K, 2) ndarray
        Destination waypoints in metres.
    destinations_indices : (N,) ndarray of int
        Current waypoint index per pedestrian.
    destinations_range : (N, K, 1) ndarray
        Arrival radius per waypoint (metres). When the active destination is
        within this range, the waypoint index may be advanced.
    desired_speeds0 : (N, 1) ndarray
        Baseline desired speeds (m/s) used at t=0.
    maximal_speeds : (N, 1) ndarray
        Upper bound used by `calc_desired_speeds` (m/s).
    boundaries : (M, 2, 2) ndarray or None
        Line-segment boundaries (walls). Each segment is [[x0,y0],[x1,y1]] in m.
    solver : scipy.integrate.RK45
        ODE solver initialised by `init_solver`.

    Notes
    -----
    - The model treats forces as accelerations (unit mass).
    - `repulsive_force` uses a cutoff distance `Verlet_sphere` (metres) in the
      Numba implementation and a hard-coded cutoff of 10 m in the NumPy path.
    """

    def __init__(self, n_pedestrians: int, rng_seed: int = 42):
        """
        Initialise model parameters and allocate state arrays.

        Parameters
        ----------
        n_pedestrians : int
            Number of pedestrians (agents) simulated.
        rng_seed : int, default=42
            Seed for the internal random number generator `self.rng`.

        Notes
        -----
        This constructor allocates arrays but does not set initial conditions.
        Call `init_pedestrians` before running the simulation.
        """
        self.n_pedestrians: int = n_pedestrians

        self.t: float = 0.0

        self.destinations: npt.NDArray = np.empty((n_pedestrians, 1, 2), dtype=float)
        self.destinations_range: npt.NDArray = np.zeros((n_pedestrians, 1), dtype=float)
        self.destinations_indices: npt.NDArray = np.zeros((n_pedestrians,), dtype=int)
        self.desired_speeds: npt.NDArray = np.empty((n_pedestrians, 1))
        self.desired_speeds0: npt.NDArray = np.ones_like(self.desired_speeds) * 1.34
        self.maximal_speeds: npt.NDArray = self.desired_speeds0.copy() * 1.3

        self.anisotropic_character: float = 0.75
        self.relaxation_times: float = 0.5
        self.radii: npt.NDArray[np.float64] = (
            np.ones((n_pedestrians, 1), dtype=float) * 0.3
        )
        self.A1: float = 0.0
        self.A2: float = 2.0
        self.B1: float = 0.3
        self.B2: float = 0.2
        self.Verlet_sphere: float = 10.0

        self.positions: npt.NDArray[np.float64] = np.zeros(
            (n_pedestrians, 2), dtype=float
        )
        self.positions0: npt.NDArray[np.float64] = np.empty_like(self.positions)
        self.velocities: npt.NDArray[np.float64] = np.empty_like(self.positions)
        self.velocities0: npt.NDArray[np.float64] = np.empty_like(self.positions)

        self.boundaries: npt.NDArray | None = None
        self.Ab: float = 5.0
        self.Bb: float = 0.1
        self.boundary_verlet_sphere: float = 1.0
        self.boundary_selection: str = "nearest"

        self.rng = np.random.default_rng(rng_seed)

        self.results: dict = {}
        self.forces: dict = {
            "total": None,
            "repulsive": None,
            "driving": None,
            "boundary": None,
        }

    def init_solver(
        self,
        t_bound: float,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        max_step: float | None = None,
    ):
        """
        Initialise the ODE integrator for the current initial state.

        Parameters
        ----------
        t_bound : float
            Final integration time (seconds) passed to the SciPy solver.
        rtol : float, default=1e-3
            Relative tolerance for the ODE solver.
        atol : float, default=1e-4
            Absolute tolerance for the ODE solver.
        max_step : float or None, default=None
            Maximum allowed step size (seconds). If None, SciPy chooses adaptively.

        Returns
        -------
        None

        Notes
        -----
        The integrated state vector is `[positions0.flatten(), velocities0.flatten()]`.
        The derivative is computed as:
        - dpositions/dt = velocities
        - dvelocities/dt = total_force(...)
        """
        y0 = np.concatenate((self.positions0.flatten(), self.velocities0.flatten()))

        def step(t, y):
            positions = y[: 2 * self.n_pedestrians].reshape(self.positions.shape)
            velocities = y[2 * self.n_pedestrians : 4 * self.n_pedestrians].reshape(
                self.velocities.shape
            )

            desired_speeds = self.calc_desired_speeds(t, positions)
            dpositions = velocities
            dvelocities = self.total_force(positions, desired_speeds, velocities)
            return np.concatenate((dpositions.flatten(), dvelocities.flatten()))

        if max_step:
            self.solver = RK45(
                step, 0.0, y0, t_bound, rtol=rtol, atol=atol, max_step=max_step
            )
        else:
            self.solver = RK45(step, 0.0, y0, t_bound, rtol=rtol, atol=atol)

    def step(self):
        """
        Advance the simulation by one adaptive RK45 step.

        Returns
        -------
        None

        Notes
        -----
        Side effects (in-place updates):
        - Updates `self.t`, `self.positions`, `self.velocities`.
        - Recomputes `self.desired_speeds`.
        - Advances `self.destinations_indices` via `update_destinations`.

        The step size is chosen internally by SciPy's RK45 controller unless
        `max_step` was provided in `init_solver`.
        """
        self.solver.step()
        y = self.solver.y
        self.t = self.solver.t
        self.positions = y[: 2 * self.n_pedestrians].reshape(self.positions.shape)
        self.velocities = y[2 * self.n_pedestrians : 4 * self.n_pedestrians].reshape(
            self.velocities.shape
        )
        self.desired_speeds = self.calc_desired_speeds(self.t, self.positions)
        self.update_destinations()

    def run(self, t_bound=None, print_freq=100, to_save=("positions",)):
        if t_bound is None:
            t_bound = self.solver.t_bound

        self.results = {key: [] for key in to_save}
        times = []

        i = 0
        while self.t < t_bound and self.solver.status == "running":
            if (
                print_freq is not None
                and print_freq is not False
                and i % print_freq == 0
            ):
                print(i)
            self.step()
            if "positions" in to_save:
                self.results["positions"].append(self.positions.copy())
            if "velocities" in to_save:
                self.results["velocities"].append(self.velocities.copy())
            if "desired_speeds" in to_save:
                self.results["desired_speeds"].append(self.desired_speeds.copy())
            if "desired_directions" in to_save:
                self.results["desired_directions"].append(
                    self.calc_desired_directions()
                )
            if "total_forces" in to_save:
                self.results["total_forces"].append(self.forces["total"])
            if "repulsive_forces" in to_save:
                self.results["repulsive_forces"].append(self.forces["repulsive"])
            if "driving_forces" in to_save:
                self.results["driving_forces"].append(self.forces["driving"])
            if "boundary_forces" in to_save:
                self.results["boundary_forces"].append(self.forces["boundary"])

            times.append(self.t)
            i += 1

        return times, self.results

    def init_pedestrians(
        self,
        positions: npt.NDArray,
        destinations,
        velocities: npt.ArrayLike | float = 0.0,
        destinations_range=None,
    ):
        """
        Set initial positions, velocities, and destinations.

        Parameters
        ----------
        positions : (N, 2) array_like
            Initial positions in metres.
        destinations : (N, K, 2) array_like
            Waypoint destinations in metres.
        velocities : (N, 2) array_like or float, default=0.0
            Initial velocities in m/s. If a float is provided, it is broadcast
            to all agents and both components.
        destinations_range : (N, K, 1) array_like or None, default=None
            Arrival radius per waypoint in metres. If None, defaults to zeros
            (i.e., waypoints are only considered reached at exact coincidence).

        Returns
        -------
        None

        Notes
        -----
        Side effects (in-place updates):
        - Sets `positions0`, `positions`, `velocities0`, `velocities`.
        - Sets `destinations`, `destinations_range`.
        - Does not reset `destinations_indices`; users should set it explicitly
          if reinitialising mid-run.
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

    def set_boundaries(
        self,
        boundaries: npt.NDArray,
        boundary_selection="nearest",
        boundary_verlet_sphere=1.0,
    ):
        """
        Set line-segment boundaries (walls).

        Parameters
        ----------
        boundaries : (M, 2, 2) array_like
            Boundary segments in metres. Each segment is `[[x0, y0], [x1, y1]]`.
        boundary_selection : string, default="nearest"
            How to compute the boundary force. Can be "nearest" to use only the
            nearest boundary, or "superpose" to use a superposition from all the
            boundaries
        boundary_verlet_sphere : float, default=1.
            Used only if boundary_selection == "superpose". Discards any boundary
            whose distance from the pedestrian is greate than its value.

        Returns
        -------
        None
        """
        self.boundaries = np.asarray(boundaries, float)
        self.boundary_selection = boundary_selection
        self.boundary_verlet_sphere = boundary_verlet_sphere

    def driving_force(self, velocities, desired_directions, desired_speeds):
        """
        Compute the driving (goal-seeking) acceleration.

        Parameters
        ----------
        velocities : (N, 2) ndarray
            Current velocities in m/s.
        desired_directions : (N, 2) ndarray
            Unit vectors pointing toward the active destination (world frame).
        desired_speeds : (N, 1) ndarray
            Desired speed magnitudes in m/s.

        Returns
        -------
        f_drive : (N, 2) ndarray
            Driving acceleration in m/s^2.

        Notes
        -----
        Implements the relaxation model:

        .. math::
            a_i = \\frac{1}{\\tau}\\,(v_i^{\\ast} \\hat{e}_i - v_i)

        where `tau = relaxation_times`, `v_i^{*}` is `desired_speeds[i]`, and
        `hat{e}_i` is `desired_directions[i]`.
        """
        force = (
            1
            / self.relaxation_times
            * (desired_speeds * desired_directions - velocities)
        )
        if "driving_forces" in self.results.keys():
            self.forces["driving"] = force
        return force

    def calc_fovs(self, velocities, unit_vectors):
        """
        Compute anisotropic field-of-view (FOV) weighting for interactions.

        Parameters
        ----------
        velocities : (N, 2) ndarray
            Velocities in m/s.
        unit_vectors : (N, N, 2) ndarray
            Unit vectors from pedestrian j to i (direction of repulsion contribution).

        Returns
        -------
        fov : (N, N, 1) ndarray
            Interaction weight in [anisotropic_character, 1], where lower values
            down-weight interactions “behind” the agent.

        Notes
        -----
        The code computes:

        - `velocity_norms[i] = v_i / ||v_i||` (or zeros if speed is 0)
        - `cos_phi = - <norm_vectors[i,j], velocity_norms[i]>`

        and then maps `cos_phi` into a convex combination between
        `anisotropic_character` and 1.
        """
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        unit_velocities = np.divide(
            velocities, speeds, out=np.zeros_like(velocities), where=(speeds != 0)
        )
        cos_phi = -np.sum(
            unit_vectors * unit_velocities[:, None, :], axis=2, keepdims=True
        )
        return (
            self.anisotropic_character
            + (1.0 - self.anisotropic_character) * (1.0 + cos_phi) / 2.0
        )

    def repulsive_force_numpy(self, positions, radii, velocities):
        """
        Compute pairwise repulsive acceleration using vectorised NumPy.

        Parameters
        ----------
        positions : (N, 2) ndarray
            Positions in metres.
        radii : (N, 1) ndarray
            Radii in metres.
        velocities : (N, 2) ndarray
            Velocities in m/s.

        Returns
        -------
        f_rep : (N, 2) ndarray
            Net repulsive acceleration on each pedestrian in m/s^2.

        Notes
        -----
        Computes an exponential repulsion of the form (schematically):

        .. math::
            a_{ij} \\propto A \\exp\\left(\\frac{r_i + r_j - d_{ij}}{B}\\right)\\,\\hat{n}_{ij}

        with two terms `(A1,B1)` and `(A2,B2)`; the first term is FOV-weighted.

        Interactions are hard-cutoff at self.Verlet_sphere m in this path.
        """
        radii_sum = radii[:, None, :] + radii[None, :, :]
        relative_positions = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(relative_positions, axis=2, keepdims=True)
        unit_vectors = np.divide(
            relative_positions,
            distances,
            out=np.zeros_like(relative_positions),
            where=(distances != 0),
        )
        fov = self.calc_fovs(velocities, unit_vectors)

        repulsive_force_mat = (
            self.A1 * np.exp((radii_sum - distances) / self.B1) * unit_vectors * fov
            + self.A2 * np.exp((radii_sum - distances) / self.B2) * unit_vectors
        )
        repulsive_force_mat = np.where(
            distances <= self.Verlet_sphere,
            repulsive_force_mat,
            np.zeros_like(repulsive_force_mat),
        )

        return np.sum(repulsive_force_mat, axis=1)

    @staticmethod
    @njit(parallel=True)
    def repulsive_force_njit(
        positions,
        radii,
        velocities,
        anisotropic_character,
        Verlet_sphere,
        A1,
        B1,
        A2,
        B2,
    ):
        """
        Compute pairwise repulsive acceleration using Numba.

        Parameters
        ----------
        positions : (N, 2) ndarray
            Positions in metres.
        radii : (N, 1) ndarray
            Radii in metres.
        velocities : (N, 2) ndarray
            Velocities in m/s.
        anisotropic_character : float
            Baseline anisotropy weight in [0, 1].
        Verlet_sphere : float
            Interaction cutoff distance in metres (skip pairs with d > cutoff).
        A1, B1, A2, B2 : float
            Exponential repulsion parameters.

        Returns
        -------
        f_rep : (N, 2) ndarray
            Net repulsive acceleration on each pedestrian in m/s^2.

        Notes
        -----
        This is a static Numba-compiled kernel. It allocates `(N, N, 2)` working
        arrays; memory use is O(N^2).
        """
        n_pedestrians = positions.shape[0]
        repulsive_force_mat = np.zeros((n_pedestrians, n_pedestrians, 2))
        unit_vectors = np.zeros_like(repulsive_force_mat)
        unit_velocities = np.zeros_like(positions)
        for i in prange(n_pedestrians):
            speeds = np.linalg.norm(velocities[i])
            if speeds != 0.0:
                unit_velocities[i] = velocities[i] / speeds
            for j in prange(n_pedestrians):
                if i == j:
                    continue
                relative_positions = positions[i] - positions[j]
                distance = np.linalg.norm(relative_positions)
                if distance > Verlet_sphere:
                    continue

                radii_sum = radii[i, 0] + radii[j, 0]
                if distance != 0:
                    unit_vectors[i, j] = relative_positions / distance
                cos_phi = -np.sum(unit_vectors[i, j] * unit_velocities[i])
                fov = (
                    anisotropic_character
                    + (1.0 - anisotropic_character) * (1 + cos_phi) / 2.0
                )

                repulsive_force_mat[i, j] = (
                    A1 * np.exp((radii_sum - distance) / B1) * unit_vectors[i, j] * fov
                    + A2 * np.exp((radii_sum - distance) / B2) * unit_vectors[i, j]
                )

        return np.sum(repulsive_force_mat, axis=1)

    def repulsive_force(self, positions, radii, velocities):
        """
        Compute repulsive acceleration between pedestrians.

        Parameters
        ----------
        positions : (N, 2) ndarray
            Positions in metres.
        radii : (N, 1) ndarray
            Radii in metres.
        velocities : (N, 2) ndarray
            Velocities in m/s.

        Returns
        -------
        f_rep : (N, 2) ndarray
            Net repulsive acceleration in m/s^2.

        Notes
        -----
        Dispatches to a Numba-accelerated kernel when `n_pedestrians > 250`,
        otherwise uses the vectorised NumPy implementation.
        """
        if self.n_pedestrians > 250:
            force = self.repulsive_force_njit(
                positions,
                radii,
                velocities,
                self.anisotropic_character,
                self.Verlet_sphere,
                self.A1,
                self.B1,
                self.A2,
                self.B2,
            )
        else:
            force = self.repulsive_force_numpy(positions, radii, velocities)
        if "repulsive_forces" in self.results.keys():
            self.forces["repulsive"] = force
        return force

    def boundary_force(self, positions: npt.NDArray):
        """
        Compute repulsive acceleration from the nearest boundary segment.

        Parameters
        ----------
        positions : (N, 2) ndarray
            Positions in metres.

        Returns
        -------
        f_wall : (N, 2) ndarray
            Boundary repulsion acceleration in m/s^2.

        Notes
        -----
        - Each agent interacts only with its nearest boundary segment (by
          Euclidean distance to the closest point on each segment).
        - If `self.boundaries is None`, returns zeros.

        The distance to a segment is computed by projection onto the segment
        parameter t in [0, 1].
        """
        if self.boundaries is None:
            return np.zeros_like(positions)
        p = positions[:, None, :]  # (N,1,2)
        b0 = self.boundaries[:, 0, :][None, :, :]  # (1,M,2) boundary starts
        b1 = self.boundaries[:, 1, :][None, :, :]  # (1,M,2) boundary ends

        v = b1 - b0  # (1,M,2) boundary directions vector
        w = p - b0  # (N,M,2) position to boundary vector

        vv = np.sum(v * v, axis=2, keepdims=True)  # (1,M,1) Norm of v vectors
        vw = np.sum(w * v, axis=2, keepdims=True)  # (N,M,1) dot product of v and w

        t = vw / vv  # (N,M,1) norm vector towards closest point on each boundary
        t = np.clip(t, 0.0, 1.0)  # projection along segment
        closest = b0 + t * v  # (N,M,2)

        diff = p - closest  # (N,M,2)

        dist = np.linalg.norm(diff, axis=2, keepdims=True)  # (N,M,1)

        if self.boundary_selection == "nearest":
            idx = np.argmin(dist, axis=1)[:, 0]  # (N,)
            nearest_dist = dist[np.arange(self.n_pedestrians), idx]
            nearest_dir = diff[np.arange(self.n_pedestrians), idx] / nearest_dist

            force_mag = self.Ab * np.exp((self.radii - nearest_dist) / self.Bb)
            force = force_mag * nearest_dir

        elif self.boundary_selection == "superpose":
            mask = dist < self.boundary_verlet_sphere
            directions = np.divide(
                diff, dist, out=np.zeros_like(diff), where=(dist != 0)
            )
            force_mag = self.Ab * np.exp((self.radii[:, None, :] - dist) / self.Bb)
            force_mag *= mask.astype(force_mag.dtype)
            force = np.sum(force_mag * directions, axis=1)
        else:
            raise

        if "boundary_forces" in self.results.keys():
            self.forces["boundary"] = force
        return force

    def total_force(self, positions, desired_speeds, velocities):
        """
        Compute total acceleration (sum of model components).

        Parameters
        ----------
        positions : (N, 2) ndarray
            Positions in metres.
        desired_speeds : (N, 1) ndarray
            Desired speeds in m/s.
        velocities : (N, 2) ndarray
            Velocities in m/s.

        Returns
        -------
        f_total : (N, 2) ndarray
            Total acceleration in m/s^2.

        Notes
        -----
        The returned acceleration is:

        - driving_force(...)
        - + repulsive_force(...)
        - + boundary_force(...)
        """
        desired_directions = self.calc_desired_directions()
        total_force = (
            self.driving_force(velocities, desired_directions, desired_speeds)
            + self.repulsive_force(positions, self.radii, velocities)
            + self.boundary_force(positions)
        )
        if "total_forces" in self.results.keys():
            self.forces["total"] = total_force
        return total_force

    def calc_desired_speeds(self, t, positions):
        """
        Compute time-varying desired speeds based on “impatience”.

        Parameters
        ----------
        t : float
            Current time in seconds.
        positions : (N, 2) ndarray
            Current positions in metres.

        Returns
        -------
        desired_speeds : (N, 1) ndarray
            Desired speeds in m/s.

        Notes
        -----
        - At `t == 0`, returns `desired_speeds0`.
        - For `t > 0`, computes an average speed since t=0 as
          `||positions - positions0|| / t`, and defines:

        .. math::
            \\text{impatience} = 1 - \\frac{\\bar{v}}{v_0}

        then interpolates between `desired_speeds0` and `maximal_speeds`.

        This is a global-from-start measure; it does not reset per waypoint.
        """
        if t == 0.0:
            return self.desired_speeds0
        pos0_to_pos = positions - self.positions0
        avg_speed = np.linalg.norm(pos0_to_pos / t, axis=1, keepdims=True)
        impatience = 1 - avg_speed / self.desired_speeds0
        new_desired_speeds = (
            1 - impatience
        ) * self.desired_speeds0 + impatience * self.maximal_speeds
        return new_desired_speeds

    def calc_desired_directions(self) -> npt.NDArray:
        """
        Compute unit direction vectors from positions to destinations.

        Returns
        -------
        desired_directions : (N, 2) ndarray
            Unit vectors pointing from each position to its destination. Zeros
            where position equals destination.
        """
        desired_directions = self.current_destinations() - self.positions
        desired_directions_norm = np.linalg.norm(
            desired_directions, axis=1, keepdims=True
        )
        desired_directions = np.divide(
            desired_directions,
            desired_directions_norm,
            out=np.zeros_like(desired_directions),
            where=(desired_directions_norm != 0),
        )
        return desired_directions

    def current_destinations(self) -> npt.NDArray:
        """
        Give current destination of pedestrians

        Returns
        -------
        current_destinations : (N, 2) ndarray
            Coordinates to which each pedestrian currently wishes to go
        """
        return self.destinations[
            np.arange(self.n_pedestrians), self.destinations_indices
        ]

    def current_destinations_range(self) -> npt.NDArray:
        """
        Give the distance from the current destination within which a pedestrian
        must be to have reached their current destination.

        Returns
        -------
        current_destinations : (N, 2) ndarray
            Coordinates to which each pedestrian currently wishes to go
        """
        return self.destinations_range[
            np.arange(self.n_pedestrians), self.destinations_indices
        ]

    def update_destinations(self):
        """
        Advance waypoint indices for pedestrians that reached their active target.

        Returns
        -------
        None

        Notes
        -----
        Side effects (in-place updates):
        - Increments `destinations_indices[i]` for pedestrians satisfying:
          - distance to active waypoint <= `destinations_range[i, idx]`
          - idx is not already the last waypoint index

        The update is vectorised; all eligible agents advance by exactly one
        waypoint per call.
        """
        pedestrians_in_range = (
            np.linalg.norm(self.current_destinations() - self.positions, axis=1)
            <= self.current_destinations_range().flatten()
        )
        pedestrians_not_at_last_dest = (
            self.destinations_indices < self.destinations.shape[1] - 1
        )
        mask = np.where(pedestrians_in_range & pedestrians_not_at_last_dest)[0]
        self.destinations_indices[mask] += 1
