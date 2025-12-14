import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import warnings

# Filter out warnings
warnings.filterwarnings("ignore")

# Create output directory for saving images
output_dir = "crowd_simulation_results"
os.makedirs(output_dir, exist_ok=True)

default_params = {
        'N': 150,  # Reduced for demo
        'R': 1.0,  # Domain radius
        'phi': 0.4,  # Reduced area fraction for easier computation
        'p': 0.2,  # Probability of non-radial moves
        'delta_r': 0.2  # (Non)Homogeneous crowd
    }

def count_points_in_a_circle(R):
    n = 0
    for i in range(int(-R), int(R) + 1):
        for j in range(int(-R), int(R) + 1):
            if i ** 2 + j ** 2 < R ** 2:
                n += 1
    return n

def find_grid_size(N):
    R = 0.5
    while count_points_in_a_circle(R) < N:
        R += 0.5
    return int(2 * R + 1)

class CrowdMCImproved:
    def __init__(self, N=300, R=1.0, phi=0.6, p=0.2, delta_r=0.2, K=30, seed = 42):
        self.N = N
        self.R = R
        self.phi = phi
        self.p = p
        self.delta_r = delta_r
        self.K = K
        np.random.seed(seed)

        # Initialize radii first
        self.radii = self._generate_radii()

        # MC parameters
        self.mc_step = 0.01
        self.target_acceptance = 0.5

        # Initialize positions using paper's method
        self.positions, self.initial_lattice = self.initialize()

        # Serving process
        self.served_agents = []
        self.serving_times = np.zeros(N)
        self.current_step = 0

        # Setting up colors for pics
        self.cmap = cm.get_cmap('winter')
        self.norm = mcolors.Normalize(vmin=min(self.radii), vmax=max(self.radii))

    def _generate_radii(self):
        """Generate agent radii based on homogeneity parameter"""
        if self.delta_r == 0:
            # Start with unit radii, will be rescaled later
            return np.ones(self.N)
        else:
            z = np.random.uniform(0, 1, self.N)
            return 1 + (2 * z - 1) * self.delta_r

    def _create_square_lattice_in_circle(self):
        grid_size = find_grid_size(self.N)

        # Create grid points
        x = np.linspace(-1., 1., grid_size)
        y = np.linspace(-1., 1., grid_size)

        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([x[i], y[j]])
                # Check if point is inside circle
                if np.linalg.norm(pos) <= self.R:
                    positions.append(pos)

        # Convert to numpy array
        positions = np.array(positions)

        # If we have more points than needed, select N random points
        if len(positions) > self.N:
            indices = np.random.choice(len(positions), self.N, replace=False)
            positions = positions[indices]

        return positions

    def _rescale_radii_to_match_phi(self):
        """Rescale radii to achieve exact area fraction φ"""
        current_total_area = np.sum(self.radii ** 2)
        target_total_area = self.phi * self.R ** 2
        scaling_factor = np.sqrt(target_total_area / current_total_area)
        self.radii *= scaling_factor

    def _check_overlap(self, pos1, pos2, r1, r2):
        """Check if two agents overlap"""
        return np.linalg.norm(pos1 - pos2) < (r1 + r2)

    def _check_boundary(self, position, radius):
        """Check if agent is within boundary"""
        return np.linalg.norm(position) <= (self.R)

    def _attempt_move(self, position, radius, other_positions, other_radii, random_dir = False):
        """Attempt Monte Carlo move for one agent"""
        if np.linalg.norm(position) == 0:
            radial_dir = np.array([0.0, 0.0])
        else:
            radial_dir = -position / np.linalg.norm(position)

        if not random_dir:
            radial_move = self.mc_step * radial_dir
            if np.random.random() < self.p:
                angle = np.random.uniform(-np.pi / 2, np.pi / 2)
                non_radial_dir = np.array([
                    radial_dir[0] * np.cos(angle) - radial_dir[1] * np.sin(angle),
                    radial_dir[0] * np.sin(angle) + radial_dir[1] * np.cos(angle)
                ])
                non_radial_move = self.mc_step * non_radial_dir
            else:
                non_radial_move = np.array([0.0, 0.0])

            new_position = position + radial_move + non_radial_move
            if np.linalg.norm(new_position) > np.linalg.norm(position):
                new_position = position
        else:
            random_move = 2 * np.random.random(2) - 1
            new_position = position + random_move

        if not self._check_boundary(new_position, radius):
            return position, False

        for i, (other_pos, other_rad) in enumerate(zip(other_positions, other_radii)):
            if self._check_overlap(new_position, other_pos, radius, other_rad):
                return position, False

        return new_position, True

    def _initial_mc_relaxation(self, M_I = 5000):
        """Perform initial MC relaxation with p=0"""

        original_mc_step = self.mc_step
        original_p = self.p
        self.p = 0

        for i in range(M_I):
            accepted_moves = 0
            total_attempts = 0

            indices = np.random.permutation(self.N)
            for idx in indices:
                current_pos = self.positions[idx].copy()
                radius = self.radii[idx]

                other_indices = [i for i in range(self.N) if i != idx]
                other_positions = self.positions[other_indices]
                other_radii = self.radii[other_indices]

                new_pos, accepted = self._attempt_move(
                    current_pos, radius, other_positions, other_radii, random_dir=True
                )

                if accepted:
                    self.positions[idx] = new_pos
                    accepted_moves += 1
                total_attempts += 1

            acceptance_rate = accepted_moves / total_attempts if total_attempts > 0 else 0

            # Adjust step sizes to maintain target acceptance
            if acceptance_rate < self.target_acceptance:
                self.mc_step *= 0.95
            else:
                self.mc_step *= 1.05

        self.p = original_p
        self.mc_step = original_mc_step
        print("Initial relaxation complete.")


    def _mc_relaxation(self, active_indices, M = 200):

        # Calibration of delta_r and delta_u before last step of M_I
        for i in range(M):
            # Calibrating mc_step size every step
            accepted_moves = 0
            total_attempts = 0

            # Random order for moves
            randomised_indices = np.random.permutation(self.N)
            indices = [i for i in randomised_indices if i in active_indices]

            for idx in indices:
                current_pos = self.positions[idx].copy()
                radius = self.radii[idx]

                # Get positions of other agents
                other_indices = [i for i in indices if i != idx]
                other_positions = self.positions[other_indices]
                other_radii = self.radii[other_indices]

                new_pos, accepted = self._attempt_move(
                    current_pos, radius, other_positions, other_radii
                )

                if accepted:
                    self.positions[idx] = new_pos
                    accepted_moves += 1

                total_attempts += 1

            acceptance_rate = accepted_moves / total_attempts if total_attempts > 0 else 0

            # Adjust step sizes to maintain target acceptance
            if acceptance_rate < self.target_acceptance:
                self.mc_step *= 0.95
            else:
                self.mc_step *= 1.05

        self.mc_step = 0.01

    def initialize(self):
        """Initialize positions following the paper's method exactly"""
        print("Initializing positions...")

        # Step 1: Square lattice placement
        initial_lattice = self._create_square_lattice_in_circle()

        # Step 2: Radius rescaling to match area fraction
        self._rescale_radii_to_match_phi()

        # Step 3: Initial MC relaxation
        print("Performing initial MC relaxation...")
        self.positions = initial_lattice.copy()
        self._initial_mc_relaxation()
        initial_lattice = self.positions.copy()

        return self.positions, initial_lattice

    def save_current_state_pic(self, step, active_indices):
        fig, ax = plt.subplots(figsize=(8, 8))

        for i, (pos, r) in enumerate(zip(self.positions[active_indices], self.radii[active_indices])):
            color = self.cmap(self.norm(r))
            circle = Circle(pos, r, fill=True, alpha=0.6, edgecolor='black', facecolor=color)
            ax.add_patch(circle)

        boundary_circle = Circle((0, 0), self.R, fill=False, color='red', linewidth=2)
        ax.add_patch(boundary_circle)
        ax.set_xlim(-self.R * 1.1, self.R * 1.1)
        ax.set_ylim(-self.R * 1.1, self.R * 1.1)
        ax.set_aspect('equal')
        ax.set_title('Step ' + str(step))
        ax.axis('off')
        #ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/states/state_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def save_initialization_plots(self, filename_prefix="init"):
        """Save initialization process as image files"""
        print("Saving initialization plots...")

        # Initial configuration
        fig, ax = plt.subplots(figsize=(8, 8))

        for i, (pos, r) in enumerate(zip(self.positions, self.radii)):
            color = self.cmap(self.norm(r))
            circle = Circle(pos, r, fill=True, alpha=0.6, edgecolor='black', facecolor=color)
            ax.add_patch(circle)

        boundary_circle = Circle((0, 0), self.R, fill=False, color='red', linewidth=2)
        ax.add_patch(boundary_circle)
        ax.set_xlim(-self.R * 1.1, self.R * 1.1)
        ax.set_ylim(-self.R * 1.1, self.R * 1.1)
        ax.set_aspect('equal')
        ax.set_title('After MC Relaxation')
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)

        actual_phi = np.sum(self.radii ** 2) / self.R ** 2
        ax.text(0.02, 0.98, f'φ = {actual_phi:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename_prefix}_after_relaxation.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 3: Distance and radius distributions
        distances = np.linalg.norm(self.positions, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(distances, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.set_xlabel('Distance from Center')
        ax1.set_ylabel('Number of Agents')
        ax1.set_title('Distance Distribution')
        ax1.grid(True, alpha=0.3)

        ax2.hist(self.radii, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
        ax2.set_xlabel('Agent Radius')
        ax2.set_ylabel('Number of Agents')
        ax2.set_title('Radius Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename_prefix}_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved initialization plots to {output_dir}/")
        return actual_phi

    def run_serving_process(self, max_steps=None, states_snapshots = False):
        """Run the serving process simulation"""
        if max_steps is None:
            max_steps = self.N

        print("Starting serving process...")
        self.served_agents = []
        self.serving_times = np.zeros(self.N)
        #self.current_step = 0

        # Store positions at different stages for visualization
        self.snapshot_steps = [0, max(1, self.N // 4), max(1, self.N // 2), max(1, 3 * self.N // 4)]
        self.snapshots = {}
        self.snapshots[0] = self.positions.copy()

        for step in range(1, max_steps + 1):
            if len(self.served_agents) >= self.N:
                break

            # Find closest active agent to serve
            active_indices = np.array([i for i in range(self.N) if i not in self.served_agents])
            if not active_indices.any():
                break

            distances = np.linalg.norm(self.positions[active_indices], axis=1)
            closest_idx = active_indices[np.argmin(distances)]

            # Serve this agent
            self.serving_times[closest_idx] = step
            self.served_agents.append(closest_idx)
            np.delete(active_indices, np.argmin(distances))

            # Perform MC relaxation

            self._mc_relaxation(active_indices)

            # Save state pic
            if states_snapshots:
                self.save_current_state_pic(step, active_indices)

            # Take snapshot if at specified step
            if step in self.snapshot_steps:
                self.snapshots[step] = self.positions.copy()

            if step % 50 == 0:
                print(f"  Step {step}/{max_steps}, Agents served: {len(self.served_agents)}")

    def save_serving_analysis(self, filename_prefix="serving"):
        """Save serving process analysis plots"""
        print("Saving serving analysis plots...")

        # Calculate serving statistics
        distances = np.linalg.norm(self.initial_lattice, axis=1)
        serving_steps = self.serving_times

        # Simple scatter plot for serving time vs distance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(distances, serving_steps / self.N, alpha=0.6, s=20, cmap=self.cmap, norm=self.norm, c=self.radii)

        # Add sequential serving law for comparison
        d_plot = np.linspace(0, self.R, 100)
        sequential_steps = (d_plot / self.R) ** 2
        ax.plot(d_plot, sequential_steps, 'r--', linewidth=2, label='Sequential Serving')

        ax.set_xlabel('Initial Distance from Counter')
        ax.set_ylabel('Serving Steps / Number of agents')
        ax.set_title('Serving Steps vs Initial Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Show crowd parameters
        ax.text(0.02, 0.98, f'{params}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename_prefix}_vs_distance_N_{params['N']}_phi_{params['phi']}_p_{params['p']}_delta_r_{params['delta_r']}.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved serving analysis plots to {output_dir}/")


def run_complete_simulation_once(params = default_params, save_init = True, save_analysis = True, summary = True, save_stats = False, save_snapshots = False, seed = 42):
    """Run complete simulation and save all results"""
    if summary:
        print("=== Crowd Dynamics Simulation ===")
        print(f"Results will be saved to: {output_dir}/")


    if summary:
        print(f"Parameters: {params}")

    try:
        # Create and initialize crowd
        crowd = CrowdMCImproved(**params, seed=seed)

        # Save initialization results
        if save_init:
            actual_phi = crowd.save_initialization_plots("initialization")

        # Run serving process
        crowd.run_serving_process(states_snapshots = save_snapshots)

        # Save serving analysis
        if save_analysis:
            crowd.save_serving_analysis("serving_analysis")

        if save_stats:
            distances = np.linalg.norm(crowd.initial_lattice, axis=1)
            serving_steps = crowd.serving_times
            return distances, serving_steps, crowd.radii

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


def run_complete_simulations(params = default_params, number = 10, seed = 42):
    dist_statistics = np.array([])
    time_statistics = np.array([])
    rad_statistics = np.array([])

    for i in range(number):
        print(f"Simulation No. {i + 1}")
        dists, times, rads = run_complete_simulation_once(params=params,save_init=False, save_analysis=False, summary=False, save_stats= True, seed=seed)
        dist_statistics = np.append(dist_statistics, dists)
        time_statistics = np.append(time_statistics, times)
        rad_statistics = np.append(rad_statistics, rads)

        seed += 1

    np.savetxt("dist_stats.csv", dist_statistics, delimiter=",")
    np.savetxt("time_stats.csv", time_statistics, delimiter=",")
    np.savetxt("rad_stats.csv", rad_statistics, delimiter=",")

    # Simple scatter plot for serving time vs distance
    cmap = cm.get_cmap('winter')
    norm = mcolors.Normalize(vmin=min(rad_statistics), vmax=max(rad_statistics))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dist_statistics, time_statistics / len(rads), alpha=0.6, s=20, cmap=cmap, norm=norm, c=rad_statistics)

    # Add sequential serving law for comparison
    d_plot = np.linspace(0, 1, 100)
    sequential_steps = (d_plot / 1) ** 2
    ax.plot(d_plot, sequential_steps, 'r--', linewidth=2, label='Sequential Serving')

    ax.set_xlabel('Initial Distance from Counter')
    ax.set_ylabel('Serving Steps / Number of agents')
    ax.set_title('Serving Steps vs Initial Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show crowd parameters
    ax.text(0.02, 0.98, f'{default_params}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/time_vs_distance_total.png",
        dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    #N = [100]
    #phi = [0.2, 0.4, 0.6]
    #p = [0.0, 0.1, 0.2, 0.4, 0.8]
    #delta_r = [0, 0.1, 0.2]

    N = [300]
    phi = [0.6]
    p = [0.2]
    delta_r = [0]
    for N_ in N:
        for phi_ in phi:
            for p_ in p:
                for del_ in delta_r:
                    params = {
                        'N': N_,  # Reduced for demo
                        'R': 1.0,  # Domain radius
                        'phi': phi_,  # Reduced area fraction for easier computation
                        'p': p_,  # Probability of non-radial moves
                        'delta_r': del_  # (Non)Homogeneous crowd
                    }
                    run_complete_simulation_once(params=params, save_snapshots=True)

    #run_complete_simulations(10)