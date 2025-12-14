import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from numpy import polyfit

# This file is only for recreating p

output_dir = "crowd_simulation_results"


dist_statistics = np.genfromtxt('dist_stats.csv', delimiter=',')
time_statistics = np.genfromtxt('time_stats.csv', delimiter=',')
rad_statistics = np.genfromtxt('rad_stats.csv', delimiter=',')

print(np.max(rad_statistics))
print(np.min(rad_statistics))

big_ind = np.array([i for i in range(len(rad_statistics)) if rad_statistics[i] > 0.051])
small_ind = np.array([i for i in range(len(rad_statistics)) if i not in big_ind])


# Simple scatter plot for serving time vs distance
cmap = cm.get_cmap('winter')
norm = mcolors.Normalize(vmin=min(rad_statistics), vmax=max(rad_statistics))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(dist_statistics[big_ind], 10 * time_statistics[big_ind] / len(rad_statistics), alpha=0.6, s=20, cmap=cmap, norm=norm, c=rad_statistics[big_ind])

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
#ax.text(0.02, 0.98, f'{default_params}', transform=ax.transAxes,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(
    f"{output_dir}/time_vs_distance_inhom_big.png",
    dpi=150, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(dist_statistics[small_ind], 10 * time_statistics[small_ind] / len(rad_statistics), alpha=0.6, s=20, cmap=cmap, norm=norm, c=rad_statistics[small_ind])

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
#ax.text(0.02, 0.98, f'{default_params}', transform=ax.transAxes,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(
    f"{output_dir}/time_vs_distance_inhom_small.png",
    dpi=150, bbox_inches='tight')
plt.close()


color = np.ones_like(rad_statistics)
color[big_ind] = np.max(rad_statistics)
color[small_ind] = np.min(rad_statistics)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(dist_statistics, 10 * time_statistics / len(rad_statistics), alpha=0.6, s=20, cmap=cmap, norm=norm, c=color)

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
#ax.text(0.02, 0.98, f'{default_params}', transform=ax.transAxes,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(
    f"{output_dir}/time_vs_distance_inhom_binary.png",
    dpi=150, bbox_inches='tight')
plt.close()



small_params = polyfit(dist_statistics[small_ind], np.sqrt(time_statistics[small_ind]), 1)
small_p = np.poly1d(small_params)
big_params = polyfit(dist_statistics[big_ind], np.sqrt(time_statistics[big_ind]), 1)
big_p = np.poly1d(big_params)


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(dist_statistics, 10 * time_statistics / len(rad_statistics), alpha=0.6, s=20, cmap=cmap, norm=norm, c=rad_statistics)

# Add sequential serving law for comparison
d_plot = np.linspace(0, 1, 100)
sequential_steps = (d_plot / 1) ** 2
ax.plot(d_plot, sequential_steps, 'r--', linewidth=2, label='Sequential Serving')
ax.plot(d_plot, 10 * small_p(d_plot) ** 2 / len(rad_statistics), 'b--', linewidth=1)
ax.plot(d_plot, 10 * big_p(d_plot) ** 2 / len(rad_statistics), 'g--', linewidth=1)


ax.set_xlabel('Initial Distance from Counter')
ax.set_ylabel('Serving Steps / Number of agents')
ax.set_title('Serving Steps vs Initial Distance')
ax.legend()
ax.grid(True, alpha=0.3)

# Show crowd parameters
#ax.text(0.02, 0.98, f'{default_params}', transform=ax.transAxes,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(
    f"{output_dir}/time_vs_distance_with_fits.png",
    dpi=150, bbox_inches='tight')
plt.close()