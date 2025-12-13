from crowd_dynamics.social_force import SocialForce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.interpolate import interp1d
from numpy.typing import ArrayLike

def regularise_positions(all_times, all_positions, regularised_timesteps):
    regularised_positions = np.zeros((len(regularised_timesteps), all_positions.shape[1], 2))
    for i in range(all_positions.shape[1]):
        for d in range(2):
            f = interp1d(all_times, all_positions[:,i,d], kind='linear')
            regularised_positions[:, i, d] = f(regularised_timesteps)
    return regularised_positions


def make_movie(all_times, all_positions, SF, show_animation=True, interval=100, save=False, fps=30, regularised_timesteps: ArrayLike | bool =False, colors: ArrayLike | None = None, title="A Title", x_bound: tuple | None = None, y_bound: tuple | None = None):
    all_positions = np.array(all_positions)
    n_pedestrians = all_positions.shape[1]
    fig, ax = plt.subplots()
    colors = colors if colors is not None else np.array(['b']*n_pedestrians)

    if regularised_timesteps is not False:
        positions = regularise_positions(all_times, all_positions, regularised_timesteps)
    else:
        positions = all_positions
#    scat = ax.scatter([positions[0][:,0]],[positions[0][:,1]], c=colors, s=100)
    circles = []
    for i in range(n_pedestrians):
        c = Circle(positions[0][i], SF.radii[i], color=colors[i])
        ax.add_patch(c)
        circles.append(c)
    if SF.boundaries is not None:
        for line in SF.boundaries:
            ax.plot(line[:, 0], line[:, 1], color='black', linewidth=2)
    if x_bound is not None:
        ax.set_xlim(x_bound)
    if y_bound is not None:
        ax.set_ylim(y_bound)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    def update(frame):
        data = positions[frame]
        for i, circle in enumerate(circles):
            circle.center = data[i]
        return circles
#        scat.set_offsets(data)
#        return [scat]

    ani = FuncAnimation(fig, update, frames=len(positions), interval=interval, blit=True)

    if save:
        ani.save(f"{save}.mp4", writer="ffmpeg", fps=fps)

    if show_animation:
        plt.show()
