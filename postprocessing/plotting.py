from crowd_dynamics.social_force import SocialForce
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.typing import ArrayLike


def plot(positions, SF, show=True, continuous=False, save=False, colors: ArrayLike | None = None, title="A Title", x_bound: tuple | None = None, y_bound: tuple | None = None, filetype='png'):
    positions = np.array(positions)
    fig, ax = plt.subplots()
    colors = colors if colors is not None else ('b',)

    if continuous:
        if len(positions.shape) == 3:
            for p in range(positions.shape[1]):
                ax.plot(positions[:,p,0],positions[:,p,1], color=colors[p])
        else:
            ax.plot(positions[:,0],positions[:,1], color=colors[p])
    else:
        ax.scatter(positions[:,0],positions[:,1], c=colors, s=100)
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

    if save:
        fig.savefig(f"{save}.{filetype}")

    if show:
        plt.show()
