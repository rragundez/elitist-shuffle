from os.path import join as pjoin
from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation


def shuffle_simulations(shuffle_func, n_items, n_simulations=5000):
    """Simulate shuffling of a sequence of items and aggregate statistics

    Args:
        shuffle_func: function to perform the shuffling. If shuffle
            inplace function should return None.
        n_items (int): number of elements in the sequence
        n_simulations (int): number of simulations to perform

    Returns:
        dict ({int: collections.Counter}): the key is the initial
            position. The value contains the statistics of the
            end position after the shuffling.
    """
    landing_counts = defaultdict(Counter)
    freq_factor = 1 / n_simulations
    for _ in range(n_simulations):
        items = list(range(n_items))
        s_items = shuffle_func(items) or items
        for item in range(n_items):
            landing_counts[item][s_items.index(item)] += freq_factor
    return landing_counts


def make_animation(agg_simulations, n_items, interval, save_gif_as=None):
    """Make a matplotlib FuncAnimation displaying the results of
    the shuffling simulation for all initial positions
    """
    def plot_single(index, all_counts, n_items):
        ax.set_xticks(range(n_items))
        ax.set_xticklabels(range(n_items))
        ax.set_ylim(0, 1.01)
        ax.set_xlim(-0.1, n_items - 1 + 0.1)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Position')
        count = all_counts[index]
        sorted_count = sorted(count.items())
        x = [c[0] for c in sorted_count]
        y = [c[1] for c in sorted_count]
        distribution.set_data(x, y)
        init_position.set_data([index, index], [0, 1])
        ax.legend(loc='upper right')
        return (distribution,)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    init_position, = ax.plot([], [], linewidth=2, color='r', linestyle='--',
                             label='Position before shuffle')
    distribution, = ax.plot([], [], linewidth=2, color='b', linestyle='-',
                            label='Probability of position after shuffle')
    anim = FuncAnimation(fig, plot_single, fargs=[agg_simulations, n_items],
                         frames=n_items, interval=interval)
    if save_gif_as:
        print(f'Saving animation to {save_gif_as}')
        anim.save(save_gif_as, writer='imagemagick', fps=1000 / interval)
    return anim


def elitist_shuffle(items, inequality):
    """Shuffle array with bias over initial ranks

    A higher ranked content has a higher probability to end up higher
    ranked after the shuffle than an initially lower ranked one.

    Args:
        items (numpy.array): Items to be shuffled
        inequality (int/float): how biased you want the shuffle to be.
            A higher value will yield a lower probabilty of a higher initially
            ranked item to end up in a lower ranked position in the
            sequence.
    """
    weights = np.power(
        np.linspace(1, 0, num=len(items), endpoint=False),
        inequality
    )
    weights = weights / np.linalg.norm(weights, ord=1)
    return list(np.random.choice(
                items, size=len(items), replace=False, p=weights
                ))
