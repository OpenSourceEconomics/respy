from matplotlib import pyplot as plt
from shapely import geometry as sg
import descartes

from auxiliary_economics import move_subdirectory

NUM_POINTS = 3
KL_MAX = 0.02


def plot():

    move_subdirectory()

    # Translate to circle coordinates.
    center_coordinates = (0.0, 0.0)
    radius = KL_MAX * 2

    # Create circles.
    a = sg.Point(center_coordinates).buffer(radius)
    b = sg.Point(center_coordinates).buffer(radius * 1.5)
    c = sg.Point(center_coordinates).buffer(radius * 2.0)

    # Determine relevant sets for plotting.
    small = a.intersection(b)
    middle = b.difference(a)
    large = c.difference(b)

    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    ax.add_patch(descartes.PolygonPatch(small, fc='r', ec='k', label=r'Low Ambiguity'))
    ax.add_patch(descartes.PolygonPatch(middle, fc='orange', ec='k', label=r'High Ambiguity'))

    # Initialize and format plot.
    plt.ylim((-0.10, 0.10));
    plt.xlim((-0.10, 0.10))

    plt.xlabel(r'$\mu_{\epsilon_{1,t}}$', fontsize=20)
    plt.ylabel(r'$\mu_{\epsilon_{2,t}}$', fontsize=20)

    # Indicate baseline model.
    plt.axhline(y=0.00, xmin=0.00, xmax=0.50, linewidth=1, color='grey', linestyle='--')

    plt.axvline(x=0.00, ymin=0.0, ymax=0.50, linewidth=1, color='grey', linestyle='--')

    plt.plot(0, 0, linestyle='--', color='grey', marker='o')

    # Positioning
    box = ax.get_position()

    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Add legend.
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=False, frameon=False,
        shadow=False, ncol=2, fontsize=20)

    # Remove first tick from x-axis.
    ax.xaxis.set_ticklabels([]), ax.yaxis.set_ticklabels([])

    # Save.
    plt.savefig('entropy.png')