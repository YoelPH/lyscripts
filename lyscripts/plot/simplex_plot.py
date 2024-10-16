import argparse
import logging
import numpy as np
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from lyscripts.plot.utils import COLORS, save_figure, p_to_xyz, add_perpendicular_crosses_3d
from lyscripts.utils import load_yaml_params

logger = logging.getLogger(__name__)

def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to the parser."""
    parser.add_argument(
        "--input", type=Path,
        help="File path of the mixture coefficients"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output path for the plot"
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file."
    )

    parser.set_defaults(run_main=main)

def plot_3d_simplex(mixture_df, data):
    subsites = list(mixture_df.columns)
    simplex_matrix = np.zeros((len(subsites), 3))
    for index, subsite in enumerate(subsites):
        simplex_matrix[index] = p_to_xyz(mixture_df[subsite])
    odered_value_counts = data['tumor']['1']['subsite'].value_counts()[subsites]
    sizes = odered_value_counts * 3

    # Sizes for each point
    ordered_value_counts = data['tumor']['1']['subsite'].value_counts()[subsites]
    sizes = np.array(ordered_value_counts) * 1  # Adjust the scaling if necessary

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates from simplex_matrix
    x_coords = simplex_matrix[:, 0]
    y_coords = simplex_matrix[:, 1]
    z_coords = simplex_matrix[:, 2]

    # Scatter plot with specific colors and sizes
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors_ordered, s=sizes, marker='o')

    # Add text labels for each subsite
    for i, subsite in enumerate(subsites):
        ax.text(x_coords[i], y_coords[i], z_coords[i], subsite)
        x_extremes = [-1,1,0,0]
        y_extremes = [-0.5773502691896258,-0.5773502691896258,1.1547005383792517,0]
        z_extremes = [-0.5773502691896258,-0.5773502691896258,-0.5773502691896258,1.2247448713915892]
    plotter_x = x_extremes.copy()
    plotter_x.append(x_extremes[0])
    plotter_x.append(x_extremes[2])
    plotter_y = y_extremes.copy()
    plotter_y.append(y_extremes[0])
    plotter_y.append(y_extremes[2])
    plotter_z = z_extremes.copy()
    plotter_z.append(z_extremes[0])
    plotter_z.append(z_extremes[2])

    ax.plot(plotter_x, plotter_y,plotter_z, c='k')
    ax.plot([x_extremes[1],x_extremes[3]], [y_extremes[1],y_extremes[3]],[z_extremes[1],z_extremes[3]], c='k')
    extremes = np.array((x_extremes,y_extremes,z_extremes)).T
    center0 = (extremes[1]+extremes[2]+extremes[3])/3
    center1 = (extremes[0]+extremes[2]+extremes[3])/3
    center2 = (extremes[0]+extremes[1]+extremes[3])/3
    center3 = (extremes[0]+extremes[1]+extremes[2])/3

    component_larynx = mixture.get_mixture_coefs()['C32.0'].argmax()
    component_oropharynx = mixture.get_mixture_coefs()['C01'].argmax()
    component_oral_cavity = mixture.get_mixture_coefs()['C03'].argmax()
    component_hypopharynx = mixture.get_mixture_coefs()['C13'].argmax()

    ax.text(extremes[component_larynx,0], extremes[component_larynx,1], extremes[component_larynx,2], "Larynx like", fontsize=10, ha='right', va='top',c = usz_orange)
    ax.text(extremes[component_oropharynx,0], extremes[component_oropharynx,1], extremes[component_oropharynx,2], "Oropharynx like", fontsize=10, ha='left', va='top',c = usz_green)
    ax.text(extremes[component_oral_cavity,0], extremes[component_oral_cavity,1], extremes[component_oral_cavity,2], "Oral cavity like", fontsize=10, ha='left', va='top',c = usz_blue)
    ax.text(extremes[component_hypopharynx,0], extremes[component_hypopharynx,1], extremes[component_hypopharynx,2], "Hypopharynx like", fontsize=10, ha='left', va='top',c = usz_red)



    add_perpendicular_crosses_3d(ax, extremes[0, 0], extremes[0, 1], extremes[0, 2], center0[0], center0[1], center0[2])
    plt.plot([extremes[0,0], center0[0]], [extremes[0,1], center0[1]],[extremes[0,2],center0[2]], color='gray', linestyle='--', linewidth=1)
    add_perpendicular_crosses_3d(ax, extremes[1, 0], extremes[1, 1], extremes[1, 2], center1[0], center1[1], center1[2])
    plt.plot([extremes[1,0], center1[0]], [extremes[1,1], center1[1]],[extremes[1,2],center1[2]], color='gray', linestyle='--', linewidth=1)
    add_perpendicular_crosses_3d(ax, extremes[2, 0], extremes[2, 1], extremes[2, 2], center2[0], center2[1], center2[2])
    plt.plot([extremes[2,0], center2[0]], [extremes[2,1], center2[1]],[extremes[2,2],center2[2]], color='gray', linestyle='--', linewidth=1)
    add_perpendicular_crosses_3d(ax, extremes[3, 0], extremes[3, 1], extremes[3, 2], center3[0], center3[1], center3[2])
    plt.plot([extremes[3,0], center3[0]], [extremes[3,1], center3[1]],[extremes[3,2],center3[2]], color='gray', linestyle='--', linewidth=1)

    legend_text = []
    for index in range(len(subsites)):
        legend_text.append(subsites[index] + ', ' + str(odered_value_counts[index]) + ' patients')

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=text)
                    for color, text in zip(colors_ordered, legend_text)]

    # Add a legend with fixed dot sizes
    # plt.legend(handles=legend_elements, loc='upper right', title='Subsites', fontsize='small')

    plt.gca().set_axis_off()
    plt.show()



def main(args: argparse.Namespace):
    mixture_df = pd.read_csv(args.input)
    nr_components = len(mixture_df)
    params = load_yaml_params(args.params)
    data = params['general']['data']
    if nr_components == 2:
        plot_2d_simplex()
    elif nr_components == 3:
        plot_3d_simplex(mixture_df, data)
    else:
        logger.info(f"Simplex not supported for {nr_components} components")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)

