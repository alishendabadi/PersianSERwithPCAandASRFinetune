import matplotlib.pyplot as plt
import seaborn as sns


def draw_line_plot(values_list: list, labels_list: list, x_label: str, y_label: str):
    """Create a line plot and return the Figure for TensorBoard logging."""
    fig, ax = plt.subplots()
    for values, label in zip(values_list, labels_list):
        ax.plot(values, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    return fig
