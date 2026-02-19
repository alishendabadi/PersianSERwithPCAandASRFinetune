import matplotlib.pyplot as plt
import seaborn as sns

def draw_line_plot(values_list:list, labels_list:list, x_label, y_label):
    for (values, label) in zip(values_list, labels_list):
      plt.plot(values, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()