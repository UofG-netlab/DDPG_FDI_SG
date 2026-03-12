import pandas as pd
from pandapower.plotting.plotly import vlevel_plotly
from utils.network import create_network, create_30_network
import os
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns


def plot_network(path = "pics/voltage_level_plot.png"):
    fig = vlevel_plotly(create_30_network())
    fig.write_image(path)

# plot_network()

def plot_curves(file_path, output_path=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found!!!!!!")

    trafo_results = pd.read_csv(file_path, sep=";")
    time_steps = trafo_results.index

    plt.figure(figsize=(10, 5))
    for trafo_id in trafo_results.columns[1:]:
        plt.plot(time_steps, trafo_results[trafo_id], label=f"Transformer {trafo_id}")

    plt.xlabel("Time Step")
    plt.ylabel("Transformer Loading (%)")
    plt.title("Transformer Loading Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Save the photo to the path: {output_path}")
    else:
        plt.show()

def plot_loss(controller):
    plt.figure(figsize=(10, 5))
    plt.plot(controller.loss_history, label="Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss Value")
    plt.title("DQN Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted FDI", "Predicted Normal"],
                yticklabels=["Actual FDI", "Actual Normal"])

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix for Transformer FDI Detection")

    plt.show()

def plot_temperature(file_path, output_path = "/Users/joshua/PandaPower/plots/pics/curves.png"):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found!!!!!!")
    trafo_results = pd.read_csv(file_path, sep = ";")
    time_steps = trafo_results.index

    plt.figure(figsize = (10, 5))
    for trafo_id in trafo_results.columns[1:]:
        plt.plot(time_steps, trafo_results[trafo_id], label = f"Transformer {trafo_id}")
    plt.xlabel("Time Step")
    plt.ylabel("Transformer Temperature")
    plt.title("Time Step")

    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Save the photo to the path: {output_path}")



def plot_service(file_path, output_path=None):
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10.colors
    in_service_df = pd.read_csv(file_path, sep=";", index_col=0)

    for i, col in enumerate(in_service_df.columns):
        y_offset = len(in_service_df.columns) - i
        status = in_service_df[col].values
        on_steps = in_service_df.index[status == 1]
        if len(on_steps) > 0:
            plt.hlines(y=y_offset, xmin=on_steps[0], xmax=on_steps[-1], color='lightgray', linewidth=2)
            for t in on_steps:
                plt.vlines(x=t, ymin=y_offset - 0.4, ymax=y_offset + 0.4, color=colors[i % len(colors)], linewidth=2)

    plt.yticks(range(1, len(in_service_df.columns) + 1), reversed(in_service_df.columns))
    plt.xlabel("Time Step")
    plt.title("Transformer In-Service Timeline (Colored)")
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Save the photo to the path: {output_path}")
    else:
        plt.show()