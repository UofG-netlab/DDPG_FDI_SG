import pandapower as pp
import pandapower.networks as pn
from pandapower.timeseries import run_timeseries, OutputWriter, DFData
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

net = pn.case_ieee30()
# create load time series (periodic)
time_steps = 200
load_profile = 50 + 10 * np.sin(np.linspace(0, 2*np.pi, time_steps))
load_df = pd.DataFrame({"p_mw": load_profile})

ds = DFData(load_df)

load_idx = net.load.index[0]

pp.control.ConstControl(net, element="load", variable="p_mw",
                        element_index=[load_idx],
                        profile_name="p_mw", data_source=ds)


def run_timeseries_with_logging(net, time_steps=200, output_path="./output_data", suffix="default"):
    # define fields to log
    log_vars = [
        ("trafo", "in_service"),
        ("res_trafo", "loading_percent"),
        ("res_bus", "vm_pu"),
        ("res_line", "loading_percent"),
        ("load", "p_mw")
    ]

    # create output path
    os.makedirs(output_path, exist_ok=True)

    # register OutputWriter (each field writes to CSV)
    OutputWriter(net, time_steps=time_steps, output_path=output_path,
                 output_file_type=".csv", log_variables=log_vars,
                 csv_separator=";")

    # run time-series simulation
    run_timeseries(net, time_steps)


run_timeseries_with_logging(net, time_steps=200, output_path="./output_data", suffix="case1")


def plot_transformer_dynamics(output_dir="./output_data", trafo_index=0):
    # build paths
    trafo_dir = os.path.join(output_dir, "trafo")
    res_trafo_dir = os.path.join(output_dir, "res_trafo")
    res_line_dir = os.path.join(output_dir, "res_line")
    res_bus_dir = os.path.join(output_dir, "res_bus")
    load_dir = os.path.join(output_dir, "load")

    in_service = pd.read_csv(os.path.join(trafo_dir, "in_service.csv"), sep=";")
    trafo_loading = pd.read_csv(os.path.join(res_trafo_dir, "loading_percent.csv"), sep=";")
    line_loading = pd.read_csv(os.path.join(res_line_dir, "loading_percent.csv"), sep=";")
    bus_voltage = pd.read_csv(os.path.join(res_bus_dir, "vm_pu.csv"), sep=";")
    load_power = pd.read_csv(os.path.join(load_dir, "p_mw.csv"), sep=";")

    time = in_service.index

    plt.figure(figsize=(16, 10))

    # temperature and loading
    plt.subplot(3, 1, 1)
    plt.plot(time, trafo_loading[str(trafo_index)], label="Trafo Loading (%)")
    plt.title(f"Transformer {trafo_index} & Loading")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # in-service status
    plt.subplot(3, 1, 2)
    plt.plot(time, in_service[str(trafo_index)] * 150, label="In Service (scaled)")
    plt.ylim(-10, 160)
    plt.title("In Service Status (Disconnected or Not)")
    plt.grid(True)
    plt.legend()

    # voltage, line loading, and load changes
    plt.subplot(3, 1, 3)
    plt.plot(time, bus_voltage[str(trafo_index)], label="Bus Voltage (pu)")
    plt.plot(time, line_loading[str(trafo_index)], label="Line Loading (%)")
    plt.plot(time, load_power[str(trafo_index)], label="Load p_mw")
    plt.title("Voltage, Current, and Load Changes")
    plt.xlabel("Time Step")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_transformer_dynamics(output_dir="./output_data", trafo_index=0)


