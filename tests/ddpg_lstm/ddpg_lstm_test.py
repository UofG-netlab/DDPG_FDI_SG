import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
import pandas as pd
import numpy as np
from pandapower.control import ConstControl
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter, DFData

from controllers.LSTM_ddpg_rl_transformer_controller import LSTMTransformerController
from envs.LSTM_multi_agent_substation_env import LSTM_MultiAgentEnv
from controllers.transformer_control import TransformerDisconnect
from plots.plot_utils import plot_curves, plot_confusion_matrix
from utils.Generate_fdi import generate_fdi_list
from utils.network import create_stable_gen_profile, create_30_network
import matplotlib.pyplot as plt
import os

# ---------------- configuration ----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
time_steps = 200
T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 147.44
model_dir = "./models_lstm_ddpg"
seq_len = 5

save_path = "./temp_histograms"
os.makedirs(save_path, exist_ok=True)

def compute_stability_metrics(env, trafo_idx):
    voltages = []
    temps = []
    unnecessary_disc = 0

    for t in range(env.total_steps):
        v = float(env.net.res_bus.vm_pu.mean())
        voltages.append(v)

        T = float(env.net.trafo.at[trafo_idx, "temperature_measured"])
        temps.append(T)

        in_service = bool(env.net.trafo.at[trafo_idx, "in_service"])
        should_disconnect = T > env.max_temperature

        if not should_disconnect and not in_service:
            unnecessary_disc += 1

    # Voltage stability
    VSS = 1 - np.mean(np.abs(np.array(voltages) - 1.0))
    VSS = max(0, min(1, VSS))

    # Thermal variance stability
    temp_var = np.var(temps)
    TSV = np.exp(-temp_var / 50.0)   # scaling factor

    # Disconnection stability
    DS = 1 - unnecessary_disc / env.total_steps
    DS = max(0, DS)

    # Combined
    SI = 0.4 * VSS + 0.3 * TSV + 0.3 * DS

    return VSS, TSV, DS, SI
def compute_stability_metricnot(env, controllers, max_temperature):
    unnecessary_switches = 0
    total_actions = 0

    for ctrl in controllers:
        idx = ctrl.trafo_index
        prev_state = None

        for t in range(env.total_steps):
            in_service = bool(env.net.trafo.at[idx, "in_service"])
            temp = float(env.net.trafo.at[idx, "actual_temperature"])

            if prev_state is not None:
                switched = (prev_state != in_service)

                if switched:
                    total_actions += 1
                    if temp < 0.9 * max_temperature:
                        unnecessary_switches += 1

            prev_state = in_service

    if total_actions == 0:
        return 1.0  # fully stable

    return 1 - (unnecessary_switches / total_actions)

def plot_fdi_defense_matrix(env, controller_list, fdi_attack_log):
    time_steps = env.total_steps
    trafo_indices = env.trafo_indices

    x, y = [], []
    colors = []
    markers = []

    for ctrl in controller_list:
        idx = ctrl.trafo_index
        for t in range(time_steps):
            if (t, idx) in fdi_attack_log:
                x.append(t)
                y.append(idx)

                in_service = bool(env.net.trafo.at[idx, "in_service"])
                try:
                    actual_temp = float(env.net.trafo.at[idx, "actual_temperature"])
                except:
                    actual_temp = 25.0  # fallback

                if not in_service:
                    if actual_temp > env.max_temperature:
                        # transformer
                        colors.append("green")
                        markers.append("o")
                    else:
                        colors.append("red")
                        markers.append("x")
                else:
                    colors.append("green")
                    markers.append("o")

    plt.figure(figsize=(12, 6))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=80)

    plt.xlabel("Timestep")
    plt.ylabel("Transformer Index")
    plt.title("FDI Attack Handling Over Time")
    plt.grid(True)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='✓ Properly Handled', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='x', color='red', label='✗ Misled by FDI', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_temperature_comparison(trafo_index, time_steps, output_path, fdi_log):
    measured_path = os.path.join(output_path, "trafo", "temperature_measured.csv")
    actual_path = os.path.join(output_path, "trafo", "actual_temperature.csv")

    # load data
    measured_df = pd.read_csv(measured_path, sep=';', index_col=0)
    actual_df = pd.read_csv(actual_path, sep=';', index_col=0)

    measured = measured_df.iloc[:, trafo_index]
    actual = actual_df.iloc[:, trafo_index]

    # collect all FDI injection timesteps for this transformer
    fdi_steps = [step for (step, idx) in fdi_log.keys() if idx == trafo_index]

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Temperature", linewidth=2)
    plt.plot(measured, label="Measured Temperature", linestyle='--', color="orange")

    # mark FDI injections
    plt.scatter(fdi_steps, measured.iloc[fdi_steps], color="red", marker='x', s=80, label="FDI Injected")

    plt.title(f"Transformer {trafo_index}: Actual vs Measured Temperature")
    plt.xlabel("Timestep")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/trafo_temp_compare_{trafo_index}.png")
    plt.show()



def debug_action_statistics(controller_list):
    all_actions = []
    for ctrl in controller_list:
        if hasattr(ctrl.env, 'p'):
            p_vals = ctrl.env.p.get(ctrl.trafo_index, None)
            if isinstance(p_vals, (int, float)):
                all_actions.append(p_vals)
    if all_actions:
        plt.hist(all_actions, bins=20, range=(0, 1))
        plt.title("Action Distribution Across Transformers")
        plt.xlabel("Action Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
    else:
        print("No valid action data found to debug.")

def add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=80.0, fluctuation=20.0):
    for i, row in net.trafo.iterrows():
        hv_bus = row["hv_bus"]
        sgen_idx = pp.create_sgen(net, bus=hv_bus, p_mw=base_p_mw, q_mvar=0.0, name=f"sgen_trafo_{i}")
        print(f"⚡ Created sgen {sgen_idx} at hv_bus {hv_bus} for Trafo {i}")
        time = np.arange(time_steps)
        profile = base_p_mw + fluctuation * np.sin(2 * np.pi * time / time_steps)
        profile = np.clip(profile, 0, None)
        profile_df = pd.DataFrame({"p_mw": profile})
        ds = DFData(profile_df)
        ConstControl(net, element="sgen", variable="p_mw", element_index=[sgen_idx], data_source=ds, profile_name="p_mw")
        print(f"✅ Attached time-varying control to sgen {sgen_idx}")

def inject_transformer_overload_safely(net, time_steps, events_per_trafo=3,
                                       base_load=10.0,
                                       min_factor=2.0, max_factor=3.0,
                                       min_duration=10, max_duration=30):
    for trafo_idx, row in net.trafo.iterrows():
        lv_bus = row["lv_bus"]
        matched_loads = net.load[net.load.bus == lv_bus]
        if matched_loads.empty:
            new_idx = pp.create_load(net, bus=lv_bus, p_mw=base_load, q_mvar=0.0, name=f"synthetic_trafo_{trafo_idx}")
            load_indices = [new_idx]
            print(f"➕ Created synthetic load {new_idx} at lv_bus {lv_bus} for Trafo {trafo_idx}")
        else:
            load_indices = matched_loads.index.tolist()
        profile = np.full(time_steps, base_load)
        for _ in range(events_per_trafo):
            dur = random.randint(min_duration, max_duration)
            start = random.randint(0, time_steps - dur)
            factor = random.uniform(min_factor, max_factor)
            profile[start:start+dur] *= factor
            print(f"🔥 Trafo {trafo_idx} overload: t={start}-{start+dur}, factor={factor:.2f}")
        profile_df = pd.DataFrame({"p_mw": profile})
        for load_idx in load_indices:
            ds = pp.timeseries.DFData(profile_df)
            pp.control.ConstControl(
                net, element="load", variable="p_mw",
                element_index=[load_idx],
                data_source=ds,
                profile_name="p_mw"
            )
            print(f"✅ Injected profile to Load {load_idx} for Trafo {trafo_idx}")

# ---------------- network build ----------------
net = create_30_network()
trafo_indices = list(net.trafo.index)

add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=30.0, fluctuation=10.0)

inject_transformer_overload_safely(net, time_steps=time_steps)

gen_profile = create_stable_gen_profile(net, time_steps=time_steps, base_gen_factor=1.5)
ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=gen_profile, profile_name='p_mw', order=0)

num_attacks = 50
fdi_list = generate_fdi_list(time_steps, num_attacks, 160, 170.0)
fdi_per_trafo = [[] for _ in trafo_indices]
for fdi in fdi_list:
    target = random.choice(range(len(trafo_indices)))
    fdi_per_trafo[target].append(fdi)

env = LSTM_MultiAgentEnv(
    net=net,
    trafo_indices=trafo_indices,
    seq_len=seq_len,
    total_steps=time_steps,
    max_temperature=max_temperature
)

for i, idx in enumerate(trafo_indices):
    TransformerDisconnect(
        net=net,
        trafo_index=idx,
        max_temperature=max_temperature,
        T_ambient=T_ambient,
        T_rated=T_rated,
        n=n,
        fdi_list=fdi_per_trafo[i],
        total_steps=time_steps
    )

for i, idx in enumerate(trafo_indices):
    model_path = f"{model_dir}/actor_trafo_{idx}.pth"
    controller = LSTMTransformerController(
        env=env,
        trafo_index=idx,
        seq_len=seq_len,
        max_temperature=max_temperature,
        T_ambient=T_ambient,
        T_rated=T_rated,
        n=n,
        fdi_list=fdi_per_trafo[i],
        total_steps=time_steps,
        model_path=model_path
    )
    print(f"[DEBUG] Loaded LSTM weights for Trafo {idx}:", controller.actor.state_dict())

log_vars = [
    ("trafo", "in_service"),
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured"),
    ("trafo", "actual_temperature")
]
output_path = "./results_lstm_ddpg"
ow = OutputWriter(net, time_steps, output_path, '.csv', log_variables=log_vars, csv_separator=';')

run_timeseries(net, time_steps=range(time_steps))

plot_curves(f"{output_path}/res_trafo/loading_percent.csv", f"{output_path}/loading_curves.png")

lstm_controllers = [ctrl for ctrl in net.controller["object"] if isinstance(ctrl, LSTMTransformerController)]

total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
print("\n**Transformer-Level Confusion Matrix Statistics (LSTM-DDPG)**")
for controller in lstm_controllers:
    controller.print_confusion_matrix()
    total_tp += controller.tp
    total_fp += controller.fp
    total_fn += controller.fn
    total_tn += controller.tn

print("Confusion Matrix:")
print(f"✅ True Positive (TP) should disconnect, disconnected = {total_tp}")
print(f"❌ False Positive (FP) should stay, disconnected = {total_fp}")
print(f"❌ False Negative (FN) should disconnect, stayed connected = {total_fn}")
print(f"✅ True Negative (TN) should stay, stayed connected = {total_tn}")



def visualize_temperature_defense(env, controller_list, fdi_attack_log, save_path=None):
    time_steps = env.total_steps
    trafo_indices = env.trafo_indices

    for t in range(time_steps):
        temps = []
        colors = []
        annotations = []

        for ctrl in controller_list:
            idx = ctrl.trafo_index
            temp = env.net.trafo.at[idx, "temperature_measured"]
            temps.append(temp)
            fdi_flag = (t, idx) in fdi_attack_log
            in_service = bool(env.net.trafo.at[idx, "in_service"])

            if fdi_flag:
                if in_service:
                    colors.append("red")
                    annotations.append("✓")
                else:
                    colors.append("red")
                    annotations.append("×")
            else:
                colors.append("gray")
                annotations.append("")

        plt.figure(figsize=(10, 4))
        bars = plt.bar(range(len(temps)), temps, color=colors)
        for i, ann in enumerate(annotations):
            if ann:
                plt.text(i, temps[i] + 1.5, ann, ha='center', color='green' if ann == '✓' else 'black', fontsize=14)

        plt.title(f"Temperature Histogram at t = {t}")
        plt.xlabel("Transformer Index")
        plt.ylabel("Temperature (°C")
        plt.ylim(0, max(temps) + 10)
        plt.grid(True)

        if save_path:
            plt.savefig(f"{save_path}/temp_hist_t{t:03d}.png")
            plt.close()
        else:
            plt.show()

fdi_attack_log = {(step, idx): temp for idx, lst in enumerate(fdi_per_trafo) for step, temp in lst}

plot_fdi_defense_matrix(env, lstm_controllers, fdi_attack_log)

conf_matrix = np.array([[total_tp, total_fp], [total_fn, total_tn]])
plot_confusion_matrix(conf_matrix)
debug_action_statistics(lstm_controllers)
def plot_stability_bar(results_dict):
    # results_dict = {"DQN": val, "DDPG": val, "LSTM-DDPG": val}
    models = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure(figsize=(7, 5))

    gray_colors = ["0.2", "0.5", "0.8"]  # dark → light grayscale

    plt.bar(models, values, color=gray_colors, width=0.6)
    plt.title("Model Stability Comparison (Higher is Better)")
    plt.ylabel("Stability Metric (0–1)")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig("stability_comparison.png", dpi=300)
    plt.show()

for ctrl in lstm_controllers:
    VSS, TSV, DS, SI = compute_stability_metrics(env, ctrl.trafo_index)
    print(f"Transformer {ctrl.trafo_index}: SI={SI:.4f}")

system_vss = []
system_tsv = []
system_ds  = []
system_si  = []

for ctrl in lstm_controllers:
    VSS, TSV, DS, SI = compute_stability_metrics(env, ctrl.trafo_index)
    system_vss.append(VSS)
    system_tsv.append(TSV)
    system_ds.append(DS)
    system_si.append(SI)

print("\n=== System Stability Metrics ===")
print("VSS_mean =", np.mean(system_vss))
print("TSV_mean =", np.mean(system_tsv))
print("DS_mean  =", np.mean(system_ds))
print("SI_mean  =", np.mean(system_si))

VSS, TSV, DS, SI = compute_stability_metrics(env, ctrl.trafo_index)
#stability_score = compute_stability_metrics(env, lstm_controllers)
print(f"\n🔒 Stability Metric (LSTM-DDPG): {stability_score:.4f}")
print('score',VSS, TSV, DS, SI )
# Example if you want to compare with other models:
#results = {
#    "DQN": 0.58,
#    "DDPG": 0.61,
#    "LSTM-DDPG": stability_score
#}
#plot_stability_bar(results)

plot_temperature_comparison(trafo_index=1, time_steps=time_steps, output_path=output_path, fdi_log=fdi_attack_log)

