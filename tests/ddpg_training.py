import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandapower.control import ConstControl
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter, DFData

from controllers.transformer_control import TransformerDisconnect
from models.DDPG import MultiAgentDDPGTrainer

from utils.Generate_fdi import generate_fdi_list
from utils.network import create_network, create_load_profile, create_stable_gen_profile, create_30_network

from controllers.ddpg_multi_agent_controller import DDPGMultiAgentController
from envs.DDPG_multi_agent_substation_env import ddpg_multi_agent_substation_env

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

total_episodes = 15
time_steps = 200
T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 147.44
# Generate FDI lists for each transformer
num_attacks = 20
min_faulty_data = 150.0
max_faulty_data = 160.0
log_vars = [
    ("trafo", "in_service"),
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured"),
    ("trafo", "actual_temperature")
]

metrics_log = {
    "episode": [],
    "TP": [],
    "FP": [],
    "FN": [],
    "TN": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "accuracy": []
}


def record_and_plot_metrics(episode, TP, FP, FN, TN, save_path="metrics_over_episodes.png"):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    metrics_log["episode"].append(episode)
    metrics_log["TP"].append(TP)
    metrics_log["FP"].append(FP)
    metrics_log["FN"].append(FN)
    metrics_log["TN"].append(TN)
    metrics_log["precision"].append(precision)
    metrics_log["recall"].append(recall)
    metrics_log["f1"].append(f1)
    metrics_log["accuracy"].append(accuracy)

    # plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["episode"], metrics_log["precision"], label='Precision', marker='o')
    plt.plot(metrics_log["episode"], metrics_log["recall"], label='Recall', marker='s')
    plt.plot(metrics_log["episode"], metrics_log["f1"], label='F1 Score', marker='^')
    plt.plot(metrics_log["episode"], metrics_log["accuracy"], label='Accuracy', marker='d')

    plt.xlabel("Episode")
    plt.ylabel("Metric")
    plt.title("Evaluation Metrics Over Episodes")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=80.0, fluctuation=20.0):

    for i, row in net.trafo.iterrows():
        hv_bus = row["hv_bus"]

        # create an sgen with base active power
        sgen_idx = pp.create_sgen(net, bus=hv_bus, p_mw=base_p_mw, q_mvar=0.0, name=f"sgen_trafo_{i}")
        print(f"⚡ Created sgen {sgen_idx} at hv_bus {hv_bus} for Trafo {i}")

        # create time-varying profile: base ± fluctuation * sin
        time = np.arange(time_steps)
        profile = base_p_mw + fluctuation * np.sin(2 * np.pi * time / time_steps)
        profile = np.clip(profile, 0, None)  # keep non-negative

        profile_df = pd.DataFrame({"p_mw": profile})
        ds = DFData(profile_df)

        ConstControl(net, element="sgen", variable="p_mw",
                     element_index=[sgen_idx],
                     data_source=ds,
                     profile_name="p_mw")
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

def build_net(time_steps=100, max_temperature=max_temperature):
    # This method is for initialization of the net with every controller.
    net = create_30_network()
    trafo_indices = list(net.trafo.index)

    add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=30.0, fluctuation=10.0)

    inject_transformer_overload_safely(net, time_steps=time_steps)


    gen_profile = create_stable_gen_profile(net, time_steps=time_steps, base_gen_factor=1.8)
    ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=gen_profile, profile_name='p_mw', order = 0)

    fdi_list = generate_fdi_list(time_steps, num_attacks, min_faulty_data, max_faulty_data)

    fdi_per_trafo = [[] for _ in trafo_indices]
    fdi_attack_log = {}  # Log to track FDI attacks for later analysis

    # Distribute FDI attacks across transformers
    for fdi in fdi_list:
        target_trafo_index = random.choice(range(len(trafo_indices)))
        fdi_per_trafo[target_trafo_index].append(fdi)
        # Log each FDI attack: {time_step: (trafo_index, faulty_temperature)}
        time_step, faulty_temperature = fdi
        fdi_attack_log[(time_step, target_trafo_index)] = faulty_temperature

    # Print the FDI attack log to review all attacks before training
    for (time_step, trafo_index), faulty_temperature in sorted(fdi_attack_log.items()):
        print(f"Time step {time_step}, Transformer {trafo_index}: Faulty temperature = {faulty_temperature}°C")

    # Create the environment
    env = ddpg_multi_agent_substation_env(net,
                                     trafo_indices=trafo_indices,
                                     delta_p=0.05,
                                     initial_p=0.0,
                                     voltage_tolerance=0.05,
                                     voltage_penalty_factor=10.0,
                                     line_loading_limit=1.0,
                                     power_flow_penalty_factor=5.0,
                                     load_reward_factor=100.0,
                                     transformer_reward_factor=50.0,
                                     disconnection_penalty_factor=50.0,
                                     total_steps=time_steps,
                                     max_temperature= max_temperature)

    # Add the Transformer Protection Controllers with specific FDI lists
    for i, index in enumerate(trafo_indices):
        TransformerDisconnect(
            net=env.net,
            trafo_index=index,
            max_temperature=max_temperature,  # Set the maximum allowable temperature
            T_ambient=T_ambient,  # Ambient temperature
            T_rated=T_rated,  # Rated temperature rise
            n=n,  # Exponent for temperature calculation
            fdi_list=fdi_per_trafo[i],  # Apply specific FDI list for each transformer
            total_steps=time_steps,
            order=1
        )

    ow = OutputWriter(
        env.net, time_steps=time_steps, output_path="./output_data",
        output_file_type='.csv', log_variables=log_vars, csv_separator=';'
    )

    return env, trafo_indices

env, trafo_indices = build_net(time_steps, max_temperature)
trainer = MultiAgentDDPGTrainer(env.get_state_size(), 1, trafo_indices)

for episode in range(total_episodes):
    print(f"\n Episode {episode + 1}/{total_episodes} started.")
    env, _ = build_net(time_steps=time_steps, max_temperature=max_temperature)
    RLController = DDPGMultiAgentController(env=env, trainer=trainer, trafo_indices=trafo_indices, order=2)
    try:

        run_timeseries(env.net, range(time_steps))
    except Exception as e:
        print(f"⚠️ Episode {episode + 1} failed due to error: {e}")
        continue
    finally:
        print(f"✅ Finished Episode {episode + 1}/{total_episodes}")
        print(f"\n[Total Confusion Matrix Stats]")
        print(f"TP: {RLController.tp}")
        print(f"FN: {RLController.fn}")
        print(f"FP: {RLController.fp}")
        print(f"TN: {RLController.tn}")

        record_and_plot_metrics(
            episode=episode + 1,
            TP=RLController.tp,
            FP=RLController.fp,
            FN=RLController.fn,
            TN=RLController.tn
        )

trainer.save_all_models()
trainer.plot_loss()

