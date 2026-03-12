import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
from models.LSTM_DDPG import MultiAgentLSTMDDPGTrainer
from utils.Generate_fdi import generate_fdi_list
from utils.network import create_30_network, create_stable_gen_profile
from controllers.LSTM_ddpg_multi_agent_controller import LSTM_DDPGMultiAgentController
from envs.LSTM_multi_agent_substation_env import LSTM_MultiAgentEnv

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

total_episodes = 20
time_steps = 200
T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 147.44
num_attacks = 20
min_faulty_data = 150.0
max_faulty_data = 160.0

log_vars = [
    ("trafo", "in_service"),
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured"),
    ("trafo", "actual_temperature")
]

metrics_log = {key: [] for key in ["episode", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]}


def plot_reward_trends(reward_history):
    plt.figure(figsize=(12, 6))
    for idx, rewards in reward_history.items():
        if len(rewards) < 5:
            continue
        rewards = np.array(rewards)
        smooth = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        plt.plot(smooth, label=f"Trafo {idx}")
    plt.axhline(y=0, linestyle="--", color="gray", linewidth=1)
    plt.title("Reward Trend per Transformer")
    plt.xlabel("Time Step")
    plt.ylabel("Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=80.0, fluctuation=20.0):

    for i, row in net.trafo.iterrows():
        hv_bus = row["hv_bus"]

        sgen_idx = pp.create_sgen(net, bus=hv_bus, p_mw=base_p_mw, q_mvar=0.0, name=f"sgen_trafo_{i}")
        print(f"⚡ Created sgen {sgen_idx} at hv_bus {hv_bus} for Trafo {i}")

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
            print(f"Injected profile to Load {load_idx} for Trafo {trafo_idx}")

def build_net(time_steps=100, max_temperature=max_temperature):
    net = create_30_network()
    trafo_indices = list(net.trafo.index)
    add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=30.0, fluctuation=10.0)

    inject_transformer_overload_safely(net, time_steps=time_steps)

    gen_profile = create_stable_gen_profile(net, time_steps=time_steps, base_gen_factor=1.8)
    ConstControl(net, element='gen', variable='p_mw', element_index=[0], data_source=gen_profile, profile_name='p_mw', order=0)

    fdi_list = generate_fdi_list(time_steps, num_attacks, min_faulty_data, max_faulty_data)

    fdi_per_trafo = [[] for _ in trafo_indices]
    fdi_attack_log = {}
    for fdi in fdi_list:
        target_trafo_index = random.choice(range(len(trafo_indices)))
        fdi_per_trafo[target_trafo_index].append(fdi)
        # Log each FDI attack: {time_step: (trafo_index, faulty_temperature)}
        time_step, faulty_temperature = fdi
        fdi_attack_log[(time_step, target_trafo_index)] = faulty_temperature

        # Print the FDI attack log to review all attacks before training
    for (time_step, trafo_index), faulty_temperature in sorted(fdi_attack_log.items()):
        print(f"Time step {time_step}, Transformer {trafo_index}: Faulty temperature = {faulty_temperature}°C")
    env = LSTM_MultiAgentEnv(net, trafo_indices=trafo_indices, seq_len=3, total_steps=time_steps, max_temperature=max_temperature)

    for i, index in enumerate(trafo_indices):
        TransformerDisconnect(net=env.net, trafo_index=index, max_temperature=max_temperature, T_ambient=T_ambient, T_rated=T_rated, n=n, fdi_list=fdi_per_trafo[i], total_steps=time_steps, order=1)
    OutputWriter(env.net, time_steps=time_steps, output_path="./output_data", output_file_type='.csv', log_variables=log_vars, csv_separator=';')
    return env, trafo_indices


env, trafo_indices = build_net(time_steps, max_temperature)
trainer = MultiAgentLSTMDDPGTrainer(env.get_state_size(), 1, trafo_indices)
reward_history = {idx: [] for idx in trafo_indices}
for episode in range(total_episodes):
    print(f"\n Episode {episode + 1}/{total_episodes} started.")
    env, trafo_indices = build_net(time_steps, max_temperature)
    RLController = LSTM_DDPGMultiAgentController(env=env, trainer=trainer, trafo_indices=trafo_indices, order=2)

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

        for idx in trafo_indices:
            reward_history[idx] += RLController.reward_history[idx]

trainer.save_all_models()
trainer.plot_loss()
plot_reward_trends(reward_history)
