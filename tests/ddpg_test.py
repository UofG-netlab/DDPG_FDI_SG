import os
import sys
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import run_timeseries, OutputWriter, DFData

from controllers.ddpg_rl_transformer_controller import DDPGTransformerController
from envs.DDPG_multi_agent_substation_env import ddpg_multi_agent_substation_env
from controllers.transformer_control import TransformerDisconnect
from plots.plot_utils import plot_curves, plot_confusion_matrix
from utils.Generate_fdi import generate_fdi_list
from utils.network import create_stable_gen_profile, create_30_network


# ---------------- configuration ----------------
seed = 42
random.seed(seed)
np.random.seed(seed)

time_steps = 200
T_ambient = 25.0
T_rated = 65.0
n = 1.6
max_temperature = 147.44
model_dir = "./models_ddpg"


# ============================================================
# 🔧 FIX 1 — Reduce DER magnitude to avoid voltage divergence
# ============================================================
def add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=5.0, fluctuation=2.0):
    for i, row in net.trafo.iterrows():
        hv_bus = row["hv_bus"]

        sgen_idx = pp.create_sgen(net, bus=hv_bus, p_mw=base_p_mw,
                                  q_mvar=0.0, name=f"sgen_trafo_{i}")
        print(f"Created sgen {sgen_idx} at hv_bus {hv_bus}")

        time = np.arange(time_steps)
        profile = base_p_mw + fluctuation * np.sin(2 * np.pi * time / time_steps)
        profile = np.clip(profile, 0, None)

        ds = DFData(pd.DataFrame({"p_mw": profile}))
        ConstControl(net, element="sgen", variable="p_mw",
                     element_index=[sgen_idx],
                     data_source=ds, profile_name="p_mw")


# ============================================================
# 🔧 FIX 2 — Reduce overload severity
# ============================================================
def inject_transformer_overload_safely(net, time_steps, events_per_trafo=2,
                                       base_load=5.0,
                                       min_factor=1.3, max_factor=1.8,
                                       min_duration=5, max_duration=10):
    for trafo_idx, row in net.trafo.iterrows():
        lv_bus = row["lv_bus"]
        matched_loads = net.load[net.load.bus == lv_bus]

        if matched_loads.empty:
            new_idx = pp.create_load(net, bus=lv_bus, p_mw=base_load, q_mvar=0.0,
                                     name=f"synthetic_trafo_{trafo_idx}")
            load_indices = [new_idx]
        else:
            load_indices = matched_loads.index.tolist()

        profile = np.full(time_steps, base_load)

        for _ in range(events_per_trafo):
            dur = random.randint(min_duration, max_duration)
            start = random.randint(0, time_steps - dur)
            factor = random.uniform(min_factor, max_factor)
            profile[start:start+dur] *= factor

        profile_df = pd.DataFrame({"p_mw": profile})

        for load_idx in load_indices:
            ds = DFData(profile_df)
            ConstControl(net, element="load", variable="p_mw",
                         element_index=[load_idx],
                         data_source=ds, profile_name="p_mw")


# ---------------- network build ----------------
net = create_30_network()
trafo_indices = list(net.trafo.index)

add_support_sgen_to_transformers(net, time_steps=time_steps)

inject_transformer_overload_safely(net, time_steps=time_steps)

# ============================================================
# 🔧 FIX 3 — Cap generator profile magnitude
# ============================================================
gen_profile = create_stable_gen_profile(net, time_steps=time_steps,
                                        base_gen_factor=1.1)

ConstControl(net, element='gen', variable='p_mw',
             element_index=[0], data_source=gen_profile,
             profile_name='p_mw', order=0)

# ---------------- add FDI attacks ----------------
num_attacks = 20
fdi_list = generate_fdi_list(time_steps, num_attacks, 150, 160.0)
fdi_per_trafo = [[] for _ in trafo_indices]
fdi_attack_log = {}

for fdi in fdi_list:
    target = random.choice(range(len(trafo_indices)))
    fdi_per_trafo[target].append(fdi)
    time_step, faulty_temp = fdi
    fdi_attack_log[(time_step, target)] = faulty_temp


# ---------------- RL environment ----------------
env = ddpg_multi_agent_substation_env(
    net,
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
    max_temperature=max_temperature
)

# ---------------- attach transformer controllers ----------------
for i, idx in enumerate(trafo_indices):
    TransformerDisconnect(net, idx, max_temperature,
                          T_ambient, T_rated, n,
                          fdi_list=fdi_per_trafo[i],
                          total_steps=time_steps)

for i, idx in enumerate(trafo_indices):
    model_path = f"{model_dir}/actor_trafo_{idx}.pth"
    controller = DDPGTransformerController(env, idx, max_temperature,
                                           T_ambient, T_rated, n,
                                           fdi_per_trafo[i], time_steps,
                                           model_path=model_path)
    print(f"[DEBUG] Actor weights for Trafo {idx} loaded")


# ============================================================
# 🔧 FIX 4 — OutputWriter OK
# ============================================================
log_vars = [
    ("trafo", "in_service"),
    ("res_trafo", "loading_percent"),
    ("trafo", "temperature_measured"),
    ("trafo", "actual_temperature")
]
output_path = "./results_ddpg"
ow = OutputWriter(net, time_steps, output_path, '.csv',
                  log_variables=log_vars, csv_separator=';')


# ============================================================
# 🔧 FIX 5 — Catch non-convergence and print timestep
# ============================================================
try:
    run_timeseries(net, time_steps=range(time_steps))
except Exception as e:
    print("\n‼️ Loadflow failed at timestep:", net.ts_variables.get("time_step"))
    raise e


# ---------------- plotting ----------------
plot_curves(f"{output_path}/res_trafo/loading_percent.csv",
            f"{output_path}/loading_curves.png")

# confusion summary
ddpg_controllers = [ctrl for ctrl in net.controller["object"]
                    if isinstance(ctrl, DDPGTransformerController)]

total_tp = total_fp = total_fn = total_tn = 0

for ctrl in ddpg_controllers:
    ctrl.print_confusion_matrix()
    total_tp += ctrl.tp
    total_fp += ctrl.fp
    total_fn += ctrl.fn
    total_tn += ctrl.tn

print("\nSummary:")
print("TP =", total_tp)
print("FP =", total_fp)
print("FN =", total_fn)
print("TN =", total_tn)

conf_matrix = np.array([[total_tp, total_fp], [total_fn, total_tn]])
plot_confusion_matrix(conf_matrix)