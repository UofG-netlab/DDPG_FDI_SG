import pandapower.networks as pn
import numpy as np
import pandapower as pp
from pandapower.control import ConstControl
import pandas as pd
import random

from pandapower.networks import case_ieee30
from pandapower.timeseries import DFData
from controllers.transformer_control import TransformerDisconnect
from utils.Generate_fdi import generate_fdi_list

def create_network():
    net = pn.case14()

    return net


def create_ds(time_steps=100, base_gen=50, gen_amplitude=50):
    p_mw = base_gen + gen_amplitude * np.sin(np.linspace(0, 2 * np.pi, time_steps))
    df_gen = pd.DataFrame({'p_mw': p_mw}, index=range(time_steps))
    return pp.timeseries.DFData(df_gen)

def create_30_network():
    net = case_ieee30()
    for idx, row in net.trafo.iterrows():
        lv_bus = row["lv_bus"]
        hv_bus = row["hv_bus"]

        connected_loads = net.load[net.load.bus == lv_bus]

        if not connected_loads.empty:
            print(f"🔍 Transformer {idx} supplies bus {lv_bus} with loads. Adding redundancy.")

            candidate_buses = list(set(net.bus.index) - {lv_bus})
            if candidate_buses:
                target_bus = candidate_buses[0]
                pp.create_line_from_parameters(
                    net,
                    from_bus=lv_bus,
                    to_bus=target_bus,
                    length_km=0.1,
                    r_ohm_per_km=0.1,
                    x_ohm_per_km=0.2,
                    c_nf_per_km=10,
                    max_i_ka=0.5,
                    name=f"tie_line_{lv_bus}_{target_bus}"
                )
                print(f"✅ Added tie-line from bus {lv_bus} to bus {target_bus}")
            else:
                pp.create_gen(
                    net,
                    bus=lv_bus,
                    p_mw=0.2,
                    vm_pu=1.02,
                    slack=False,
                    name=f"backup_gen_bus{lv_bus}"
                )
                print(f"⚡ Added backup generator at bus {lv_bus}")
    return net



def create_stable_gen_profile(net, time_steps=100, base_gen_factor=2.0):
    max_load = sum(net.load["p_mw"]) * len(net.gen) * base_gen_factor
    gen_profile = np.full(time_steps, max_load, dtype=float)
    df = pd.DataFrame({"p_mw": gen_profile}, index=range(time_steps))
    return DFData(df)


def create_load_profile(time_steps=100, base_load=60, load_amplitude=30, overload_steps=None, overload_factor=2.0):
    hours = np.linspace(0, 24, time_steps)
    daily_variation = load_amplitude * np.sin(2 * np.pi * (hours - 6) / 24)

    dynamic_load_profile = base_load + daily_variation

    if overload_steps is None:
        overload_steps = random.sample(range(time_steps), int(time_steps * 0.1))
    overload_mask = np.zeros(time_steps, dtype=bool)
    overload_mask[overload_steps] = True

    dynamic_load_profile[overload_mask] *= overload_factor

    noise = np.random.uniform(-5, 5, time_steps)
    dynamic_load_profile = np.clip(dynamic_load_profile + noise, 0, None)

    return pd.DataFrame({"p_mw": dynamic_load_profile})


def add_support_sgen_to_transformers(net, time_steps=200, base_p_mw=30.0, fluctuation=10.0):
    for idx, row in net.trafo.iterrows():
        hv_bus = row["hv_bus"]
        sgen_idx = pp.create_sgen(net, bus=hv_bus, p_mw=base_p_mw, q_mvar=0.0, name=f"sgen_trafo_{idx}")
        time = np.arange(time_steps)
        profile = base_p_mw + fluctuation * np.sin(2 * np.pi * time / time_steps)
        profile = np.clip(profile, 0, None)
        profile_df = pd.DataFrame({"p_mw": profile})
        ds = DFData(profile_df)
        ConstControl(net, element="sgen", variable="p_mw",
                     element_index=[sgen_idx],
                     data_source=ds,
                     profile_name="p_mw")


def inject_transformer_overload_safely(net, time_steps, events_per_trafo=3,
                                       base_load=10.0,
                                       min_factor=2.0, max_factor=3.0,
                                       min_duration=10, max_duration=30):
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
            profile[start:start + dur] *= factor

        profile_df = pd.DataFrame({"p_mw": profile})
        for load_idx in load_indices:
            ds = DFData(profile_df)
            pp.control.ConstControl(
                net, element="load", variable="p_mw",
                element_index=[load_idx],
                data_source=ds,
                profile_name="p_mw"
            )


def build_net_for_rl(time_steps=200,
                     max_temperature=147.44,
                     num_attacks=20,
                     min_faulty_data=150.0,
                     max_faulty_data=160.0,
                     base_gen_factor=1.5,
                     return_metadata=False):
    net = create_30_network()
    trafo_indices = list(net.trafo.index)

    add_support_sgen_to_transformers(net, time_steps=time_steps)
    inject_transformer_overload_safely(net, time_steps=time_steps)

    gen_profile = create_stable_gen_profile(net, time_steps=time_steps, base_gen_factor=base_gen_factor)
    ConstControl(net, element='gen', variable='p_mw', element_index=[0],
                 data_source=gen_profile, profile_name='p_mw', order=0)

    fdi_list = generate_fdi_list(time_steps, num_attacks, min_faulty_data, max_faulty_data)
    fdi_per_trafo = [[] for _ in trafo_indices]
    for fdi in fdi_list:
        target = random.choice(range(len(trafo_indices)))
        fdi_per_trafo[target].append(fdi)

    for i, index in enumerate(trafo_indices):
        TransformerDisconnect(
            net=net,
            trafo_index=index,
            max_temperature=max_temperature,
            T_ambient=25.0,
            T_rated=65.0,
            n=1.6,
            fdi_list=fdi_per_trafo[i],
            total_steps=time_steps,
            order=1
        )

    if return_metadata:
        metadata = {
            "fdi_per_trafo": fdi_per_trafo,
            "num_const_controls": int(sum(1 for _, row in net.controller.iterrows()
                                          if isinstance(row["object"], ConstControl))),
            "num_transformer_disconnect_controls": int(sum(1 for _, row in net.controller.iterrows()
                                                           if isinstance(row["object"], TransformerDisconnect))),
        }
        return net, trafo_indices, metadata
    return net, trafo_indices


