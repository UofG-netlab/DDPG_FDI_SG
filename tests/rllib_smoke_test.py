import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.pz_substation_env import SubstationParallelEnv


def run_smoke_test(mode="ddpg", steps=5):
    env = SubstationParallelEnv(
        mode=mode,
        total_steps=steps,
        net_factory_kwargs={"num_attacks": steps}
    )
    observations, _ = env.reset()
    initial_time = env.simulator.current_time
    load_snapshots = [tuple(env.simulator.net.load["p_mw"].round(6).tolist())]
    first_step_rewards = None
    saw_fdi_delta = False
    done = False

    for _ in range(steps):
        actions = {}
        for agent_id in env.agents:
            action = env.action_space(agent_id).sample()
            actions[agent_id] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        load_snapshots.append(tuple(env.simulator.net.load["p_mw"].round(6).tolist()))

        if first_step_rewards is None:
            first_step_rewards = rewards

        net = env.simulator.net
        if "temperature_measured" in net.trafo.columns and "actual_temperature" in net.trafo.columns:
            delta = (net.trafo["temperature_measured"] - net.trafo["actual_temperature"]).abs().max()
            if float(delta) > 0:
                saw_fdi_delta = True

        if all(truncations.values()):
            done = True
            break

    assert env.simulator.current_time > initial_time, "Timestep did not advance."
    assert len(set(load_snapshots)) > 1, "Exogenous load profile did not change over steps."
    assert first_step_rewards is not None and all(v == 0.0 for v in first_step_rewards.values()), \
        "Delayed reward semantics violated: first-step rewards must be zero."
    assert saw_fdi_delta, "FDIA effect was not observed in measured temperature."
    assert done, "Episode did not truncate at total_steps."
    return observations, rewards


if __name__ == "__main__":
    run_smoke_test(mode="ddpg", steps=5)
    run_smoke_test(mode="lstm", steps=5)
