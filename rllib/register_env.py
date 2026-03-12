from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv

from envs.pz_substation_env import SubstationParallelEnv


def make_env(env_config):
    return PettingZooEnv(SubstationParallelEnv(**env_config))


def register_substation_env(env_name="SubstationParallelEnv"):
    register_env(env_name, make_env)


def default_ppo_config(env_name="SubstationParallelEnv"):
    return {
        "env": env_name,
        "framework": "torch",
        "env_config": {
            "mode": "ddpg",
            "seq_len": 3,
            "total_steps": 200,
            "max_temperature": 147.44,
        },
        "num_workers": 0,
    }
