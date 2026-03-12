import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from sim.substation_simulator import SubstationSimulator
from utils.network import build_net_for_rl


class SubstationParallelEnv(ParallelEnv):
    metadata = {"name": "SubstationParallelEnv"}

    def __init__(self,
                 mode="ddpg",
                 seq_len=3,
                 total_steps=200,
                 max_temperature=147.44,
                 net_factory=None,
                 net_factory_kwargs=None):
        self.mode = mode
        self.seq_len = seq_len
        self.total_steps = total_steps
        self.max_temperature = max_temperature
        self.net_factory = net_factory or build_net_for_rl
        self.net_factory_kwargs = net_factory_kwargs or {}

        self.simulator = None
        self.net = None
        self.agents = []
        self.possible_agents = []
        self._action_spaces = {}
        self._observation_spaces = {}

    def _init_spaces(self, observations):
        action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._action_spaces = {agent: action_space for agent in self.agents}

        self._observation_spaces = {}
        for agent, obs in observations.items():
            obs_arr = np.asarray(obs, dtype=np.float32)
            self._observation_spaces[agent] = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_arr.shape, dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        self.simulator = SubstationSimulator(
            mode=self.mode,
            seq_len=self.seq_len,
            total_steps=self.total_steps,
            max_temperature=self.max_temperature,
            net_factory=self.net_factory,
            net_factory_kwargs=self.net_factory_kwargs,
        )
        observations, infos = self.simulator.reset(seed=seed)
        self.net = self.simulator.net
        self.agents = list(self.simulator.agents)
        self.possible_agents = list(self.agents)
        self._init_spaces(observations)
        return observations, infos

    def step(self, actions):
        return self.simulator.step(actions)

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
