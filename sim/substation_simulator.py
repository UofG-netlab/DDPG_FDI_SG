import numpy as np
import pandapower as pp
from pandapower.control import ConstControl

from controllers.transformer_control import TransformerDisconnect
from envs.DDPG_multi_agent_substation_env import ddpg_multi_agent_substation_env
from envs.LSTM_multi_agent_substation_env import LSTM_MultiAgentEnv
from utils.network import build_net_for_rl


class SubstationSimulator:
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

        self.net = None
        self.local_env = None
        self.trafo_indices = []
        self.agents = []
        self.current_time = 0
        self.pending_actions = {}
        self.current_observations = {}
        self.log_history = []
        self._const_controllers = []
        self._fdi_controllers = []

    def _build(self):
        self.net, self.trafo_indices = self.net_factory(
            time_steps=self.total_steps,
            max_temperature=self.max_temperature,
            **self.net_factory_kwargs
        )
        if self.mode == "lstm":
            self.local_env = LSTM_MultiAgentEnv(
                net=self.net,
                trafo_indices=self.trafo_indices,
                seq_len=self.seq_len,
                total_steps=self.total_steps,
                max_temperature=self.max_temperature
            )
        else:
            self.local_env = ddpg_multi_agent_substation_env(
                net=self.net,
                trafo_indices=self.trafo_indices,
                total_steps=self.total_steps,
                max_temperature=self.max_temperature
            )
        self.agents = [f"trafo_{idx}" for idx in self.trafo_indices]
        self._collect_controllers()

    def _collect_controllers(self):
        self._const_controllers = []
        self._fdi_controllers = []
        if not hasattr(self.net, "controller") or self.net.controller.empty:
            return

        indexed = []
        for _, row in self.net.controller.iterrows():
            obj = row["object"]
            order = row["order"] if "order" in row else getattr(obj, "order", 0)
            level = row["level"] if "level" in row else getattr(obj, "level", 0)
            indexed.append((order, level, obj))
        indexed.sort(key=lambda x: (x[0], x[1]))

        for _, _, obj in indexed:
            if isinstance(obj, ConstControl):
                self._const_controllers.append(obj)
            elif isinstance(obj, TransformerDisconnect):
                self._fdi_controllers.append(obj)

    def _run_controllers_for_time(self, time_idx):
        for ctrl in self._const_controllers + self._fdi_controllers:
            if hasattr(ctrl, "time_step"):
                ctrl.time_step(self.net, time_idx)

        for ctrl in self._const_controllers:
            ctrl.control_step(self.net)

        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            pass

        for ctrl in self._fdi_controllers:
            ctrl.control_step(self.net)

    def _collect_observations(self):
        obs = {}
        self.local_env.step_count = self.current_time
        for idx in self.trafo_indices:
            agent_id = f"trafo_{idx}"
            state = self.local_env.get_local_state(idx)
            obs[agent_id] = np.asarray(state, dtype=np.float32)
        return obs

    def _compute_reward_from_state(self, idx, state):
        if self.mode == "lstm":
            return float(self.local_env.compute_reward(state, idx))
        return float(self.local_env.get_local_reward(state, idx))

    def _apply_actions(self, action_dict):
        normalized = {}
        for idx in self.trafo_indices:
            agent_id = f"trafo_{idx}"
            action = action_dict.get(agent_id, 0.0)
            if isinstance(action, (list, tuple, np.ndarray)):
                action = float(np.asarray(action).squeeze())
            action = float(np.clip(action, 0.0, 1.0))
            normalized[agent_id] = action
            self.local_env.p[idx] = action
            self.net.trafo.at[idx, "in_service"] = action < 0.5
        return normalized

    def _append_log(self):
        step_log = {"time": self.current_time}
        for idx in self.trafo_indices:
            step_log[f"loading_{idx}"] = float(np.nan_to_num(self.net.res_trafo.at[idx, "loading_percent"], nan=0.0))
            step_log[f"in_service_{idx}"] = bool(self.net.trafo.at[idx, "in_service"])
            if "temperature_measured" in self.net.trafo.columns:
                step_log[f"temp_measured_{idx}"] = float(np.nan_to_num(self.net.trafo.at[idx, "temperature_measured"], nan=25.0))
        self.log_history.append(step_log)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._build()
        self.current_time = 0
        self.pending_actions = {}
        self.log_history = []

        self._run_controllers_for_time(time_idx=0)
        self.current_observations = self._collect_observations()
        self._append_log()
        infos = {agent: {} for agent in self.agents}
        return self.current_observations, infos

    def step(self, action_dict):
        rewards = {}
        if not self.pending_actions:
            for agent in self.agents:
                rewards[agent] = 0.0
        else:
            for idx in self.trafo_indices:
                agent_id = f"trafo_{idx}"
                rewards[agent_id] = self._compute_reward_from_state(idx, self.current_observations[agent_id])

        normalized_actions = self._apply_actions(action_dict)
        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            pass

        self.current_time += 1
        self.local_env.step_count = self.current_time

        if self.current_time < self.total_steps:
            self._run_controllers_for_time(time_idx=self.current_time)

        self.current_observations = self._collect_observations()
        self.pending_actions = normalized_actions
        self._append_log()

        terminations = {agent: False for agent in self.agents}
        truncated = self.current_time >= self.total_steps
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {"time": self.current_time} for agent in self.agents}

        return self.current_observations, rewards, terminations, truncations, infos
