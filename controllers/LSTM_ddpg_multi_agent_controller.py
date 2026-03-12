import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pandapower.control.basic_controller import Controller
from models.LSTM_DDPG import Actor, Critic, MultiAgentLSTMDDPGTrainer
import matplotlib.pyplot as plt

class LSTM_DDPGMultiAgentController(Controller):
    def __init__(self, env, trainer: MultiAgentLSTMDDPGTrainer, trafo_indices, max_temperature = 147.0, T_ambient = 25, T_rated = 65, n = 1.6, **kwargs):
        super().__init__(env.net, **kwargs)
        self.env = env
        self.trainer = trainer
        self.trafo_indices = trafo_indices
        self.pending_transitions = {idx: None for idx in self.trafo_indices}
        self.reward_history = {idx: [] for idx in self.trafo_indices}
        self.applied = False
        self.max_temperature = max_temperature
        self.T_ambient = T_ambient
        self.T_rated = T_rated
        self.n = n
        self.tp = self.fp = self.fn = self.tn = 0

    def control_step(self, net):
        current_time = self.env.step_count

        # Step 1: collect current state
        state_dict = {idx: self.normalize_state(self.env.get_local_state(idx)) for idx in self.trafo_indices}
        action_dict = {}

        for idx in self.trafo_indices:
            state = state_dict[idx]
            action = self.trainer.select_action(idx, state)
            action = float(np.clip(action, 0.0, 1.0))
            self.env.p[idx] = action
            net.trafo.at[idx, "in_service"] = action < 0.5  # apply action (high action => disconnect)
            print(f"[TRAIN] Trafo {idx} | action={action:.6f}")
            action_dict[idx] = action

            print(f"[t = {current_time}] Transformer {idx} action = {action:.3f}, in_service = {net.trafo.at[idx, 'in_service']}")

        # Step 2: finalize delayed transition from previous step
        for idx in self.trafo_indices:
            prev = self.pending_transitions[idx]
            if prev is None:
                continue
            prev_state, prev_action = prev
            current_state = state_dict[idx]
            reward = self.env.compute_reward(current_state, idx)
            self.reward_history[idx].append(reward)
            done = self.env.step_count >= self.env.total_steps
            self.trainer.store_experience(idx, prev_state, float(prev_action), reward, current_state, done)

        # Step 3: cache current state and action for next step
        for idx in self.trafo_indices:
            self.pending_transitions[idx] = (state_dict[idx], action_dict[idx])

        # Step 4: run multi-agent training after each step
        self.trainer.learn_all()
        self.env.step_count += 1

        # Step 5: update TP/FP/FN/TN each step
        for idx in self.trafo_indices:
            action = self.env.p[idx]
            in_service = net.trafo.at[idx, "in_service"]
            real_loading = net.res_trafo.loading_percent.at[idx]
            actual_temp = self.calculate_temperature(real_loading)

            should_disconnect = (actual_temp > self.max_temperature)
            did_disconnect = not in_service  # in_service=False means disconnected

            if should_disconnect and did_disconnect:
                self.tp += 1
            elif should_disconnect and not did_disconnect:
                self.fn += 1
            elif not should_disconnect and did_disconnect:
                self.fp += 1
            elif not should_disconnect and not did_disconnect:
                self.tn += 1

        self.applied = True

    def is_converged(self, net):
        return self.applied

    def time_step(self, net, time):
        self.applied = False

    def normalize_state(self, state_seq):
        if state_seq is None or len(state_seq) == 0:
            return np.zeros_like(state_seq)

        min_vals = np.min(state_seq, axis=0)
        max_vals = np.max(state_seq, axis=0)
        denom = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
        norm_seq = (state_seq - min_vals) / denom
        return np.nan_to_num(norm_seq)

    def calculate_temperature(self, loading_percent):
        return self.T_ambient + self.T_rated * (loading_percent / 100) ** self.n


