import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pandapower.control.basic_controller import Controller


class DQN(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # stay or disconnect
        )

    def forward(self, x):
        return self.net(x)


class DQNTransformerController(Controller):
    def __init__(self, env, trafo_index, model_path=None,
                 T_ambient=25.0, T_rated=65.0, n=1.6,
                 max_temperature=147.44, fdi_list=None, **kwargs):

        super().__init__(env.net, order=0, level=0)

        self.env = env
        self.trafo_index = trafo_index
        self.model_path = model_path
        self.T_ambient = T_ambient
        self.T_rated = T_rated
        self.n = n
        self.max_temperature = max_temperature
        self.fdi_list = fdi_list or []

        self.state_dim = env.get_state_size()
        self.policy = DQN(self.state_dim)

        if model_path:
            self.policy.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.policy.eval()

        # metrics
        self.tp = self.fp = self.fn = self.tn = 0

        self.current_time_step = None
        self.controller_converged = False


    def calculate_temp(self, loading_percent):
        load = loading_percent / 100.0
        return self.T_ambient + self.T_rated * (load ** self.n)


    def control_step(self, net):
        if self.controller_converged:
            return

        t = self.current_time_step

        # get loading
        loading_percent = float(np.nan_to_num(
            net.res_trafo.at[self.trafo_index, "loading_percent"], 0.0
        ))

        actual_temp = self.calculate_temp(loading_percent)
        measured_temp = actual_temp

        # inject FDI
        for step, fake_temp in self.fdi_list:
            if step == t:
                measured_temp = fake_temp

        net.trafo.at[self.trafo_index, "actual_temperature"] = actual_temp
        net.trafo.at[self.trafo_index, "temperature_measured"] = measured_temp

        # RL STATE
        state = np.array(self.env.get_local_state(self.trafo_index), dtype=float)
        state = (state - state.mean()) / (state.std() + 1e-6)
        state = torch.FloatTensor(state).unsqueeze(0)

        # DQN ACTION
        with torch.no_grad():
            q_values = self.policy(state)
            action = int(torch.argmax(q_values))

        disconnect = (action == 1)
        net.trafo.at[self.trafo_index, "in_service"] = not disconnect

        # Ground-truth label
        should_disconnect = actual_temp > self.max_temperature

        if should_disconnect and disconnect:
            self.tp += 1
        elif should_disconnect and not disconnect:
            self.fn += 1
        elif not should_disconnect and disconnect:
            self.fp += 1
        else:
            self.tn += 1

        self.controller_converged = True


    def time_step(self, net, time):
        self.current_time_step = time
        self.controller_converged = False


    def is_converged(self, net):
        return self.controller_converged


    def print_confusion_matrix(self):
        print("\n[DQN Confusion Matrix] Trafo", self.trafo_index)
        print("TP:", self.tp, " FP:", self.fp)
        print("FN:", self.fn, " TN:", self.tn)

        precision = self.tp / (self.tp + self.fp + 1e-9)
        recall = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print(f"Precision {precision:.3f}  Recall {recall:.3f}  F1 {f1:.3f}")