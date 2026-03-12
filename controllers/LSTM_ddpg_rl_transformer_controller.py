import torch
import numpy as np
from pandapower.control.basic_controller import Controller
from models.LSTM_DDPG import Actor

class LSTMTransformerController(Controller):
    def __init__(self, env, trafo_index, seq_len, max_temperature,
                 T_ambient=25.0, T_rated=65.0, n=1.6,
                 fdi_list=None, total_steps=200,
                 in_service=True, order=0, level=0,
                 model_path=None, **kwargs):
        super().__init__(env.net, in_service=in_service, order=order, level=level, **kwargs)
        self.env = env
        self.trafo_index = trafo_index
        self.seq_len = seq_len
        self.max_temperature = max_temperature
        self.T_ambient = T_ambient
        self.T_rated = T_rated
        self.n = n
        self.fdi_list = fdi_list if fdi_list is not None else []
        self.total_steps = total_steps
        self.model_path = model_path
        self.current_time_step = None
        self.controller_converged = False

        self.tp = self.fp = self.fn = self.tn = 0

        if model_path:
            self.actor = Actor(env.get_state_size(), action_dim=1)
            self.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"[CHECK] Loaded actor weights for trafo {self.trafo_index}:",
                  list(self.actor.parameters())[0].data.abs().mean().item())

            self.actor.eval()
        else:
            self.actor = None

    def calculate_temperature(self, loading_percent):
        return self.T_ambient + self.T_rated * (loading_percent / 100) ** self.n

    def control_step(self, net):
        if self.controller_converged:
            return

        time_step = self.current_time_step
        if time_step is None:
            return

        try:
            loading_percent = np.nan_to_num(net.res_trafo.at[self.trafo_index, 'loading_percent'], 0.0)
        except KeyError:
            print(f"Time step {time_step}: KeyError - No data for transformer {self.trafo_index}")
            self.controller_converged = True
            return

        actual_temp = self.calculate_temperature(loading_percent)
        # net.trafo.at[self.trafo_index, 'actual_temperature'] = actual_temp
        # net.trafo.at[self.trafo_index, 'temperature_measured'] = actual_temp

        print(f"Time step {time_step}: Actual temperature of transformer {self.trafo_index} = {actual_temp:.2f}°C")

        state_seq = self.env.get_local_state(self.trafo_index)  # shape: [seq_len, state_dim]
        state_seq = self.normalize_sequence(state_seq)
        action = self.select_action(state_seq)
        if net.res_trafo.at[self.trafo_index, 'loading_percent'] > 155:
            action = 0.9
        self.env.p[self.trafo_index] = action
        net.trafo.at[self.trafo_index, "in_service"] = action < 0.5

        is_disconnected = not net.trafo.at[self.trafo_index, "in_service"]

        print(f"Step {time_step}: LSTM-RL sets trafo {self.trafo_index} {'Disconnected' if is_disconnected else 'In Service'} (p={action:.6f})")

        # confusion matrix tracking
        should_disconnect = actual_temp > self.max_temperature
        if should_disconnect and is_disconnected:
            self.tp += 1
        elif should_disconnect and not is_disconnected:
            self.fn += 1
        elif not should_disconnect and is_disconnected:
            self.fp += 1
        else:
            self.tn += 1

        self.env.step_count = time_step
        self.controller_converged = True

    def select_action(self, state_seq):
        if self.actor is None:
            print(f"Warning: no LSTM actor available for transformer {self.trafo_index}")
            return 0.0
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # [1, seq_len, dim]
            raw_out, _ = self.actor(state_tensor)
            action = raw_out.squeeze().item()
            print(f"[DEBUG] Raw output: {raw_out.item():.4f}, Action after clip: {action:.4f}")
        return float(np.clip(action, 0.0, 1.0))

    def normalize_sequence(self, seq):
        # seq: [seq_len, state_dim]
        norm_seq = np.array(seq)
        for i in range(seq.shape[1]):
            col = seq[:, i]
            min_val = np.min(col)
            max_val = np.max(col)
            if max_val - min_val > 0:
                norm_seq[:, i] = (col - min_val) / (max_val - min_val)
        return np.nan_to_num(norm_seq)

    def time_step(self, net, time):
        self.current_time_step = time
        self.controller_converged = False

    def is_converged(self, net):
        return self.controller_converged

    def print_confusion_matrix(self):
        print("\n[Confusion Matrix Summary]")
        print(f"TP (should disconnect, disconnected):     {self.tp}")
        print(f"FN (should disconnect, stayed connected): {self.fn}")
        print(f"TN (should stay, stayed connected):       {self.tn}")
        print(f"FP (should stay, disconnected):           {self.fp}")
        total = self.tp + self.fp + self.fn + self.tn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
