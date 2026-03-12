import numpy as np
from collections import deque

class LSTM_MultiAgentEnv:
    def __init__(self, net, trafo_indices, seq_len=3, total_steps=200, max_temperature=147.44):
        self.net = net
        self.trafo_indices = trafo_indices
        self.seq_len = seq_len
        self.total_steps = total_steps
        self.max_temperature = max_temperature
        self.step_count = 0
        self.p = {idx: 0.0 for idx in trafo_indices}
        self.temperature_history = {idx: np.zeros(5) for idx in trafo_indices}
        self.loading_history = {idx: np.zeros(3) for idx in trafo_indices}
        self.state_history = {idx: deque(maxlen=seq_len) for idx in trafo_indices}

    def get_state_size(self):
        return 10

    def get_local_state(self, trafo_index):
        try:
            loading = float(np.nan_to_num(self.net.res_trafo.at[trafo_index, 'loading_percent'], nan=0.0))
            loading = loading / 100.0 # around [0, 2.0]
            temp = float(np.nan_to_num(self.net.trafo.at[trafo_index, 'temperature_measured'], nan=25.0))
            temp = (temp-25) / 150.0  # around 0.2 ~ 1.0

            self.temperature_history[trafo_index] = np.roll(self.temperature_history[trafo_index], -1)
            self.temperature_history[trafo_index][-1] = temp
            temp_trend = float(np.mean(self.temperature_history[trafo_index]))
            jump = temp - temp_trend
            jump = np.tanh(jump / 10.0) #  [-1, 1]

            self.loading_history[trafo_index] = np.roll(self.loading_history[trafo_index], -1)
            self.loading_history[trafo_index][-1] = loading
            delta_loading = self.loading_history[trafo_index][-1] - self.loading_history[trafo_index][-2] # difference between loading at time step t-1 and loading t-2
            delta_temp = self.temperature_history[trafo_index][-1] - self.temperature_history[trafo_index][-2] # difference between measured tempearture at time step t-1 and temerapture at t
            delta_loading = np.tanh(delta_loading / 10.0) # [-0.2, 0.2]
            delta_temp = np.tanh(delta_temp / 10.0) # [-0.2, 0.2]

            lv_bus = self.net.trafo.at[trafo_index, "lv_bus"] # The bus ID connected to the low-voltage side of the transformer
            vm_pu = float(np.nan_to_num(self.net.res_bus.at[lv_bus, "vm_pu"], nan=1.0)) # standard voltage value
            vm_pu = vm_pu - 1.0 # [-0.1, 0.1]

            connected_lines = self.net.line[(self.net.line.from_bus == lv_bus) | (self.net.line.to_bus == lv_bus)].index
            if not connected_lines.empty:
                line_loads = self.net.res_line.loc[connected_lines, "loading_percent"].values
                mean_line_loading = float(np.mean(line_loads)) # trafo neighbour lines avaerge loading
                max_line_loading = float(np.max(line_loads)) # trafo neighbour lines max loading
                mean_line_loading = mean_line_loading / 100.0
                max_line_loading = max_line_loading / 100.0
            else:
                mean_line_loading = 0.0 # [0, 1.0]
                max_line_loading = 0.0 # [0, 1.0]

            local_loads = self.net.load[self.net.load.bus == lv_bus] # The loads connected with transformer lv_bus
            total_load_p = float(local_loads["p_mw"].sum()) if not local_loads.empty else 0.0 # Sum of the all loads' power
            total_load_p = total_load_p / 100.0 # [0, 1.5]
            # Normalisation
            norm_time = self.step_count / self.total_steps
            state = np.array([
                loading,
                temp,
                jump,
                vm_pu,
                mean_line_loading,
                max_line_loading,
                total_load_p,
                norm_time,
                delta_loading,
                delta_temp
            ])

            self.state_history[trafo_index].append(state)
            history = list(self.state_history[trafo_index])
            if len(history) < self.seq_len:
                padding = [np.zeros_like(state)] * (self.seq_len - len(history))
                history = padding + history

            return np.stack(history, axis=0)

        except Exception as e:
            print(f"[Env ERROR] get_local_state failed for Trafo {trafo_index}: {e}")
            return np.zeros((self.seq_len, self.get_state_size()))

    def compute_reward(self, state_seq, trafo_index):
        s = state_seq[-1]
        (
            loading, # [0, 2.0]
            temp, # [0.17, 1.5]
            jump, # [-1, 1]
            vm_pu, # [-0.1, 0.1]
            mean_line, # [0, 1.0]
            max_line, # [0, 1.0]
            load_p, # [0, 1.5]
            norm_time, # [0, 1]
            delta_loading, # [-0.2, 0.2]
            delta_temp # [-0.2, 0.2]
        ) = s
        in_service = float(self.net.trafo.at[trafo_index, "in_service"])
        disconnect = not bool(in_service)

        overload = loading > 1.5

        suspected_fdi = abs(jump) > 0.3 and delta_temp > 0.0 and delta_loading < 0.3

        # Part 1: safe behavior
        r_safety = 0.0
        if overload and disconnect:
            r_safety += 3.0
        elif overload and not disconnect:
            r_safety -= 5.0
        elif not overload and disconnect:
            r_safety -= 1
            if suspected_fdi:
                r_safety += 0.5
        elif not overload and not disconnect:
            r_safety += 0.5

        # Part 2: Voltage penalty
        r_power = 0.0
        if vm_pu < -0.05:  # equivalent to vm_pu < 0.95
            r_power -= (abs(vm_pu + 0.05)) * 2.0
        if not disconnect and max_line > 1.2:  # actual max loading > 120%
            r_power -= (max_line - 1.2) * 1.0

        # Part 3: Trend penalty
        r_trend = 0.0
        if not overload and jump > 0.3 and delta_temp > 0 and loading > 0.9:
            r_trend -= 1.0
        if overload and delta_temp < 0:
            r_trend += 0.5

        reward = 12 * r_safety + r_power + r_trend
        reward = np.clip(reward, -20, 20)

        return reward
