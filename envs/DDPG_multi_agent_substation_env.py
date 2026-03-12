import numpy as np
import pandapower as pp

class ddpg_multi_agent_substation_env:
    def __init__(self, net, trafo_indices, delta_p=0.05, initial_p=0.0,
                 voltage_tolerance=0.05, voltage_penalty_factor=10.0,
                 line_loading_limit=20.0, power_flow_penalty_factor=5.0,
                 load_reward_factor=20.0, transformer_reward_factor=20.0,
                 disconnection_penalty_factor=50.0, total_steps=200, max_temperature=90):
        self.net = net
        self.trafo_indices = trafo_indices
        self.delta_p = delta_p
        self.initial_p = initial_p
        self.voltage_tolerance = voltage_tolerance
        self.voltage_penalty_factor = voltage_penalty_factor
        self.line_loading_limit = line_loading_limit
        self.power_flow_penalty_factor = power_flow_penalty_factor
        self.load_reward_factor = load_reward_factor
        self.transformer_reward_factor = transformer_reward_factor
        self.disconnection_penalty_factor = disconnection_penalty_factor
        self.temperature_history = {idx: np.zeros(5) for idx in trafo_indices}
        self.total_steps = total_steps
        self.step_count = 0
        self.T_ambient = 25.0
        self.T_rated = 65.0
        self.n = 1.6
        self.max_temperature = max_temperature
        self.p = {idx: initial_p for idx in trafo_indices}

    def get_state_size(self):
        return 12  # add actual_temperature and fdi_delta features

    def get_local_state(self, trafo_index):
        try:
            real_loading = float(np.nan_to_num(self.net.res_trafo.at[trafo_index, 'loading_percent'], nan=0.0))
            reading = float(self.net.trafo.at[trafo_index, 'temperature_measured'])

            try:
                actual_temp = float(self.net.trafo.at[trafo_index, 'actual_temperature'])
            except:
                actual_temp = reading  # fallback

            fdi_delta = reading - actual_temp

            self.temperature_history[trafo_index] = np.roll(self.temperature_history[trafo_index], -1)
            self.temperature_history[trafo_index][-1] = reading
            trend = float(np.mean(self.temperature_history[trafo_index]))
            jump = reading - trend

            lv_bus = self.net.trafo.at[trafo_index, "lv_bus"]
            vm_pu = float(np.nan_to_num(self.net.res_bus.at[lv_bus, "vm_pu"], nan=1.0))

            connected_lines = self.net.line[
                (self.net.line.from_bus == lv_bus) | (self.net.line.to_bus == lv_bus)
            ].index
            if not connected_lines.empty:
                local_line_loading = self.net.res_line.loc[connected_lines, "loading_percent"]
                line_mean = float(local_line_loading.mean())
                line_max = float(local_line_loading.max())
            else:
                line_mean = 0.0
                line_max = 0.0

            local_loads = self.net.load[self.net.load.bus == lv_bus]
            total_load_p = float(local_loads["p_mw"].sum()) if not local_loads.empty else 0.0

            in_service = float(self.net.trafo.at[trafo_index, "in_service"])

            state = np.array([
                real_loading,
                reading,
                actual_temp,
                fdi_delta,
                jump,
                vm_pu,
                line_mean,
                line_max,
                total_load_p,
                in_service,
                self.step_count / self.total_steps,
                float(abs(fdi_delta) > 5.0)
            ])
            return state
        except Exception as e:
            print(f"[Env ERROR] get_local_state failed for Trafo {trafo_index}: {e}")
            return np.zeros(self.get_state_size())

    def get_local_reward(self, state, trafo_index):
        (
            real_loading, reading, actual_temp, fdi_delta,
            jump, vm_pu, line_mean, line_max,
            total_load_p, in_service, norm_time, fdi_flag
        ) = state

        disconnect = not bool(in_service)
        overheat = actual_temp > self.max_temperature
        suspected_fdi = abs(fdi_delta) > 5.0

        reward = 0.0

        # correct disconnect
        if overheat and disconnect:
            reward += 6.0
        # missed disconnect
        elif overheat and not disconnect:
            reward -= 6.0
        # false disconnect
        elif not overheat and disconnect:
            reward -= 5.0
            if suspected_fdi:
                reward += 1.5  # reduce penalty if misled by FDI
        # keep running
        elif not overheat and not disconnect:
            reward += 2.0

        # low-voltage penalty
        if vm_pu < 0.95:
            reward -= (0.95 - vm_pu) * 2.0

        # line overload penalty
        if not disconnect and line_max > self.line_loading_limit:
            reward -= 2.0

        return reward
