from pandapower.control.basic_controller import Controller

class FDIAttackController(Controller):
    def __init__(self, net, trafo_index, T_ambient=25.0, ΔT_rated=65.0, n=1.6, fdi_list=None,
                 total_steps=200, in_service=True, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level, **kwargs)
        self.net = net
        self.trafo_index = trafo_index
        self.T_ambient = T_ambient
        self.ΔT_rated = ΔT_rated
        self.n = n
        self.fdi_list = fdi_list if fdi_list is not None else []
        self.total_steps = total_steps
        self.current_time_step = None
        self.controller_converged = False

    def calculate_temperature(self, loading_percent):
        return self.T_ambient + self.ΔT_rated * (loading_percent / 100) ** self.n

    def control_step(self, net):
        if self.controller_converged:
            return

        time_step = self.current_time_step

        if time_step is None:
            return
        try:
            actual_loading_percent = net.res_trafo.at[self.trafo_index, 'loading_percent']
        except KeyError:
            print(f"Time step {time_step}: Transformer {self.trafo_index} - No loading data available.")
            self.controller_converged = True
            return

        actual_temperature = self.calculate_temperature(actual_loading_percent)
        print(f"Time step {time_step}: Transformer {self.trafo_index} actual temperature: {actual_temperature:.2f}°C")

        current_temperature = actual_temperature
        for f_step, fake_temperature in self.fdi_list:
            if f_step == time_step:
                current_temperature = fake_temperature
                print(f"Time step {time_step}: FDI Injected! Fake temperature for Transformer {self.trafo_index} = {current_temperature:.2f}°C")
                break

        print(f"Time step {time_step}: Transformer {self.trafo_index} reported temperature: {current_temperature:.2f}°C\n")

        self.controller_converged = True

    def time_step(self, net, time):
        self.current_time_step = time
        self.controller_converged = False

    def is_converged(self, net):
        return self.controller_converged
