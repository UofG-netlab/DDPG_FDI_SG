from distutils.dep_util import newer

import numpy as np
from pandapower.control.basic_controller import Controller
import pandapower as pp

class MonitorController(Controller):
    def __init__(self, net, **kwargs):
        super().__init__(net, **kwargs)
        self.net = net
        self.not_coverged_steps = []
        self.transformer_overloads_steps = []
        self.low_voltage_steps = []
        self.current_time_step = None
        self.controller_converged = False

    def control_step(self, net):
        if self.controller_converged:
            return
        time_step = self.current_time_step

        if time_step is None:
            return

        if not net.converged:
            self.not_coverged_steps.append(time_step)
            print(f"Time step {time_step} not converged")
            return

        if "loading_percent" in net.res_trafo.columns:
            overloaded_trafos = net.res_trafo[net.res_trafo["loading_percent"] > 1.05]
            if not overloaded_trafos.empty:
                self.transformer_overloads_steps.append(overloaded_trafos)
                print(f"Time Step {time_step} overloads are detected")
                print(f"the overload trafos are {overloaded_trafos}")

        if "vm_pu" in net.res_bus.columns:
            low_voltage_buses = net.res_bus[net.res_bus["vm_pu"] < 0.9]
            if not low_voltage_buses.empty:
                self.low_voltage_steps.append(time_step)
                print(f"Time Step {self.low_voltage_steps} low voltage buses are detected")
                print(f"the low voltage buses are {low_voltage_buses}")

        self.controller_converged = True


    def time_step(self, net, time):
        self.current_time_step = time
        self.controller_converged = False

    def is_converged(self, net):
        return self.controller_converged
