import numpy as np
import attr
from scipy.integrate import solve_ivp

from . import _utils as ut

@attr.s(kw_only=True)
class SIR_model:
    """
    """
    S0 = attr.ib(1e6, validator=ut.positive)
    I0 = attr.ib(20., validator=ut.positive)
    R0 = attr.ib(2., validator=ut.positive)
    beta = attr.ib(0.0000004, validator=ut.positive)
    gamma = attr.ib(0.2, validator=ut.positive)
    time = attr.ib(150, type=int, validator=ut.positive)
    gran = attr.ib(1000, type=int, validator=ut.positive)

    def SIR(self, t, y):
        """
        """
        S, I, R = y
        return [-self.beta * S * I, self.beta * S * I - self.gamma * I, self.gamma * I]

    def integrate(self):
        """
        """
        ivp = solve_ivp(self.SIR, [0, self.time], [self.S0, self.I0, self.R0],
                        t_eval=np.linspace(0, self.time, self.gran), vectorized=True)
        return ivp
