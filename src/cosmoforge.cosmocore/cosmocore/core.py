from abc import ABC

from .settings import InputParams


class Core(ABC):
    def __init__(self, ell_max, noise_cl=None):
        self.ell_max = ell_max
        self.noise_cl = noise_cl

    def read_params(self, params: InputParams | str | dict):
        if isinstance(params, InputParams):
            self.params = params
        elif isinstance(params, str):
            self.params = InputParams.read_parameter_file(params)
        elif isinstance(params, dict):
            self.params = InputParams()
            self.params.update(params)
        else:
            msg = (
                "params must be an instance of InputParams, "
                "a string with the path to a parameter file, "
                "or a dictionary with parameters."
            )
            raise TypeError(msg)
