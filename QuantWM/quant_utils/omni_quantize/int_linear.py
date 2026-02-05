import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import UniformAffineQuantizer
from models.ptq.observer import build_observer
from models.ptq.quantizer import build_quantizer



class QuantLinear(nn.Module):
    """
    OmniQuant-compatible wrapper for existing QLinear.

    - Reuses QLinear's observer / quantizer / bit_type
    - Supports LET (temp_weight / temp_bias)
    - Does NOT do activation quant (handled by QAct)
    """

    def __init__(self, org_module: nn.Linear):
        super().__init__()

        assert isinstance(org_module, nn.Linear)

        # -------- basic linear --------
        self.fwd_func = F.linear
        self.fwd_kwargs = {}

        self.register_buffer("weight", org_module.weight)
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias)
        else:
            self.bias = None

        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        # -------- quant state --------
        self.use_weight_quant = False
        self.use_temporary_parameter = False

        # -------- reuse QLinear quantizer if exists --------
        if hasattr(org_module, "quantizer") and hasattr(org_module, "observer"):

            self.quantizer = org_module.quantizer
            self.observer = org_module.observer
            self.bit_type = org_module.bit_type
            self.calibration_mode = org_module.calibration_mode
            self.observer_str = org_module.observer_str
            self.quantizer_str = org_module.quantizer_str
            self.use_weight_quant = org_module.quant
        else:
            self.quantizer = None
            self.observer = None
            self.bit_type = None

    def forward(self, x: torch.Tensor):

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias


        elif self.use_weight_quant and self.quantizer is not None:
            weight = self.quantizer(self.weight)
            bias = self.bias

        else:
            weight = self.weight
            bias = self.bias

        return self.fwd_func(x, weight, bias, **self.fwd_kwargs)


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
