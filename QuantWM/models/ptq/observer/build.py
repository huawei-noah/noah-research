# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .ema import EmaObserver
from .minmax import MinmaxObserver
from .omse import OmseObserver
from .percentile import PercentileObserver
from .ptf import PtfObserver

str2observer = {
    'minmax': MinmaxObserver,
    'ema': EmaObserver,
    'omse': OmseObserver,
    'percentile': PercentileObserver,
    'ptf': PtfObserver,
    'awq': MinmaxObserver,
    'omniquant': MinmaxObserver,
}


def build_observer(observer_str, module_type, bit_type, calibration_mode,shape=None):
    if bit_type is None:
        return None
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode,shape)
