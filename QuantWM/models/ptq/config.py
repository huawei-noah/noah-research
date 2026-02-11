from .bit_type import BIT_TYPE_DICT


class Config:

    def __init__(self, w_bit=8, a_bit=8, w_quant_method='minmax', a_quant_method='minmax', calib_mode_a='layer_wise'):

        self.BIT_TYPE_W = BIT_TYPE_DICT[f'int{w_bit}']
        self.BIT_TYPE_A = BIT_TYPE_DICT[f'uint{a_bit}']

        self.OBSERVER_W = w_quant_method
        self.OBSERVER_A = a_quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_S = 'log2'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = calib_mode_a

