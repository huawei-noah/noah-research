# NEW

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import GPyOpt
import numpy as np
import torch

sys.path[0] = str(Path(os.path.realpath(__file__)).parent.parent)

from core.utils.utils_query import query_exps
from core.bo_runner import run_exp_bo
from core.utils.utils_parsing import get_tuning_config
from core.utils.utils_save import save_best_hyperparams, exp_folder_rel_path
from core.utils.utils_tuning import CustomSingleObjective, get_X_Y_tuning, get_result_dir
from core.params_helper import OptScheduledParamSpace, OptParamSpace

#  ---------------------- ---------------------- ---------------------- ---------------------- ---------------------- #
#                                   Define bounds of hyperparameters for several optimizers                           #
#  ---------------------- ---------------------- ---------------------- ---------------------- ---------------------- #

# ----------------------------- Adam ----------------------------- #

ADAM_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'beta1', 'type': 'continuous', 'domain': (0.05, 0.999)},
    {'name': 'beta2', 'type': 'continuous', 'domain': (-6, -1)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (-8, -1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

ADAM_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'beta1': lambda beta1: beta1},
    {'beta2': lambda beta2: 1 - 10 ** beta2},
    {'weight_decay': lambda weight_decay: 10 ** weight_decay},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

ADAM_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'beta1': lambda beta1: beta1},
    {'beta2': lambda beta2: np.log10(1 - beta2)},
    {'weight_decay': lambda weight_decay: np.log10(weight_decay)},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]


class AdamBaseH(OptScheduledParamSpace):

    def get_true_opt_kwargs(self, all_params_kwargs) -> Dict[str, Any]:
        all_params_kwargs['betas'] = (all_params_kwargs.pop('beta1'), all_params_kwargs.pop('beta2'))
        return all_params_kwargs

    def get_search_opt_kwargs(self, all_params_kwargs) -> Dict[str, Any]:
        all_params_kwargs['beta1'], all_params_kwargs['beta2'] = all_params_kwargs.pop('betas')
        return self.reorder_params(all_params_kwargs)


class AdamH(AdamBaseH):

    def __init__(self):
        super(AdamH, self).__init__(ADAM_HYPERPARAMS_BOUNDS, ADAM_HYPERPARAMS_TRANSFO,
                                    rec_transfos=ADAM_HYPERPARAMS_REC_TRANSFO)


Adam_H = AdamH()

# ----------------------------- AdamW ----------------------------- #

ADAMW_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'beta1', 'type': 'continuous', 'domain': (0.05, 0.999)},
    {'name': 'beta2', 'type': 'continuous', 'domain': (-6, -1)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (-8, -1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

ADAMW_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'beta1': lambda beta1: beta1},
    {'beta2': lambda beta2: 1 - 10 ** beta2},
    {'weight_decay': lambda weight_decay: 10 ** weight_decay},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

ADAMW_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'beta1': lambda beta1: beta1},
    {'beta2': lambda beta2: np.log10(1 - beta2)},
    {'weight_decay': lambda weight_decay: np.log10(weight_decay)},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]


class AdamWH(AdamBaseH):

    def __init__(self):
        super(AdamWH, self).__init__(ADAMW_HYPERPARAMS_BOUNDS, ADAMW_HYPERPARAMS_TRANSFO,
                                     rec_transfos=ADAMW_HYPERPARAMS_REC_TRANSFO)


AdamW_H = AdamWH()

# ----------------------------- Adamax ----------------------------- #

ADAMAX_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'beta1', 'type': 'continuous', 'domain': (0.05, 0.999)},
    {'name': 'beta2', 'type': 'continuous', 'domain': (-6, -1)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (-8, -1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

ADAMAX_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'beta1': lambda beta1: beta1},
    {'beta2': lambda beta2: 1 - 10 ** beta2},
    {'weight_decay': lambda weight_decay: 10 ** weight_decay},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

ADAMAX_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'beta1': lambda beta1: beta1},
    {'beta2': lambda beta2: np.log10(1 - beta2)},
    {'weight_decay': lambda weight_decay: np.log10(weight_decay)},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]


class AdamaxH(AdamBaseH):

    def __init__(self):
        super(AdamaxH, self).__init__(ADAMAX_HYPERPARAMS_BOUNDS, ADAMAX_HYPERPARAMS_TRANSFO,
                                      rec_transfos=ADAMAX_HYPERPARAMS_REC_TRANSFO)


Adamax_H = AdamaxH()

# ----------------------------- Adadelta ----------------------------- #

ADADELTA_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'rho', 'type': 'continuous', 'domain': (-3, -0.05)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (-8, -1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

ADADELTA_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'rho': lambda rho: 1 - 10 ** rho},
    {'weight_decay': lambda weight_decay: 10 ** weight_decay},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

ADADELTA_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'rho': lambda rho: np.log10(1 - rho)},
    {'weight_decay': lambda weight_decay: np.log10(weight_decay)},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]


class AdadeltaH(OptScheduledParamSpace):

    def __init__(self):
        super(AdadeltaH, self).__init__(ADADELTA_HYPERPARAMS_BOUNDS, ADADELTA_HYPERPARAMS_TRANSFO,
                                        rec_transfos=ADADELTA_HYPERPARAMS_TRANSFO)


Adadelta_H = AdadeltaH()

# ----------------------------- Adagrad ----------------------------- #

Adagrad_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'lr_decay', 'type': 'continuous', 'domain': (-7, 1)},
    {'name': 'initial_accumulator_value', 'type': 'continuous', 'domain': (-8, -.5)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (-8, -1)},
]

Adagrad_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'lr_decay': lambda lr_decay: 10 ** lr_decay},
    {'initial_accumulator_value': lambda initial_accumulated_value: 1 - 10 ** initial_accumulated_value},
    {'weight_decay': lambda weight_decay: 10 ** weight_decay},
]

Adagrad_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'lr_decay': lambda lr_decay: np.log10(lr_decay)},
    {'initial_accumulator_value': lambda initial_accumulated_value: np.log10(initial_accumulated_value)},
    {'weight_decay': lambda weight_decay: np.log10(weight_decay)},
]


class AdagradH(OptParamSpace):

    def __init__(self):
        super(AdagradH, self).__init__(Adagrad_HYPERPARAMS_BOUNDS, Adagrad_HYPERPARAMS_TRANSFO,
                                       rec_transfos=Adagrad_HYPERPARAMS_REC_TRANSFO)


Adagrad_H = AdagradH()

# ----------------------------- SGD ----------------------------- #

SGD_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'momentum', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (-8, -1)},
    {'name': 'dampening', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'nesterov', 'type': 'discrete', 'domain': (0, 1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

SGD_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'momentum': lambda momentum: momentum},
    {'weight_decay': lambda weight_decay: 10 ** weight_decay},
    {'dampening': lambda dampening: dampening ** 3},
    {'nesterov': lambda nesterov: nesterov},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

SGD_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'momentum': lambda momentum: momentum},
    {'weight_decay': lambda weight_decay: np.log10(weight_decay)},
    {'dampening': lambda dampening: np.log(dampening) / np.log(3.)},
    {'nesterov': lambda nesterov: nesterov},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]

SGD_HYPERPARAMS_CONSTRAINTS = [
    # ensures we don't have `nesterov` and (`momentum` <= 0 or `dampening` != 0)
    {'name': 'nest_damp_mom',
     'constraint': 'np.logical_and(x[:,4] > 0, np.logical_or(x[:,1] <= 0, x[:,3] != 0))'}
]


class SGDH(OptScheduledParamSpace):

    def __init__(self):
        super(SGDH, self).__init__(SGD_HYPERPARAMS_BOUNDS, SGD_HYPERPARAMS_TRANSFO,
                                   SGD_HYPERPARAMS_CONSTRAINTS, SGD_HYPERPARAMS_REC_TRANSFO)


SGD_H: SGDH = SGDH()

# ----------------------------- Rprop ----------------------------- #

Rprop_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'eta1', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'eta2', 'type': 'continuous', 'domain': (1, 3)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

Rprop_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'eta1': lambda eta1: eta1},
    {'eta2': lambda eta2: eta2},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

Rprop_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'eta1': lambda eta1: eta1},
    {'eta2': lambda eta2: eta2},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]

Rprop_HYPERPARAMS_CONSTRAINTS = [
    # ensures 0.0 < eta1 < 1.0 < eta2
    {'name': 'eta1', 'constraint': 'np.logical_not(np.logical_and(0 < x[:, 1], x[:, 1] < 1))'},
    {'name': 'eta2', 'constraint': 'np.logical_not(1 < x[:, 2])'}
]


class RpropH(OptScheduledParamSpace):

    def __init__(self):
        super(RpropH, self).__init__(Rprop_HYPERPARAMS_BOUNDS, Rprop_HYPERPARAMS_TRANSFO,
                                     Rprop_HYPERPARAMS_CONSTRAINTS, Rprop_HYPERPARAMS_REC_TRANSFO)

    def get_true_opt_kwargs(self, all_params_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        all_params_kwargs['etas'] = (all_params_kwargs.pop('eta1'), all_params_kwargs.pop('eta2'))
        return all_params_kwargs

    def get_search_opt_kwargs(self, all_params_kwargs) -> Dict[str, Any]:
        all_params_kwargs['eta1'], all_params_kwargs['eta2'] = all_params_kwargs.pop('etas')
        return self.reorder_params(all_params_kwargs)


Rprop_H: RpropH = RpropH()

# ----------------------------- RMSprop ----------------------------- #

RMSprop_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'momentum', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'alpha', 'type': 'continuous', 'domain': (-6, -.5)},
    {'name': 'centered', 'type': 'discrete', 'domain': (0, 1)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (-8, -1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

RMSprop_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'momentum': lambda momentum: momentum},
    {'alpha': lambda alpha: 1 - 10 ** alpha},
    {'centered': lambda eta2: eta2},
    {'weight_decay': lambda weight_decay: 10 ** weight_decay},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

RMSprop_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'momentum': lambda momentum: momentum},
    {'alpha': lambda alpha: np.log10(1 - alpha)},
    {'centered': lambda eta2: eta2},
    {'weight_decay': lambda weight_decay: np.log10(weight_decay)},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]


class RMSpropH(OptScheduledParamSpace):

    def __init__(self):
        super(RMSpropH, self).__init__(RMSprop_HYPERPARAMS_BOUNDS, RMSprop_HYPERPARAMS_TRANSFO,
                                       rec_transfos=RMSprop_HYPERPARAMS_REC_TRANSFO)


RMSprop_H: RMSpropH = RMSpropH()

# ----------------------------- ASGD ----------------------------- #

ASGD_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, -.5)},
    {'name': 'lambd', 'type': 'continuous', 'domain': (-6, -1)},
    {'name': 'alpha', 'type': 'continuous', 'domain': (.1, 1)},
    {'name': 't0', 'type': 'continuous', 'domain': (0, 3)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (-7, -.5)}  # /!\ this is for scheduling (with `ExponentialLR`)
]

ASGD_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'lambd': lambda lambd: 10 ** lambd},
    {'alpha': lambda alpha: alpha},
    {'t0': lambda t0: int(10 ** t0)},
    {'gamma': lambda gamma: 1 - 10 ** gamma}
]

ASGD_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'lambd': lambda lambd: np.log10(lambd)},
    {'alpha': lambda alpha: alpha},
    {'t0': lambda t0: np.log10(t0)},
    {'gamma': lambda gamma: np.log10(1 - gamma)}
]


class ASGDH(OptScheduledParamSpace):

    def __init__(self):
        super(ASGDH, self).__init__(ASGD_HYPERPARAMS_BOUNDS, ASGD_HYPERPARAMS_TRANSFO,
                                    rec_transfos=ASGD_HYPERPARAMS_REC_TRANSFO)


ASGD_H: ASGDH = ASGDH()

# ----------------------------- Adamos ----------------------------- #

ADAMOS_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, 0)},
    {'name': 'mu', 'type': 'continuous', 'domain': (-3, -0.05)},
    {'name': 'C_gamma', 'type': 'continuous', 'domain': (0.5, 1)},
    {'name': 'alpha_decay', 'type': 'continuous', 'domain': (0.02, 0.5)},
    {'name': 'mu_decay', 'type': 'continuous', 'domain': (0.8, 1.2)},
    {'name': 'gamma2_decay', 'type': 'continuous', 'domain': (0.2, 0.8)}
]

ADAMOS_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'mu': lambda mu: 1 - 10 ** mu},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]

ADAMOS_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'mu': lambda mu: np.log10(1 - mu)},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]


class AdamosH(OptParamSpace):

    def __init__(self):
        super(AdamosH, self).__init__(ADAMOS_HYPERPARAMS_BOUNDS, ADAMOS_HYPERPARAMS_TRANSFO,
                                      rec_transfos=ADAMOS_HYPERPARAMS_REC_TRANSFO)


Adamos_H = AdamosH()

# ----------------------------- CAdam ----------------------------- #

CADAM_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, 0)},
    {'name': 'beta', 'type': 'continuous', 'domain': (0.001, 0.999)},
    {'name': 'mu', 'type': 'continuous', 'domain': (-3, -0.05)},
    {'name': 'C_gamma', 'type': 'continuous', 'domain': (0.5, 1)},
    {'name': 'alpha_decay', 'type': 'continuous', 'domain': (0.02, 0.5)},
    {'name': 'mu_decay', 'type': 'continuous', 'domain': (0.8, 1.2)},
    {'name': 'gamma2_decay', 'type': 'continuous', 'domain': (0.2, 0.8)}
]

CADAM_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'beta': lambda beta: beta},
    {'mu': lambda mu: 1 - 10 ** mu},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]

CADAM_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'beta': lambda beta: beta},
    {'mu': lambda mu: np.log10(1 - mu)},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]


class CAdamH(OptParamSpace):

    def __init__(self):
        super(CAdamH, self).__init__(CADAM_HYPERPARAMS_BOUNDS, CADAM_HYPERPARAMS_TRANSFO,
                                     rec_transfos=CADAM_HYPERPARAMS_REC_TRANSFO)


CAdam_H = CAdamH()

# ----------------------------- TrueCAdam ----------------------------- #

TRUECADAM_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, 0)},
    {'name': 'beta', 'type': 'continuous', 'domain': (0.001, 0.999)},
    {'name': 'mu', 'type': 'continuous', 'domain': (-3, -0.05)},
    {'name': 'C_gamma', 'type': 'continuous', 'domain': (0.5, 1)},
    {'name': 'alpha_decay', 'type': 'continuous', 'domain': (0.02, 0.5)},
    {'name': 'mu_decay', 'type': 'continuous', 'domain': (0.8, 1.2)},
    {'name': 'gamma2_decay', 'type': 'continuous', 'domain': (0.2, 0.8)}
]

TRUECADAM_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'beta': lambda beta: beta},
    {'mu': lambda mu: 1 - 10 ** mu},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]

TRUECADAM_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'beta': lambda beta: beta},
    {'mu': lambda mu: np.log10(1 - mu)},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]


class TrueCAdamH(OptParamSpace):

    def __init__(self):
        super(TrueCAdamH, self).__init__(TRUECADAM_HYPERPARAMS_BOUNDS, TRUECADAM_HYPERPARAMS_TRANSFO,
                                         rec_transfos=TRUECADAM_HYPERPARAMS_REC_TRANSFO)


TrueCAdam_H = TrueCAdamH()

# ----------------------------- TrueCAdam ----------------------------- #

TRUECADAM2_HYPERPARAMS_BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (-5, 0)},
    {'name': 'beta', 'type': 'continuous', 'domain': (0.001, 0.999)},
    {'name': 'mu', 'type': 'continuous', 'domain': (-3, -0.05)},
    {'name': 'C_gamma', 'type': 'continuous', 'domain': (0.5, 1)},
    {'name': 'alpha_decay', 'type': 'continuous', 'domain': (0.02, 0.5)},
    {'name': 'mu_decay', 'type': 'continuous', 'domain': (0.8, 1.2)},
    {'name': 'gamma2_decay', 'type': 'continuous', 'domain': (0.2, 0.8)}
]

TRUECADAM2_HYPERPARAMS_TRANSFO = [
    {'lr': lambda lr: 10 ** lr},
    {'beta': lambda beta: beta},
    {'mu': lambda mu: 1 - 10 ** mu},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]

TRUECADAM2_HYPERPARAMS_REC_TRANSFO = [
    {'lr': lambda lr: np.log10(lr)},
    {'beta': lambda beta: beta},
    {'mu': lambda mu: np.log10(1 - mu)},
    {'C_gamma': lambda C_gamma: C_gamma},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'mu_decay': lambda mu_decay: mu_decay},
    {'gamma2_decay': lambda gamma2_decay: gamma2_decay}
]


class TrueCAdam2H(OptParamSpace):

    def __init__(self):
        super(TrueCAdam2H, self).__init__(TRUECADAM2_HYPERPARAMS_BOUNDS, TRUECADAM2_HYPERPARAMS_TRANSFO,
                                          rec_transfos=TRUECADAM2_HYPERPARAMS_REC_TRANSFO)


TrueCAdam2_H = TrueCAdam2H()

# ----------------------------- NASA ----------------------------- #

NASA_HYPERPARAMS_BOUNDS = [
    {'name': 'a', 'type': 'continuous', 'domain': (.1, 10)},
    {'name': 'b', 'type': 'continuous', 'domain': (.1, 10)},
    {'name': 'beta', 'type': 'continuous', 'domain': (.1, 10)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0.5, 1.)}
]

NASA_HYPERPARAMS_TRANSFO = [
    {'a': lambda a: a},
    {'b': lambda b: b},
    {'beta': lambda beta: beta},
    {'gamma': lambda gamma: gamma}
]

NASA_HYPERPARAMS_REC_TRANSFO = [
    {'a': lambda a: a},
    {'b': lambda b: b},
    {'beta': lambda beta: beta},
    {'gamma': lambda gamma: gamma}
]


class NASAH(OptParamSpace):

    def __init__(self):
        super(NASAH, self).__init__(NASA_HYPERPARAMS_BOUNDS, NASA_HYPERPARAMS_TRANSFO,
                                    rec_transfos=NASA_HYPERPARAMS_REC_TRANSFO)


NASA_H = NASAH()

# ----------------------------- SCGD ----------------------------- #

SCGD_HYPERPARAMS_BOUNDS = [
    {'name': 'alpha_start', 'type': 'continuous', 'domain': (-4, 0)},
    {'name': 'beta_start', 'type': 'continuous', 'domain': (-3, -.5)},
    {'name': 'alpha_decay', 'type': 'continuous', 'domain': (.4, .95)},
    {'name': 'beta_decay', 'type': 'continuous', 'domain': (0.2, 0.8)}
]

SCGD_HYPERPARAMS_TRANSFO = [
    {'alpha_start': lambda alpha_start: 10 ** alpha_start},
    {'beta_start': lambda beta_start: 1 - 10 ** beta_start},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'beta_decay': lambda beta_decay: beta_decay}
]

SCGD_HYPERPARAMS_REC_TRANSFO = [
    {'alpha_start': lambda alpha_start: np.log10(alpha_start)},
    {'beta_start': lambda beta_start: np.log10(1 - beta_start)},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'beta_decay': lambda beta_decay: beta_decay}
]


class SCGDH(OptParamSpace):

    def __init__(self):
        super(SCGDH, self).__init__(SCGD_HYPERPARAMS_BOUNDS, SCGD_HYPERPARAMS_TRANSFO,
                                    rec_transfos=SCGD_HYPERPARAMS_REC_TRANSFO)


SCGD_H = SCGDH()

# ----------------------------- ASCGD ----------------------------- #

ASCGD_HYPERPARAMS_BOUNDS = [
    {'name': 'alpha_start', 'type': 'continuous', 'domain': (-4, 0)},
    {'name': 'beta_start', 'type': 'continuous', 'domain': (-3, -.5)},
    {'name': 'alpha_decay', 'type': 'continuous', 'domain': (.4, .95)},
    {'name': 'beta_decay', 'type': 'continuous', 'domain': (0.25, 0.85)}
]

ASCGD_HYPERPARAMS_TRANSFO = [
    {'alpha_start': lambda alpha_start: 10 ** alpha_start},
    {'beta_start': lambda beta_start: 1 - 10 ** beta_start},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'beta_decay': lambda beta_decay: beta_decay}
]

ASCGD_HYPERPARAMS_REC_TRANSFO = [
    {'alpha_start': lambda alpha_start: np.log10(alpha_start)},
    {'beta_start': lambda beta_start: np.log10(1 - beta_start)},
    {'alpha_decay': lambda alpha_decay: alpha_decay},
    {'beta_decay': lambda beta_decay: beta_decay}
]


class ASCGDH(OptParamSpace):

    def __init__(self):
        super(ASCGDH, self).__init__(ASCGD_HYPERPARAMS_BOUNDS, ASCGD_HYPERPARAMS_TRANSFO,
                                     rec_transfos=ASCGD_HYPERPARAMS_REC_TRANSFO)


ASCGD_H = ASCGDH()


# ----------------------------------------------------------------------------------------------------#

def get_optimizer(optimizer_name):
    try:
        return getattr(sys.modules[__name__], optimizer_name.split('-')[0] + '_H')
    except ValueError as e:
        print(f"Given optimizer {optimizer_name} cannot be tuned as we don't know its hyperparameters")
        raise e
    except Exception as e:
        raise e


def BO_objective(hyperparams: np.ndarray, *args, **kwargs) -> float:
    """ Run a BO experiment with fixed hyperparamers

     Args:
        hyperparams: array of hyperparemeters for optimizer (and scheduler)
        kwargs:
            - `k`: config  | `v`: a `Namespace` object containing configurations for the BO experiment
            - `k`: opt_h  | `v`: a `OptHyperparamter` associated to the current optimizer

    Returns:
        mean_last_regret: the immediate regret obtained at the end of the BO averaged over the different trials
    """

    # catch kwargs
    config: Namespace = kwargs['config']
    opt_h: OptParamSpace = kwargs['opt_h']
    assert hyperparams.ndim == 2
    # get hyperparameters in search space
    hyperparams = hyperparams.flatten()

    # recover real hyperparameters (apply transformations)
    all_params_kwargs: Dict[str, Any] = opt_h.get_real_params(hyperparams)

    scheduler, scheduler_kwargs = opt_h.get_scheduler(all_params_kwargs)
    optimizer_kwargs = opt_h.get_true_opt_kwargs(all_params_kwargs)
    print('hyper-parameters used:', optimizer_kwargs)
    print('scheduler used:', (scheduler, scheduler_kwargs))

    config.optimizer_kwargs = optimizer_kwargs
    print(config.optimizer_kwargs)
    config.scheduler = scheduler
    config.scheduler_kwargs = scheduler_kwargs
    mean_last_regret = run_exp_bo(config)
    return mean_last_regret


if __name__ == '__main__':
    config_ = get_tuning_config()
    root_result_dir_memo = config_.root_result_dir
    config_.root_result_dir = 'results/test_best'
    test_best_dir = exp_folder_rel_path(config_)
    test_best_exps = query_exps(test_best_dir, {'covar': config_.covar})
    if len(test_best_exps) > 0:
        print('\n---  Experiment has already been tested with best parameters  ---\n')

    else:
        config_.root_result_dir = root_result_dir_memo

        num_cpu = len(os.sched_getaffinity(0))
        torch.set_num_threads(max(1, (3 * num_cpu) // 4))
        torch.set_num_interop_threads(max(1, num_cpu - torch.get_num_threads()))
        print(torch.get_num_threads(), torch.get_num_interop_threads())

        opt_hyp_ = get_optimizer(config_.optimizer)

        kwargs_ = {'config': config_, 'opt_h': opt_hyp_}

        custom_objective = CustomSingleObjective(
            func=BO_objective,
            **kwargs_
        )
        initial_design_numdata_: int = config_.initial_design_numdata
        max_iter_: int = config_.max_iter
        X, Y = get_X_Y_tuning(config_)
        X = np.array(list(map(
            lambda params_dict: opt_hyp_.get_list_params_from_dict(opt_hyp_.get_search_params(params_dict)), X
        )))
        assert X.shape[0] == Y.shape[0]
        budget = Y.shape[0]
        if budget == 0:
            X, Y = None, None
        elif 0 < budget <= initial_design_numdata_:
            initial_design_numdata_ -= budget
        else:
            budget -= initial_design_numdata_
            initial_design_numdata_ = 0
            max_iter_ -= budget
        if Y is not None:
            print(
                f"There are {Y.shape[0]} existing experiments for tuning with this configuration stored in {get_result_dir(config_)}, so"
                f" we use {initial_design_numdata_} initial data points and {max_iter_} acquisition steps")
        bopt = GPyOpt.methods.BayesianOptimization(
            f=custom_objective.func,
            domain=opt_hyp_.bounds,
            constraints=opt_hyp_.constraints,
            acquisition_type='LCB',
            initial_design_numdata=initial_design_numdata_,
            X=X,
            Y=Y,
            num_cores=1,
            de_duplication=True
        )
        bopt.run_optimization(max_iter=max_iter_, verbosity=config_.verbose, eps=-1)
        if len(bopt.X) != (config_.initial_design_numdata + config_.max_iter):
            raise ValueError(
                f"Wrong number of acquisitions, expected {config_.initial_design_numdata + config_.max_iter}, got {len(bopt.X)}")
        print(bopt.x_opt)
        opt_hyper = opt_hyp_.get_real_params(bopt.x_opt.flatten())
        if config_.save:
            print('save best hyperparams')
            save_best_hyperparams(opt_hyper, config_)
        print(opt_hyper)
