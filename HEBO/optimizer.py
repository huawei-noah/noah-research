"""Optimizer class for running autonomous black-box optimisation."""

import numpy as np
import pandas as pd
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.util import parse_space_from_bayesmark


class HEBOOptimizer(AbstractOptimizer):
    """HEBO Black Box Optimizer Observe, Suggest class."""
    
    primary_import = "bayesmark"
    
    def __init__(self, api_config):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        space = parse_space_from_bayesmark(api_config)
        # 4 iterations of random search with `n_suggestions = 8`
        self.opt = HEBO(space, rand_sample=32, verbose=True)
    
    def suggest(self, n_suggestions=1):
        """Suggest n_suggestions params."""
        print('Start optimization', flush=True)
        rec = self.opt.suggest(n_suggestions)
        x_guess = [row.to_dict() for _, row in rec.iterrows()]
        for guess in x_guess:
            for name in guess:
                if self.api_config[name]['type'] == 'int':
                    guess[name] = int(guess[name])
        return x_guess
    
    def observe(self, X, y):
        """Observe parameters X and black-box evaluations y. Where X.shape[0]==y.shape[0]."""
        self.opt.observe(pd.DataFrame(X), np.array(y).reshape(-1, 1))
        print('Observe finished', flush=True)


if __name__ == "__main__":
    experiment_main(HEBOOptimizer)
