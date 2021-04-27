from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
import pandas as pd
import numpy as np

from hebo.optimizers.hebo import HEBO
from hebo.optimizers.util import parse_space_from_bayesmark


class HEBOOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
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
        space    = parse_space_from_bayesmark(api_config)
        self.opt = HEBO(space, rand_sample = 32, verbose = True) # 4 iterations of random search with `n_suggestions = 8`

    def suggest(self, n_suggestions=1):
        print('Start optimization', flush = True)
        rec     = self.opt.suggest(n_suggestions)
        x_guess = [row.to_dict() for _, row in rec.iterrows()]
        for guess in x_guess:
            for name in guess:
                if self.api_config[name]['type'] == 'int':
                    guess[name] = int(guess[name])
        return x_guess

    def observe(self, X, y):
        self.opt.observe(pd.DataFrame(X), np.array(y).reshape(-1, 1))
        print('Observe finished', flush = True)

if __name__ == "__main__":
    experiment_main(HEBOOptimizer)
