import argparse
from argparse import ArgumentParser

from utils.utils_parsing import parse_dict, parse_list, get_config_from_parser


# @formatter:off
def get_common_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(description='Run configuration')
    parser.register('type', dict, parse_dict)

    parser.add_argument('--input_dim', type=int, default=1, help='input space dimensionality')
    parser.add_argument('--num_initial', type=int, default=3, help='num. initial observed pairs (X, Y)')
    parser.add_argument('--do_normalize', type=int, default=1, choices=[0, 1], help='normalize the Xs in training dataset for acquisition function optimizqtion')
    parser.add_argument('--do_standardize', type=int, default=1, choices=[0, 1], help='normalize the Ys in training dataset for acquisition function optimization')
    parser.add_argument('--test_func', type=str, default='Levy', help='ID for synthetic test_function to optimize')
    parser.add_argument('--lb_offset', type=float, default=0, help='Lower bound offset fo test function domain')
    parser.add_argument('--ub_offset', type=float, default=0, help='Upper bound offset fo test function domain')
    parser.add_argument('--covar', type=str, default='matern-5/2', help='Covariance module for GP')
    parser.add_argument('--covar_kw', type=dict, default='{}', help='kwargs of covariance module for GP')
    parser.add_argument('--negate', type=int, default=0, choices=[0, 1], help='consider maximization of -test_func instead of maximization of +test_functino ')
    parser.add_argument('--noise_free', type=int, default=0, choices=[0, 1], help='get noise-free evalution of the black-box function')
    parser.add_argument('--num_acq_steps', type=int, default=5, help='number of acquisition steps')
    parser.add_argument('--num_trials', type=int, default=5, help='number of trials of Bayesian optimisation (number of times experiment is rerun)')
    parser.add_argument('--seed', type=int, default=0, help='seed for the experiment')

    parser.add_argument('--acq_func', type=str, default='qExpectedImprovement', help='Name of the acquisition function')
    parser.add_argument('--acq_func_kwargs', type=dict, default='{}', help='string-specified dictionary for acquisition function (e.g. in compositional ones: K_g ,...)')
    parser.add_argument('--num_MC_samples_acq', type=int, default=100, help='number of samples for MC acquisition loss estimation')
    parser.add_argument('--num_starts', type=int, default=10, help='number of starts (multistart SGD) for optimization of acquisition function')
    parser.add_argument('--num_raw_samples', type=int, default=100, help='number of raw starts considered among which `num_starts` will be selected')
    parser.add_argument('--q', type=int, default=1, help='number of new points acquired each time')
    parser.add_argument('--optimizer', type=str, default='adam', help='ID for the torch optimizer used to optimize the acquisition function')
    parser.add_argument('--num_opt_steps', type=int, default=50, help='number of optimization steps')
    parser.add_argument('--early_stop', type=int, default=0, help='Whether to allow early stop in optimization of the acquisition function')

    parser.add_argument('--cuda', type=int, default=None, help='cuda id (e.g. 0) - if not specificed then use cpu only')

    parser.add_argument('--save', type=int, default=1, choices=[0, 1], help='save results')
    parser.add_argument('--save_trajs', type=int, default=0, choices=[0, 1], help='save optimization trajectories')
    parser.add_argument('--save_losses', type=int, default=0, choices=[0, 1], help='save optimizer loss')
    parser.add_argument('--root_result_dir', type=str, default='results', help='path to results directory')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='re-run/overwrite existing experiments')
    parser.add_argument('--verbose', type=float, default=1, help='level of verbosity')

    return parser
# @formatter:on


# @formatter:off
def get_exp_config() -> argparse.Namespace:
    parser = get_common_parser()

    parser.add_argument('--optimizer_kwargs', type=dict, default='{}', help='string-specified dictionary for optimizer options (lr, weight_decay,...)')
    parser.add_argument('--scheduler', type=str, default=None, help='name of torch optimizer scheduler to use (e.g. MultiStepLR, ExponentialLR...)')
    parser.add_argument('--scheduler_kwargs', type=dict, default={}, help='string-specified dictionary for scheduler')
    parser.add_argument('--ignore_existing', type=int, choices=(0,1), default=1, help='whether to ignore existing experiments with same configuration that have already been run')

    return get_config_from_parser(parser)


# @formatter:off
def get_tuning_config() -> argparse.Namespace:
    parser = get_common_parser()

    parser.add_argument('--initial_design_numdata', type=int, default=3, help='Number of points randomly picked to initialize hyperparameters tuning via BO')
    parser.add_argument('--max_iter', type=int, default=10, help='Number of points to acquire for hyperparameters tuning via BO')

    return get_config_from_parser(parser)
# @formatter:on


# @formatter:off
def get_test_best_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run configuration')
    parser.register('type', list, parse_list)

    parser.add_argument('--cuda', type=int, default=None, help='cuda id (e.g. 0) - if not specificed then use cpu only')


    parser.add_argument('--seed', type=int, default=0, help='seed for the experiment')

    parser.add_argument('--num_trials', type=int, default=16, help='number of trials of Bayesian optimisation (number of times experiment is rerun)')
    parser.add_argument('--do_random_search', type=int, default=0, choices=[0, 1], help='Whether to run experiment with random search')

    parser.add_argument('--test_func', type=str, default='Levy', help='ID for synthetic test function to optimize')
    parser.add_argument('--lb_offset', type=float, default=0, help='Lower bound offset fo test function domain')
    parser.add_argument('--ub_offset', type=float, default=0, help='Upper bound offset fo test function domain')
    parser.add_argument('--covar', type=str, default='matern-5/2', help='Covariance module for GP')
    parser.add_argument('--input_dim', type=int, default=1, help='input space dimensionality')
    parser.add_argument('--acq_func', type=str, default='qExpectedImprovement', help='Name of the acquisition function')
    parser.add_argument('--q', type=int, default=1, help='number of new points acquired each time')

    parser.add_argument('--selected_optimizers', type=list, default="[]", help='string-encoded list containg optimizers to evaluate')
    parser.add_argument('--excluded_optimizers', type=list, default="[]", help='string-encoded list containg optimizers that should not be evaluated')

    parser.add_argument('--root_tuning_dir', type=str, default="results/hyperparams_tunings", help='path where hyperparamaters tunings results have been stored')
    parser.add_argument('--root_test_dir', type=str, default="results/test_best", help='path where test results should be stored')
    parser.add_argument('--save_trajs', type=int, default=0, choices=[0, 1], help='save optimization trajectories')
    parser.add_argument('--save_losses', type=int, default=0, choices=[0, 1], help='save optimizer loss')

    parser.add_argument('--generalize', type=int, default=0, help='Whether to test hyperparameters that have been tuned on the given objective function or'
                                                                  'testing another objective function')
    parser.add_argument('--verbose', type=float, default=1, help='level of verbosity')

    return get_config_from_parser(parser)
# @formatter:on


# @formatter:off
def get_test_time_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run configuration')
    parser.register('type', list, parse_list)
    parser.register('type', dict, parse_dict)

    parser.add_argument('--cuda', type=int, default=None, help='cuda id (e.g. 0) - if not specificed then use cpu only')


    parser.add_argument('--seed', type=int, default=0, help='seed for the experiment')

    parser.add_argument('--num_trials', type=int, default=16, help='number of trials of Bayesian optimisation (number of times experiment is rerun)')
    parser.add_argument('--do_random_search', type=int, default=0, choices=[0, 1], help='Whether to run experiment with random search')

    parser.add_argument('--test_func', type=str, default='Levy', help='ID for synthetic test function to optimize')
    parser.add_argument('--lb_offset', type=float, default=0, help='Lower bound offset fo test function domain')
    parser.add_argument('--ub_offset', type=float, default=0, help='Upper bound offset fo test function domain')
    parser.add_argument('--covar', type=str, default='matern-5/2', help='Covariance module for GP')
    parser.add_argument('--covar_kw', type=dict, default='{}', help='Covariance kwargs for GP')
    parser.add_argument('--input_dim', type=int, default=1, help='input space dimensionality')
    parser.add_argument('--acq_func', type=str, default='qExpectedImprovement', help='Name of the acquisition function')
    parser.add_argument('--q', type=int, default=1, help='number of new points acquired each time')

    parser.add_argument('--selected_optimizers', type=list, default="[]", help='string-encoded list containg optimizers to evaluate')
    parser.add_argument('--excluded_optimizers', type=list, default="[]", help='string-encoded list containg optimizers that should not be evaluated')

    parser.add_argument('--root_test_best_dir', type=str, default="results/test_best", help='path where test best results have been stored')
    parser.add_argument('--root_test_time_dir', type=str, default="results/test_time", help='path where test time results should be stored')

    parser.add_argument('--verbose', type=float, default=1, help='level of verbosity')

    return get_config_from_parser(parser)
# @formatter:on