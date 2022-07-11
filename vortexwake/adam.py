import numpy as np


class Adam:
    """ Implementation of the Adam optimiser as described in :cite:t:`Kingma2015`.
    
    Parameters for the optimiser may be specified using a configuration dictionary.
    All parameters unspecified will be set to the default values below.

         - `alpha` = 1e-3, maximum step size parameter
         - `beta_1` = 0.9, first decay rate for momentum estimate
         - `beta_2`  = 0.999, second decay rate for momentum estimate
         - `eps` = 1e-8
         - `max_iter` = 10, maximum number of iterations in optimisation

    :param config: dictionary of parameters for the optimiser
    """

    def __init__(self, config=None):
        self.alpha = 1e-3
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-8
        self.max_iter = 10

        self.xt = 0  # initial parameter vector
        self.mt = 0  # initialise 1st moment vector
        self.vt = 0  # initialise 2nd moment vector
        self.t = 0  # initialise optimisation step
        self.f = 0
        self.fh = np.zeros(self.max_iter)
        self.xh = 0
        if config is not None:
            self.set_parameters_from_config(config)
            self.reset_initial_conditions()

    def reset_initial_conditions(self):
        """Reset properties of Adam optimiser.
        Set to 0 all of: parameter vector `xt`, moment estimates `mt`, `vt`, optimisation step `t`, objective function
        history `fh` and solution history `xh`.

        """
        self.xt = 0  # initial parameter vector
        self.mt = 0  # initialise 1st moment vector
        self.vt = 0  # initialise 2nd moment vector
        self.t = 0  # initialise optimisation step
        self.fh = np.zeros(self.max_iter)
        self.xh = 0

    def set_parameters_from_config(self, config):
        """Set parameters for the Adam optimisater using a configuration dictionary.
        Unspecified parameters are set to the defaults as in the initialisation of the class.

        :param config: dictionary of parameters for the optimiser
        """
        # learning rate
        self.alpha = config.get("alpha", 1e-3)
        #  exponential decay rates for moment estimates
        self.beta_1 = config.get("beta_1", 0.9)
        self.beta_2 = config.get("beta_2", 0.999)
        self.eps = config.get("eps", 1e-8)
        # maximum iterations for optimisation
        self.max_iter = config.get("max_iter", 10)
        self.reset_initial_conditions()

    def minimise(self, fun, x0, q0):
        """Minimise the objective function `fun` starting from an initial condition `q0` with initial guess `x0`
        for the optimisation.

        :param fun: objective function that takes a control signal like `x` and initial condition `q0`, and returns a
            tuple (objective, gradient)
        :param x0: initial guess for the optimisation
        :param q0: initial condition for the objective function

        :return: control signal along optimiser trajectory for which objective function is minimal
        """
        # self.reset_initial_conditions()
        self.f = 0
        self.xh = np.zeros((self.max_iter, len(x0)))
        self.xt = x0.copy()

        # todo: break loop for NaN values in solution or boundary exceeded?
        while self.not_converged():
            self.t = self.t + 1
            # get gradients w.r.t. objective
            self.f, gt = fun(self.xt, q0)

            # store solution
            self.fh[self.t - 1] = self.f
            self.xh[self.t - 1] = self.xt

            # update biased first moment estimate
            self.mt = self.beta_1 * self.mt + (1 - self.beta_1) * gt
            # update biased second moment estimate
            self.vt = self.beta_2 * self.vt + (1 - self.beta_2) * gt ** 2
            # compute bias-corrected 1st moment estimate
            mt_hat = self.mt / (1 - self.beta_1 ** self.t)
            # compute bias-corrected 2nd moment estimate
            vt_hat = self.vt / (1 - self.beta_2 ** self.t)
            self.xt = self.xt - self.alpha * mt_hat / (np.sqrt(vt_hat) + self.eps)

            # more efficient version??
            #     alphat = alpha * np.sqrt(1-beta_2**t) / (1-beta_1**t)
            #     xt = xt - alphat * mt / (np.sqrt(vt)+eps)

            print("Iterate {:d} - f: {:.3e}".format(self.t, self.f))
            if np.any(np.isnan(self.xh[1])):
                raise ValueError("NaN in solution.")

        return self.xh[np.argmin(self.fh)]

    def not_converged(self):
        """Test convergence of optimisation.
        Currently only tests for maximum number of iterations.

        :return: boolean, `True` as long as convergence is not reached
        """
        # todo: implement other convergence criterion
        return self.t < self.max_iter
