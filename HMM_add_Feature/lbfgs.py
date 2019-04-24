import tensorflow_probability as tfp
from HMM_add_Feature.logistic_model import LogisticModel
lg = LogisticModel()

class LBFGS:

    def __init__(self, e, f, k):
        self.expect_count = e
        self.feature = f
        self.k = k

    # The objective function and the gradient.
    def quadratic(self, w):
        value = lg.loss_function(w, self.feature, self.expect_count, self.k)
        grad = lg.grad_weight(w, self.feature, self.expect_count, self.k)
        return value, grad

    def lbgfs(self, w):
        optimize_results = tfp.optimizer.lbfgs_minimize(
            self.quadratic, initial_position=w, num_correction_pairs=10,
            tolerance=1e-3, max_iterations=10)
        return optimize_results
