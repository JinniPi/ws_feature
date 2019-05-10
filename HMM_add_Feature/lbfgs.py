import tensorflow_probability as tfp
import tensorflow as tf
from HMM_add_Feature.logistic_model import LogisticModel
lg = LogisticModel()

class LBFGS:

    def __init__(self, f, k, e=None):
        self.expect_count = e
        self.feature = f
        self.k = k

    def set_e(self, e):
        self.expect_count = e
        return 0

    # The objective function and the gradient.
    def quadratic(self, w):
        value = lg.loss_function(w, self.feature, self.expect_count, self.k)
        grad = lg.grad_weight(w, self.feature, self.expect_count, self.k)
        return value, grad

    def lbgfs(self, w):
        optimize_results = tfp.optimizer.lbfgs_minimize(
            self.quadratic, initial_position=w, num_correction_pairs=10,
            tolerance=1e-3, max_iterations=10)
        with tf.Session() as session:
            results = session.run(optimize_results)
            assert (results.converged)
        return results.position

    def quadratic_sum(self, w):
        value = lg.loss_function_sum(w, self.feature, self.expect_count, self.k)
        grad = lg.grad_weight_sum(w, self.feature, self.expect_count, self.k)
        return value, grad

    def lbgfs_sum(self, w):
        optimize_results = tfp.optimizer.lbfgs_minimize(
            self.quadratic_sum, initial_position=w, num_correction_pairs=10,
            tolerance=1e-3, max_iterations=10)
        with tf.Session() as session:
            results = session.run(optimize_results)
            assert (results.converged)
        return results.position
