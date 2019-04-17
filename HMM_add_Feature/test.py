import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from HMM_add_Feature.Dictionary import Dictionary

class LBFGS:

    def __init__(self, e, f, k):
        self.expect_count = e
        self.feature = f
        self.k = k

    def get_probabilities(self, W):
        res = []
        for index, feature_state in enumerate(self.feature):
            feature_state = np.array(feature_state, dtype=np.float64)
            Z_state = feature_state.dot(W)
            e_Z = np.exp(Z_state - np.max(Z_state, axis=0, keepdims=True))
            A = (e_Z + 1) / (e_Z.sum(axis=0) + len(feature_state))
            res.append(A)
        return res

    def counting_theta_feature(self, W, feature):
        """
        w (vecto)
        feature_array : each feature is vector which have dimension = dimension of w
        return sum(theta(w)*feature)
        :return:
        """
        matrix_probabilities = self.get_probabilities(W)
        len_theta_w_state = len(feature[0][0])
        theta_w = []
        for i, feature_state in enumerate(feature):
            theta_w_state = np.zeros(len_theta_w_state, dtype=np.float64)
            for j, value in enumerate(feature_state):
                theta_w_state += matrix_probabilities[i][j] * np.array(value, dtype=np.float64)
            theta_w.append(theta_w_state)
        return theta_w

    def grad_loss_w_e(self, matrix_count_e, W, feature, k):
        """

        :param matrix_count_e:
        :param W:
        :param feature:
        :param k:
        :return:
        """
        matrix_denta_w_dct = self.denta_w_dct(W, feature)
        len_vector_feature = len(feature[0][0])
        grad_loss = np.zeros(len_vector_feature, dtype=np.float64)
        for state, feature_state in enumerate(feature):
            for i, value in enumerate(matrix_count_e[state]):
                grad_loss += value * np.array(matrix_denta_w_dct[state][i], dtype=np.float64)
        grad_loss -= 2 * k * W
        return grad_loss

    def denta_w_dct(self, W, feature):
        """
        return matrix loss w of each feature
        :return:
        """
        matrix_theta = self.counting_theta_feature(W, feature)
        matrix_loss = []
        for i, feature_state in enumerate(feature):
            matrix_loss_state = []
            for j, value in enumerate(feature_state):
                matrix_loss_state.append(np.array(value, dtype=np.float64) - matrix_theta[i])
            matrix_loss.append(matrix_loss_state)
        return matrix_loss

    def loss_e_w(self, w):
        probabilities = self.get_probabilities(w)
        value = np.sum(self.expect_count*np.log(probabilities)) - self.k*norm(w)
        return value

    # The objective function and the gradient.
    def quadratic(self, w):
        value = self.loss_e_w(w)
        grad = self.grad_loss_w_e(self.expect_count, w, self.feature, self.k)
        return value, grad

    def lbgfs(self, w):
        optim_results = tfp.optimizer.lbfgs_minimize(
            self.quadratic, initial_position=w, num_correction_pairs=10,
            tolerance=1e-8)
        return optim_results

if __name__ == "__main__":

    diction = Dictionary()

    W_transition = np.array([7, 3, 7, 3], dtype=np.float64)
    vocab_t = diction.gen_feature_basic_t()
    k = 0.2
    e = np.array([[5, 6], [4, 1]])
    lbgfs = LBFGS(e, vocab_t, k)
    print(lbgfs.get_probabilities(W_transition))
    print(lbgfs.denta_w_dct(W_transition, vocab_t))
    print(lbgfs.counting_theta_feature(W_transition, vocab_t))
    print(lbgfs.grad_loss_w_e(e, W_transition, vocab_t, k))
    print(lbgfs.loss_e_w(W_transition))

