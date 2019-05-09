import numpy as np
from numpy.linalg import norm

class LogisticModel:

    def __init__(self, weight=None, feature=None):
        self.weight = weight
        self.feature = feature

    @staticmethod
    def get_probabilities(weight, feature):

        """

        :param weight:
        :param feature:
        :return:
        """
        feature = np.array(feature, dtype=np.float64)
        score = np.dot(weight, feature.T)
        score_exp = np.exp(score/np.max(score, axis=0, keepdims=True))
        probabilities_matrix = score_exp/score_exp.sum(axis=0)
        return probabilities_matrix

    def sum_probabilities_feature(self, weight, feature):
        """

        :param weight:
        :param feature:
        :return:<class 'numpy.ndarray'>
        """
        probabilites = self.get_probabilities(weight, feature)
        feature = np.array(feature, dtype=np.float64)
        sum_probabilities = np.dot(feature.T, probabilites)
        return sum_probabilities

    def difference_weight_feature(self, weight, feature):
        """

        :param weight:
        :param feature:
        :return:<class 'numpy.ndarray'>
        """
        sum_probabilities = self.sum_probabilities_feature(weight, feature)
        difference_weight_feature_matrix = np.array(feature) - sum_probabilities
        return difference_weight_feature_matrix

    def grad_weight(self, weight, feature, e_count, k):
        """

        :param weight:
        :param feature:
        :param e_count:
        :param k
        :return:
        """
        e_count = np.array(e_count, dtype=np.float64)
        e_count = e_count/np.max(e_count, axis=0, keepdims=True)
        # print("e_count", e_count)
        denta_weight_matrix = self.difference_weight_feature(weight, feature)
        # print("denta_weighr", denta_weight_matrix)
        # print("sum", np.dot(e_count, denta_weight_matrix))
        grad = np.dot(e_count, denta_weight_matrix) - 2*k*weight
        return grad

    def grad_weight_sum(self, W, feature, e_count, k):
        """

        :param weight:
        :param feature:
        :param e_count:
        :param k:
        :return:
        """
        grad = np.zeros(W.shape)
        for index, feature_state in enumerate(feature):
            grad += self.grad_weight(W, feature_state, e_count[index], k)
        grad += 2*k*W
        return grad

    def loss_function(self, weight, feature, e_count, k):
        """

        :param weight:
        :param feature:
        :param e_count:
        :param k:
        :return:
        """
        probabilities = self.get_probabilities(weight, feature)
        e_count = np.array(e_count, dtype=np.float64)
        e_count = e_count/np.max(e_count, axis=0, keepdims=True)
        loss = np.dot(e_count, np.log(probabilities)) - k*norm(weight)
        return loss

    def loss_function_sum(self, W, feature, e_count, k):
        """

        :param W:
        :param feature:
        :param e_count:
        :param k:
        :return:
        """
        loss = 0
        for index, feature_state in enumerate(feature):
            loss_state = self.loss_function(W, feature_state, e_count[index], k)
            loss += loss_state
        loss += k*norm(W)
        return loss

    def has_converged(self, theta_new, grad_weight_new, stop_point):
        """

        :param theta_new:
        :param grad_weight_new:
        :param stop_point
        :return:
        """
        return np.linalg.norm(grad_weight_new)/len(theta_new) < stop_point

    def gradient_descent_momentum(self, weight_init, feature, e_count, k, eta, gamma=0.9, max_iterations=20, stop_point=1e-3):
        """

        :param weight_init:
        :param feature:
        :param e_count
        :param k
        :param eta: learning_rate
        :param gamma:
        :param max_iterations
        :param stop_point
        :return:
        """
    # Suppose we want to store history of theta
        theta = [weight_init]
        v_old = np.zeros_like(weight_init)
        for it in range(1, max_iterations):
            v_new = gamma * v_old + eta * self.grad_weight(theta[-1], feature, e_count, k)
            weight_new = theta[-1] - v_new
            grad_weight_new = self.grad_weight(weight_new, feature, e_count, k)
            if self.has_converged(weight_new, grad_weight_new, stop_point):
                break
            theta.append(weight_new)
            v_old = v_new
        return theta[-1]

    def gradient_descent_momentum_sum(self, weight_init, feature, e_count, k, eta, gamma=0.9, max_iterations=20, stop_point=1e-3):
        """

        :param weight_init:
        :param feature:
        :param e_count:
        :param k:
        :param eta:
        :param gamma:
        :param max_iterations:
        :param stop_point:
        :return:
        """
        theta = [weight_init]
        v_old = np.zeros_like(weight_init)
        for it in range(1, max_iterations):
            v_new = gamma * v_old + eta * self.grad_weight_sum(theta[-1], feature, e_count, k)
            weight_new = theta[-1] - v_new
            grad_weight_new = self.grad_weight_sum(weight_new, feature, e_count, k)
            if self.has_converged(weight_new, grad_weight_new, stop_point):
                break
            theta.append(weight_new)
            v_old = v_new
        return theta[-1]

    def gradient_descent(self, w_init, feature, e_count, k, eta, stop_point=1e-3, max_iteration=10):
        """

        :param w_init:
        :param feature:
        :param e_count:
        :param k:
        :param eta:
        :param stop_point
        :param max_iteration
        :return:
        """
        w = [w_init]
        for it in range(1, max_iteration):
            w_new = w[-1] - eta * self.grad_weight(w[-1], feature, e_count, k)
            grad_w_new = self.grad_weight(w_new, feature, e_count, k)
            print(self.loss_function(w_new, feature, e_count, k))
            if np.linalg.norm(grad_w_new) / len(w_new) < stop_point:
                break
            w.append(w_new)
        return w[-1]











if __name__ == "__main__":

    weight = np.array([0.9, 0.1, 0.5, 0.3])
    e = [[84558176.20886347, 34353113.57229507], [84558176.20886347, 34353113.57229507]]
    print(weight)
    feature = [[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]]
    logistic = LogisticModel()
    print("1", logistic.get_probabilities(weight, feature[0]))
    # print("2", logistic.sum_probabilities_feature(weight, feature))
    # print("3", logistic.difference_weight_feature(weight, feature))
    print("4", logistic.grad_weight_sum(weight, feature, e, 0.1))
    # print("5", logistic.loss_function(weight, feature, e, 0.1))
    # print("6", logistic.gradient_descent(weight, feature, e, 0.1, 0.0000000001))
