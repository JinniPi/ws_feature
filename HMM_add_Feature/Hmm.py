"""class for HMM algorithm write by jinniPi"""
import numpy as np
from numpy.linalg import norm
import threading
import time
import math
from HMM_add_Feature.test import LBFGS

class HiddenMarkovModel:

    def __init__(self, states, w_transitions, w_emissions,
                 start_probabilities, vocab_feature_e, feature_t, vocab_number):
        self.states = states  # tập trạng thái
        # self.observations = observations  # tập quan sát
        self.w_transitions = w_transitions  # xs chuyển
        self.W_emissions = w_emissions  # ma trận trọng số xs sinh trạng thái
        # self.W_emissions_I = W_emissions_I # ma trận trọng số xs sinh trạng thái I
        self.start_probabilities = start_probabilities  # xs ban đầu
        self.vocab_feature_e = vocab_feature_e
        self.feature_t = feature_t
        self.vocab_number = vocab_number

    def get_w_transition(self):
        return self.w_transitions

    def get_matrix_transition(self):
        w_transition = self.get_w_transition()
        res = self.get_probabilities(w_transition, self.feature_t)
        return res

    def get_probabilities(self, weight, feature):
        """
        input weight matrix and feature matrix
        Each state has a corresponding weight _state.
        The w_state dimension is equal to the dimension of a feature f.
        :param weight:
        :param feature:
        :return:
        """
        res = []
        for state in self.states:
            feature_state_i = np.array(feature[state], dtype=np.float64)
            Z_state = feature_state_i.dot(weight)
            e_Z = np.exp(Z_state - np.max(Z_state, axis=0, keepdims=True))
            A = (e_Z + 1) / (e_Z.sum(axis=0) + len(feature[state]))
            res.append(A)
        return np.array(res)

    def get_start_probabilities(self):

        return np.array(self.start_probabilities)

    def get_w_emission(self):
        return self.W_emissions

    def get_matrix_emission(self):
        """
            Z = X.dot(W)
            X: matrix of feature shape: m.n
             m: number word of Vocab
             n: d of feature eg. [0,1,0] d = 3
            W.shape = (d,)
            Compute softmax values for each sets of scores in Z.
            each column of Z is a set of score.
           """

        w_emission = np.array(self.W_emissions, dtype=np.float64)
        res = self.get_probabilities(w_emission, self.vocab_feature_e)
        return res

    def forward_algorithm(self, observations_sequence, emission_matrix, transition_matrix, start_probabilities):

        # emission_matrix = self.get_matrix_emission()
        # transition_matrix = self.get_matrix_transition()
        # start_probabilities = self.get_start_probabilities()

        """

        :param observations_sequence:
        :param emission_matrix:
        :param transition_matrix:
        :param start_probabilities:
        :return:
        """
        forward_matrix = []
        
        for index_observation, observation in enumerate(observations_sequence):
            forward_array = []
            key_observation = list(observation.keys())[0]
            # tinh alpha
            if index_observation == 0:
                for index_state, state in enumerate(self.states):
                    alpha_i = start_probabilities[index_state] * \
                        emission_matrix[index_state][observation.get(key_observation)]
                    forward_array.append(alpha_i)
            else:
                alpha_previous_states = forward_matrix[-1]
                for index_state, state in enumerate(self.states):
                    alpha_i = 0
                    for index_previous_state, alpha_previous_state in enumerate(alpha_previous_states):
                        alpha_i += alpha_previous_state * \
                                   transition_matrix[index_previous_state][index_state]

                    alpha_i *= emission_matrix[index_state][observation.get(key_observation)]
                    forward_array.append(alpha_i)
            forward_matrix.append(forward_array)

        final_probabilities = 0
        last_forward_matrix = forward_matrix[-1]
        end_probabilities = list(map(lambda state: 1, self.states))
        for index, state in enumerate(self.states):
            final_probabilities = last_forward_matrix[index] *\
                                   end_probabilities[index]
        return {
            'final_probabilities': final_probabilities,
            'forward_matrix': forward_matrix
        }

        pass

    def backward_algorithm(self, observations_sequence, emission_matrix, transition_matrix, start_probabilities):
        # print("input", observations_sequence)
        """

        :param observations_sequence:
        :param emission_matrix:
        :param transition_matrix:
        :param start_probabilities:
        :return:
        """
        backward_matrix = []
        inverse_observations_sequence = observations_sequence[::-1]

        end_probabilities = list(map(lambda state: 1, self.states))
        backward_matrix.append(end_probabilities)
        for index_observation, observation in enumerate(inverse_observations_sequence):

            if index_observation == 0:
                continue
            previous_observation = inverse_observations_sequence[index_observation - 1]
            key = list(previous_observation.keys())[0]

            backward_array = []
            beta_previous_states = backward_matrix[-1]
            for index_state, state in enumerate(self.states):
                beta_i = 0
                for index_previous_state, beta_previous_state in enumerate(beta_previous_states):
                    beta_i += beta_previous_state * \
                              transition_matrix[index_state][index_previous_state] * \
                              emission_matrix[index_previous_state][previous_observation.get(key)]
                backward_array.append(beta_i)
            backward_matrix.append(backward_array)

        final_probabilities = 0
        last_backward_matrix = backward_matrix[-1]
        first_observation = observations_sequence[0]
        key_first = list(observations_sequence[0].keys())[0]
        # end_probabilities = list(map(lambda transition: transition[-1], self.transitions))
        for index, state in enumerate(self.states):
            final_probabilities += start_probabilities[index] * \
                                   last_backward_matrix[index] * \
                                   emission_matrix[index][first_observation.get(key_first)]
        return {
            'final_probabilities': final_probabilities,
            'backward_matrix': backward_matrix[::-1]
        }


    def veterbi_algorithm(self):
        """

        :return:
        """
        pass

    def baum_welch_algorithm(self, list_observations_sequence, number_thread):
        check_convergence = False
        iteration_number = 1
        matrix_emission_previous = self.get_matrix_emission()
        matrix_transition_previous = self.get_matrix_transition()
        sub_list_observations_sequence = []
        self.emission_changes = []
        self.transition_changes = []
        for index_observation_sequence, observations_sequence in \
            enumerate(list_observations_sequence):
            if index_observation_sequence < number_thread:
                sub_list_observations_sequence.append([observations_sequence])
            else:
                index_sub = index_observation_sequence % number_thread
                sub_list_observations_sequence[index_sub].append(observations_sequence)
        while not check_convergence:
            print('===================*Iteration %i*===================' % iteration_number)
            list_counting = []

            start_time = time.time()
            thread_array = []
            for sub_list in sub_list_observations_sequence:
                thread_array.append(threading.Thread(
                    target=self.counting_emissions_and_transition,
                    args=(sub_list, list_counting,)
                ))
                thread_array[-1].start()
            for thread in thread_array:
                thread.join()
            end_time = time.time()

            print('Processing time:', (end_time - start_time))
            counting_emissions = list_counting[0][0]
            counting_transition = list_counting[0][1]
            for index_counting, counting in enumerate(list_counting):
                if index_counting == 0:
                    continue
                counting_emissions = self.sum_counting(counting_emissions, counting[0])
                counting_transition = self.sum_counting(counting_transition, counting[1])

            # Bước M # Bước này phải cập nhật W :0)
            # calculate new weight emission matrix
            feature_emission = self.vocab_feature_e
            print("count e", counting_emissions)
            print("count t", counting_transition)
            # self.W_emissions = self.GD_momentum(self.get_w_emission(), feature_emission, 0.2, counting_emissions, 0.5, 0.9)
            lb_1 = LBFGS(counting_emissions, self.vocab_feature_e, 0.2)
            w_init = self.W_emissions
            self.W_emissions = lb_1.quadratic(w_init)[1]

            # calculate new weight transition matrix

            feature_transition = self.feature_t

            # self.w_transitions = self.GD_momentum(self.get_w_transition(), feature_transition, 0.2, counting_transition, 0.5, 0.9)

            lb_2 = LBFGS(counting_transition, self.feature_t, 0.2)
            self.w_transitions = lb_2.quadratic(self.w_transitions)[1]

            print("w_emission", self.get_w_emission())
            print("w_transion", self.get_w_transition())

            check_convergence = self.__check_convergence(matrix_emission_previous, matrix_transition_previous)
            iteration_number += 1
            matrix_emission_previous = self.get_matrix_emission()
            matrix_transition_previous = self.get_matrix_transition()

    def counting_theta_feature(self, W, feature):
        """
        w (vecto)
        feature_array : each feature is vector which have dimension = dimension of w
        return sum(theta(w)*feature)
        :return:
        """
        matrix_probabilities = self.get_probabilities(W, feature)
        len_theta_w_state = len(feature[0][0])
        theta_w = []
        for i, W_state in enumerate(W):
            theta_w_state = np.zeros(len_theta_w_state, dtype=np.float64)
            for j, value in enumerate(feature[i]):
                theta_w_state += matrix_probabilities[i][j]*np.array(value, dtype=np.float64)
            theta_w.append(theta_w_state)
        return theta_w

    def denta_w_dct(self, W, feature):
        """
        return matrix loss w of each feature
        :return:
        """
        matrix_theta = self.counting_theta_feature(W, feature)
        matrix_loss = []
        for i, W_state in enumerate(W):
            matrix_loss_state = []
            for j, value in enumerate(feature[i]):
                matrix_loss_state.append(np.array(value, dtype=np.float64)-matrix_theta[i])
            matrix_loss.append(matrix_loss_state)
        return matrix_loss

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
        grad_loss = []
        for state in self.states:
            grad_loss_state = np.zeros(len_vector_feature, dtype=np.float64)
            for i, value in enumerate(matrix_count_e[state]):
                grad_loss_state += value*np.array(matrix_denta_w_dct[state][i], dtype=np.float64)
            grad_loss_state -= 2*k*W[state]
            grad_loss.append(grad_loss_state)
        return grad_loss

    def loss_w_e(self, matrix_count_e, W, feature, k):

        matrix_probabilities = self.get_probabilities(W, feature)
        loss_w_e = []
        for state in self.states:
            loss_w_e_state = 0
            for i, value in enumerate(matrix_count_e[state]):
                loss_w_e_state += matrix_count_e[state][i]*math.log(matrix_probabilities[state][i])

            loss_w_e_state = loss_w_e_state - k*norm(matrix_probabilities[state])
            loss_w_e.append(loss_w_e_state)
        return loss_w_e

    def has_converged(self,theta_new, grad_theta_new):
        return np.linalg.norm(grad_theta_new)/theta_new.size < 1e-3

    def GD_momentum(self, w_init, f, k, count_e, eta, gamma):
    # Suppose we want to store history of theta
        theta = [w_init]

        print(theta[-1])

        v_old = np.zeros_like(w_init)
        for it in range(100):
            v_new = gamma * v_old + eta * np.array(self.grad_loss_w_e(count_e, theta[-1], f, k))
            theta_new = theta[-1] - v_new
            print(self.get_probabilities(theta_new, f))
            grad_theta_new = self.grad_loss_w_e(count_e, theta_new, f, k)
            if self.has_converged(theta_new, grad_theta_new):
                break
            theta.append(theta_new)
            v_old = v_new
        return theta[-1]

    def check_w_convergence(self, grad_loss_w):
        check_convergence = False
        count_check = 0
        for state in self.states:
            if np.linalg.norm(grad_loss_w[state])/len(grad_loss_w) < 1e-3:
                count_check += 1
        if count_check == len(self.states):
            check_convergence = True
        return check_convergence

    def update_w(self, grad, W, k):
        W -= k*np.array(grad, dtype=np.float64)
        return W

    def gradient_desent(self, W_init, eta):
        pass

    def counting_emissions_and_transition(self, list_observations_sequence, list_counting):
        # print 'Start thread at: %f' % time.time()
        counting_emissions = []
        counting_transition = []
        emission_matrix = self.get_matrix_emission()
        transition_matrix = self.get_matrix_transition()
        start_probabilities = self.get_start_probabilities()
        for state in self.states:
            emisstion_zero_arrays = np.zeros(len(self.vocab_number), dtype=np.float64)
            counting_emissions.append(emisstion_zero_arrays)

            transition_zero_arrays = np.zeros(len(self.states), dtype=np.float64)
            counting_transition.append(transition_zero_arrays)

        for observations_sequence in list_observations_sequence:
            forward_matrix = self.forward_algorithm(observations_sequence, emission_matrix, transition_matrix, start_probabilities)

            final_probabilities = forward_matrix['final_probabilities']
            forward_matrix = forward_matrix['forward_matrix']

            backward_matrix = self.backward_algorithm(observations_sequence, emission_matrix, transition_matrix, start_probabilities)
            backward_matrix = backward_matrix['backward_matrix']

            if final_probabilities == 0:
                continue
            # caculate P(h_t=i, X)
            for index_observation, observation in enumerate(observations_sequence):
                key_observation = list(observation.keys())[0]
                for index_state, state in enumerate(self.states):
                    concurrent_probability_it = forward_matrix[index_observation][index_state]\
                                                * backward_matrix[index_observation][index_state]
                    concurrent_probability_it /= final_probabilities
                    # if observation.get(key_observation) == 5:
                    #     print(concurrent_probability_it)
                    counting_emissions[index_state][observation.get(key_observation)] += concurrent_probability_it

            # caculate P(h_t=i, h_t+1=j, X)
            for index_observation, observation in enumerate(observations_sequence):
                if index_observation == len(observations_sequence) - 1:
                    continue
                current_forward = forward_matrix[index_observation]
                next_backward = backward_matrix[index_observation + 1]
                for index_state, state in enumerate(self.states):
                    for index_next_state, state in enumerate(self.states):
                        transition_probability_ijt = current_forward[index_state] * \
                            self.get_matrix_transition()[index_state][index_next_state] * \
                            next_backward[index_next_state]
                        transition_probability_ijt /= final_probabilities

                        counting_transition[index_state][index_next_state] += \
                            transition_probability_ijt
        # print 'End thread at: %f' % time.time()

        return list_counting.append([counting_emissions, counting_transition])

    @staticmethod
    def sum_counting(counting_1, counting_2):

        for index_state, states_counting in enumerate(counting_1):
            # print(states_counting)
            for index_observation, observation_counting in enumerate(states_counting):
                counting_1[index_state][index_observation] += counting_2[index_state][index_observation]
        return counting_1

    def __check_convergence(self, old_emission_matrix, old_transition_matrix):

        new_emission_matrix = self.get_matrix_emission()
        new_transition_matrix = self.get_matrix_transition()
        emission_change = 0
        for index_state, state_emission in enumerate(new_emission_matrix):
            for index_observation, observation_emission in enumerate(state_emission):
                emission_change += abs(observation_emission - old_emission_matrix[index_state][index_observation])
        self.emission_changes.append(emission_change)

        transition_change = 0
        for index_state, state_transaction in enumerate(new_transition_matrix):
            for index_next_state, next_step_transacion in enumerate(state_transaction):
                transition_change += abs(next_step_transacion - old_transition_matrix[index_state][index_next_state])
        self.transition_changes.append(transition_change)

        print('Emission change:', emission_change)
        print('transition_change:', transition_change)

        check = (transition_change < 0.0001) and (emission_change < 0.0001)
        return check
