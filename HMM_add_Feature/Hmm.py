"""class for HMM algorithm write by jinniPi"""
import numpy as np
import threading
import time
import math
from HMM_add_Feature.logistic_model import LogisticModel
import pickle
# from HMM_add_Feature.lbfgs import LBFGS
# from HMM_add_Feature.document import Document

lg = LogisticModel()

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
        matrix_transition = []
        for state in self.states:
            probabilities_state = lg.get_probabilities(self.w_transitions[state], self.feature_t[state])
            matrix_transition.append(probabilities_state)
        return np.array(matrix_transition)

    def get_start_probabilities(self):

        return np.array(self.start_probabilities)

    def get_w_emission(self):
        return self.W_emissions

    def get_matrix_emission(self):
        matrix_emission = []
        for state in self.states:
            weight = self.W_emissions[state]
            feature = self.vocab_feature_e[state]
            probabilities_state = lg.get_probabilities(weight, feature)
            matrix_emission.append(probabilities_state)
        return np.array(matrix_emission)

    def forward_algorithm(self, observations_sequence, emission_matrix, transition_matrix, start_probabilities):

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

    def veterbi_algorithm(self, observations_sequence, using_sub_params=False,
        bigram_hash=None, invert_bigram_hash=None, number_occurrences=None,
        invert_dictionary=None):
        """

        :param observations_sequence:
        :param using_sub_params:
        :param bigram_hash:
        :param invert_bigram_hash:
        :param number_occurrences:
        :param invert_dictionary:
        :return:
        """
        emissions = self.get_matrix_emission()
        transitions = self.get_matrix_transition()
        veterbi_matrix = []
        backtrace_matrix = []
        if using_sub_params:
            if not (bigram_hash and invert_bigram_hash):
                print('Bigram hash and invert bigram hash is required when using sub params!')
                return []

        for index_observation, observation in enumerate(observations_sequence):
            # key_observation = list(observation.keys())[0]
            veterbi_array = []
            backtrace_array = []

            if index_observation == 0:
                for index_state, state in enumerate(self.states):
                    alpha_i = self.start_probabilities[index_state] * \
                        emissions[index_state][observation]
                    beta_i = 0
                    veterbi_array.append(alpha_i)
                    backtrace_array.append(beta_i)
            else:
                alpha_previous_states = veterbi_matrix[-1]
                for index_state, state in enumerate(self.states):
                    alpha_i = 0
                    beta_i = 0
                    for index_previous_state, alpha_previous_state in enumerate(alpha_previous_states):
                        new_alpha_i = alpha_previous_state * \
                            transitions[index_previous_state][index_state] * \
                            emissions[index_state][observation]

                        if index_previous_state == 0:
                            alpha_i = new_alpha_i
                            beta_i = index_previous_state
                        elif alpha_i < new_alpha_i:
                            alpha_i = new_alpha_i
                            beta_i = index_previous_state

                    if using_sub_params and state == 1:

                        sub_parameter = self.__calculate_pmi(
                            bigram_hash, invert_bigram_hash, observation,
                            observations_sequence[index_observation - 1],
                            number_occurrences, invert_dictionary)
                        alpha_i = alpha_i * sub_parameter

                    veterbi_array.append(alpha_i)
                    backtrace_array.append(beta_i)
            veterbi_matrix.append(veterbi_array)
            backtrace_matrix.append(backtrace_array)

        best_score = 0
        last_veterbi_matrix = veterbi_matrix[-1]
        last_state = 0
        end_probabilities = list(map(lambda state: 1, self.states))
        for index, state in enumerate(self.states):
            final_score = last_veterbi_matrix[index] * \
                end_probabilities[index]
            if index == 0:
                best_score = final_score
                last_state = state
            elif best_score < final_score:
                best_score = final_score
                last_state = state

        #get state sequence with the highest probability
        states_sequence = [last_state]
        for index in range(1, len(backtrace_matrix))[::-1]:
            back_state = states_sequence[-1]
            states_sequence.append(backtrace_matrix[index][back_state])

        result = []
        for s in states_sequence[::-1]:
            if s==0:
                result.append("B")
            else:
                result.append("I")
        return states_sequence[::-1], result

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
            print("count e", counting_emissions)
            print("count t", counting_transition)
            for state in self.states:
                # lb_e = LBFGS(counting_emissions[state], self.vocab_feature_e[state], 0.1)
                # lb_t = LBFGS(counting_transition[state], self.feature_t[state], 0.1)
                # self.W_emissions[state] = lb_e.lbgfs(self.W_emissions[state])
                # self.w_transitions[state] = lb_t.lbgfs(self.w_transitions[state])

                self.W_emissions[state] = lg.gradient_descent_momentum(
                    self.W_emissions[state],
                    self.vocab_feature_e[state],
                    counting_emissions[state],
                    0.2,
                    0.01
                )

                self.w_transitions[state] = lg.gradient_descent_momentum(
                    self.w_transitions[state],
                    self.feature_t[state],
                    counting_transition[state],
                    0.2,
                    0.01

                )

            print("w_emission", self.get_w_emission())
            print("emiss", self.get_matrix_emission())

            print("w_transion", self.get_w_transition())
            print("trans", self.get_matrix_transition())
            # break

            check_convergence = self.__check_convergence(matrix_emission_previous, matrix_transition_previous)
            iteration_number += 1
            matrix_emission_previous = self.get_matrix_emission()
            matrix_transition_previous = self.get_matrix_transition()

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

        check = (transition_change < 0.002) and (emission_change < 0.001)
        return check

    def __calculate_pmi(self, bigram_hash, invert_bigram_hash, syllable_index,
        previous_syllable_index, number_occurrences, invert_dictionary):
        # print(syllable_index)
        syllable = invert_dictionary[str(syllable_index)].lower()
        previous_syllable = invert_dictionary[str(previous_syllable_index)].lower()
        if previous_syllable not in bigram_hash:
            # print(previous_syllable)
            return 1
        if syllable not in invert_bigram_hash:
            # print(syllable)
            return 1
        bigram = previous_syllable + ' ' + syllable
        previous_syllable_hash = bigram_hash[previous_syllable]
        syllable_hash = invert_bigram_hash[syllable]

        if bigram in previous_syllable_hash:
            # print("yes")
            forward_number_occurrences = previous_syllable_hash[bigram]['number_occurrences']
        else:
            return 1

        previous_syllable_occurrences = previous_syllable_hash['number_occurrences']
        syllable_occurrences = syllable_hash['number_occurrences']

        total_unigram_occurrences = number_occurrences['number_unigram_occurrences']
        return math.log(float(forward_number_occurrences) * total_unigram_occurrences /\
            (previous_syllable_occurrences * syllable_occurrences))

    def viterbi(self, V, a, b, initial_distribution):
        # print("a", a)
        # print("print", b)
        T = V.shape[0]
        # print("T", V)
        M = a.shape[0]
        # print(M)

        omega = np.zeros((T, M))
        omega[0, :] = np.log(initial_distribution * b[:, V[0]])

        prev = np.zeros((T - 1, M))

        for t in range(1, T):
            for j in range(M):
                # Same as Forward Probability
                probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

                # This is our most probable state given previous state at time t (1)
                prev[t - 1, j] = np.argmax(probability)

                # This is the probability of the most probable state (2)
                omega[t, j] = np.max(probability)

        # Path Array
        S = np.zeros(T)

        # Find the most probable last hidden state
        last_state = np.argmax(omega[T - 1, :])

        S[0] = last_state

        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1

        # Flip the path array since we were backtracking
        S = np.flip(S, axis=0)

        # Convert numeric values to actual hidden states
        result = []
        for s in S:
            if s==0:
                result.append("B")
            else:
                result.append("I")

        return result, S

    def save_model(self, file_model_name):
        """

        :param file_model_name:
        :return:
        """
        model = {"state": self.states,
                 "start_probabilities": self.start_probabilities,
                 "emission": self.W_emissions,
                 "transition": self.w_transitions,
                 "vocab": self.vocab_number
        }

        with open(file_model_name, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        return file_model_name

