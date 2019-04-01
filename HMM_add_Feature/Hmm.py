"""class for HMM algorithm write by jinniPi"""
import numpy as np
from numpy.linalg import norm
from HMM_add_Feature.document import Document
from HMM_add_Feature.Helper import Helper
from HMM_add_Feature.Dictionary import Dictionary
from utils.settings import DATA_MODEL_DIR
from os.path import join
import threading
import time
import math


class HiddenMarkovModel:

    def __init__(self, states,transitions, W_emissions,
                 start_probabilities, vocab_feature, vocab_number):
        self.states = states  # tập trạng thái
        # self.observations = observations  # tập quan sát
        self.transitions = transitions  # xs chuyển
        self.W_emissions = W_emissions  # ma trận trọng số xs sinh trạng thái
        # self.W_emissions_I = W_emissions_I # ma trận trọng số xs sinh trạng thái I
        self.start_probabilities = start_probabilities  # xs ban đầu
        self.vocab_feature = vocab_feature
        self.vocab_number = vocab_number

    def get_matrix_transition(self):

        return np.array(self.transitions)

    def get_start_probabilities(self):

        return np.array(self.start_probabilities)

    def get_w_emission(self):
        return np.array(self.W_emissions, dtype=np.float64)
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

        list_feature = []
        for key, value in self.vocab_feature.items():
            temp = value
            list_feature.append(temp)

        w_emission = np.array(self.W_emissions)
        X = np.array(list_feature, dtype = np.float64)
        res = []
        for state in w_emission:
            Z_state = X.dot(state)
            e_Z = np.exp(Z_state - np.max(Z_state, axis=0, keepdims=True))
            A = e_Z / e_Z.sum(axis=0)
            res.append(A)
        return np.array(res)

    def forward_algorithm(self, observations_sequence):

        emission_matrix = self.get_matrix_emission()
        transition_matrix = self.get_matrix_transition()
        start_probabilities = self.get_start_probabilities()

        """

        :param observations_sequence: preprocessing (convert to number)
        :return: final_probabilities, forward_matrix

        """
        forward_matrix = []
        
        for index_observation, observation in enumerate(observations_sequence):
            forward_array = []
            key_observation = list(observation.keys())[0]
            # print(key_observation)
            #tinh alpha
            if index_observation == 0:
                for index_state, state in enumerate(self.states):
                    alpha_i =start_probabilities[index_state] * \
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
        # print(last_forward_matrix)
        # end_probabilities = list(map(lambda transition: transition[-1], self.transitions))
        end_probabilities = list(map(lambda state: 1, self.states))
        # print("end", end_probabilities)
        for index, state in enumerate(self.states):
            # print("indext", index)
            # print(state)
            final_probabilities += last_forward_matrix[index] * \
                end_probabilities[index]
        return {
            'final_probabilities': final_probabilities,
            'forward_matrix': forward_matrix
        }

        pass

    def backward_algorithm(self, observations_sequence):
        # print("input", observations_sequence)
        """

        :param observation_sequence:
        :return: final_probabilities, backward_matrix
        """
        emission_matrix = self.get_matrix_emission()
        transition_matrix = self.get_matrix_transition()
        start_probabilities = self.get_start_probabilities()
        backward_matrix = []
        inverse_observations_sequence = observations_sequence[::-1]

        end_probabilities = list(map(lambda state: 1, self.states))
        # end_probabilities = list(map(lambda transition: transition[-1], self.transitions))
        backward_matrix.append(end_probabilities)
        for index_observation, observation in enumerate(inverse_observations_sequence):

            if index_observation == 0:
                continue
            previous_observation = inverse_observations_sequence[index_observation - 1]
            # print("pre",previous_observation)
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
        check_w_convergence = False
        iteration_number = 1
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
        # print(sub_list_observations_sequence)
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

            print('Processing time:', (end_time - start_time) )
            print("list_counting", list_counting)
            counting_emissions = list_counting[0][0]
            print("counting emissions", counting_emissions)
            counting_transition = list_counting[0][1]
            # print("counting_transition", counting_transition)
            for index_counting, counting in enumerate(list_counting):
                print("counting0 ", counting[0])
                print(index_counting)
                if index_counting == 0:
                    continue
                counting_emissions = self.sum_counting(counting_emissions, counting[0])
                counting_transition = self.sum_counting(counting_transition, counting[1])

            # Bước M # Bước này phải cập nhật W :0)
            #calculate new emission matrix
            old_matrix_emission = self.get_matrix_emission()
            while not check_w_convergence:
                grad_w = self.grad_loss_w_e(counting_emissions, 0.2)
                self.update_w(grad_w, 0.2)
                print()
                loss_w_e_matrix = self.loss_w_e(counting_emissions, 0.2)
                check_w_convergence = self.check_w_convergence(loss_w_e_matrix)
            new_emission_matrix = self.get_matrix_emission()

            #calculatate new transition matrix
            new_transition_matrix = []
            for state_transaction in counting_transition:
                total_count = 0
                for next_step_transacion in state_transaction:
                    total_count += next_step_transacion

                new_state_transaction_probabilities = []
                for next_step_transacion in state_transaction:
                    new_state_transaction_probabilities.append(next_step_transacion /total_count)
                new_transition_matrix.append(new_state_transaction_probabilities)
            # new_transition_matrix[-1] = [1, 0, 0]

            check_convergence = self.__check_convergence(new_transition_matrix, new_emission_matrix, old_matrix_emission)
            self.transitions = new_transition_matrix
            print("new_transition", self.transitions)
            # print("")
            # self.emissions = new_emission_matrix
            iteration_number += 1

    def counting_theta_feature(self):
        """
        return sum(theta(w)*feature)
        :return:
        """

        emission_matrix = self.get_matrix_emission()
        list_feature = []
        for key, value in self.vocab_feature.items():
            temp = value
            list_feature.append(temp)
        matrix_theta = []
        for state in self.states:
            theta_state_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # print(type(theta_state_i))
            for i, value in enumerate(list_feature):
                # print(i)
                # print(emission_matrix[state][i])
                # print(value)
                theta_state_i += emission_matrix[state][i]*np.array(value, dtype=np.float64)
            matrix_theta.append(theta_state_i)
        return matrix_theta

    def denta_w_dct(self):
        """
        return matrix loss w of each feature
        :return:
        """
        matrix_theta = self.counting_theta_feature()
        matrix_loss = []
        list_feature = []
        for key, value in self.vocab_feature.items():
            temp = value
            list_feature.append(temp)
        for state in self.states:
            matrix_loss_state = []
            for i, value in enumerate(list_feature):
                matrix_loss_state.append(np.array(value, dtype=np.float64)-matrix_theta[state])
            matrix_loss.append(matrix_loss_state)
        return matrix_loss

    def grad_loss_w_e(self, matrix_count_e, k):
        """
        return l(w,e)
        :param matrix_count_e:
        :param k:
        :return:
        """
        print("matrix_count_e", matrix_count_e)
        w_emission = self.get_w_emission()
        matrix_denta_w_dct = self.denta_w_dct()
        print("matrix denta w dct", matrix_denta_w_dct)
        grad_loss = []
        for state in self.states:
            grad_loss_state = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
            for i, value in enumerate(matrix_count_e[state]):
                # print("value", value)
                # print("matrix_denta_w_dct[i]", matrix_denta_w_dct[state][i])
                # print(value*matrix_denta_w_dct[state][i])
                grad_loss_state+= value*matrix_denta_w_dct[state][i]
            print("grad", grad_loss_state)
            grad_loss_state -= 2*k*w_emission[state]
            grad_loss.append(grad_loss_state)
        return grad_loss

    def loss_w_e(self, matrix_count_e, k):
        matrix_emission = self.get_matrix_emission()
        loss_w_e = []
        for state in self.states:
            loss_w_e_state = 0
            for i, value in enumerate(matrix_count_e[state]):
                print(state)
                print(i)
                print(matrix_emission[state][i])
                loss_w_e_state += matrix_count_e[state][i]*math.log(matrix_emission[state][i])

            loss_w_e_state = loss_w_e_state - k*norm(matrix_emission[state])
            # print(loss_w_e_state)
            loss_w_e.append(loss_w_e_state)
        return loss_w_e

    def check_w_convergence(self, loss_w_e_matrix):
        check_convergence = False
        count_check = 0
        for state in self.states:
            if loss_w_e_matrix[state] < 0.1:
                count_check += 1
        if count_check == len(self.states):
            check_convergence = True
        return check_convergence

    def update_w(self, grad, k):
        for state in self.states:
            self.W_emissions[state]=self.W_emissions[state] - k*grad[state]
        return self.W_emissions

    def counting_emissions_and_transition(self, list_observations_sequence, list_counting):
        # print 'Start thread at: %f' % time.time()
        counting_emissions = []
        counting_transition = []
        for state in self.states:
            emisstion_zero_arrays = [0] * len(self.vocab_number)
            counting_emissions.append(emisstion_zero_arrays)

            transition_zero_arrays = [0] * len(self.states)
            counting_transition.append(transition_zero_arrays)

        for observations_sequence in list_observations_sequence:
            # print(observations_sequence)
            forward_matrix = self.forward_algorithm(observations_sequence)

            final_probabilities = forward_matrix['final_probabilities']
            # print(final_probabilities)
            forward_matrix = forward_matrix['forward_matrix']

            backward_matrix = self.backward_algorithm(observations_sequence)
            backward_matrix = backward_matrix['backward_matrix']

            if final_probabilities == 0:
                continue
            # caculate P(h_t=i, X)
            for index_observation, observation in enumerate(observations_sequence):
                key_observation = list(observation.keys())[0]
                # print("key", key_observation)
                # concurrent_probability_i = []
                for index_state, state in enumerate(self.states):
                    concurrent_probability_it = forward_matrix[index_observation][index_state]\
                                                * backward_matrix[index_observation][index_state]
                    concurrent_probability_it /= final_probabilities
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
                            self.transitions[index_state][index_next_state] * \
                            next_backward[index_next_state]
                        transition_probability_ijt /= final_probabilities

                        counting_transition[index_state][index_next_state] += \
                            transition_probability_ijt
        # print 'End thread at: %f' % time.time()

        return list_counting.append([counting_emissions, counting_transition])

    @staticmethod
    def sum_counting(counting_1, counting_2):
        # print("------------------------------")
        # print("couting_1", counting_1)
        # print("couting_2", counting_2)
        for index_state, states_counting in enumerate(counting_1):
            # print(states_counting)
            for index_observation, observation_counting in enumerate(states_counting):
                counting_1[index_state][index_observation] += counting_2[index_state][index_observation]
        return counting_1

    def __check_convergence(self, new_transition_matrix, new_emission_matrix, old_emission_matrix):
        emission_change = 0
        for index_state, state_emission in enumerate(new_emission_matrix):
            for index_observation, observation_emission in enumerate(state_emission):
                emission_change += abs(observation_emission - old_emission_matrix[index_state][index_observation])
        self.emission_changes.append(emission_change)

        transition_change = 0
        for index_state, state_transaction in enumerate(new_transition_matrix):
            for index_next_state, next_step_transacion in enumerate(state_transaction):
                transition_change += abs(next_step_transacion - self.transitions[index_state][index_next_state])
        self.transition_changes.append(transition_change)

        print('Emission change:', emission_change)
        print('transition_change:', transition_change)

        check = (transition_change < 0.001) and (emission_change < 0.001)
        return check

if __name__ == "__main__":

    helper = Helper()
    punctuation = helper.load_punctuation()
    path_stopword = join(DATA_MODEL_DIR, "stop_word/c_e_l_viettreebank.txt")
    path_vlsp = join(DATA_MODEL_DIR, "vlsp")
    path_syllable_vn = join(DATA_MODEL_DIR, "syllable_vn/syllables_dictionary_1.txt")
    vocab_number = helper.loadfile_data_json("vocab_vlsp.json")
    data = helper.load_data_vlsp_path(path_vlsp)
    syllable_vn = helper.load_dictionary(path_syllable_vn)
    doc = Document()
    dic = Dictionary()
    vocab_feature = dic.covert_number_to_feature(vocab_number, path_stopword)
    # print(vocab_number)
    W_emission = [[1, 2, 5, 2, 5, 6], [1, 1, 7, 5, 1, 6]]
    maxtrix_transition = [[0.5, 0.5], [0.7, 0.3]]
    start_probabilities = [1, 0]
    states = [0, 1]
    hmm = HiddenMarkovModel(states,maxtrix_transition, W_emission, start_probabilities, vocab_feature, vocab_number)
    list_doc = []
    for doc_i in data:
        doc_i_ = doc.convert_doc_to_number(doc_i,vocab_number,syllable_vn,punctuation)
        if len(doc_i_) > 0:
          list_doc.extend(doc_i_)
        else:
            continue

    result = []
    # print(".....",list_doc[0])
    for i, doc_i in enumerate(list_doc):
        if doc_i != []:
            result.append(doc_i)
        else:
            continue

    for i, item in enumerate(result):
        if item == []:
            print(i)

    print(lis)
    # print(hmm.vocab_number)
    # print(hmm.get_matrix_emission())