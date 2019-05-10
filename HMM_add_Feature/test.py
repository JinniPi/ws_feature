# import numpy as np
# from HMM_add_Feature.document import Document
# from HMM_add_Feature.Helper import Helper
# from  HMM_add_Feature.Hmm import HiddenMarkovModel
# from utils.settings import DATA_MODEL_DIR
# from os.path import join
# from HMM_add_Feature.evaluate import Evalute
# from HMM_add_Feature.Dictionary import Dictionary
# from HMM_add_Feature.process_vlsp_data import ProcessDataVlsp
# pdv = ProcessDataVlsp()
#
#
#
# # path data
# path_data_test = join(DATA_MODEL_DIR, "vlsp/test/train")
# # path_vocab = join(DATA_MODEL_DIR, "vlsp/vocab_vlsp_punt_normal.json")
# file_feature_e_b = join(DATA_MODEL_DIR, "vlsp/feature_not_independent/feature_basic_punt_normal_B.json")
# file_feature_e_i = join(DATA_MODEL_DIR, "vlsp/feature_not_independent/feature_basic_punt_normal_I.json")
#
# #load data
# helper = Helper()
# data = pdv.load_data_path(path_data_test)
# # vocab_number = helper.loadfile_data_json(path_vocab)
# syllables_vn = helper.load_dictionary("syllables_dictionary_1.txt")
# punt = helper.load_punctuation()
# diction = Dictionary()
# vocab_feature_e = diction.load_file_feature_e(file_feature_e_b, file_feature_e_i)
# vocab_e = diction.covert_feature_to_array(vocab_feature_e)
# # print(vocab_feature_e)
# vocab_t = diction.gen_feature_basic_t()
#
# #loadmodel
# model = helper.load_model("model_basic_1.pickle")
# w_emission = model["emission"]
# # print(len(w_emission[0]))
# w_transition = model["transition"]
# states = model["state"]
# vocab = model["vocab"]
# # print(vocab)
# start = model["start_probabilities"]
#
# hmm = HiddenMarkovModel(states, w_transition, w_emission, start, vocab_e, vocab_t, vocab)
# matrix = hmm.get_matrix_emission()
# print(hmm.get_matrix_transition())
# f = open("emission_basic.txt", "w")
# f.writelines("W     B   I" + "\n")
# for index, word in enumerate(vocab):
#     lines = word + "\t" + str(matrix[0][index]) + "\t" + str(matrix[1][index])+"\n"
#     f.writelines(lines)
# f.close()



# ev = Evalute(hmm)
# # print(data[99])
# # print(ev.model.get_matrix_emission())
# list_sentence_unlabel = ev.convert_data(data, syllables_vn, punt)
# # print(list_sentence_unlabel[1][55])
# list_sentence_label = pdv.get_word_all_sentence(data)
# # print(list_sentence_unlabel[1][1])
# # print(ev.model.veterbi_algorithm(list_sentence_unlabel[1]))
#
# # evalute
# result = ev.report_precision(list_sentence_unlabel, list_sentence_label)
# A high-dimensional quadratic bowl.
