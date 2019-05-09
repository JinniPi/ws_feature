from HMM_add_Feature.hmm_new import HiddenMarkovModel
from HMM_add_Feature.document import Document
from HMM_add_Feature.Helper import Helper
from HMM_add_Feature.Dictionary import Dictionary
from HMM_add_Feature.process_vlsp_data import ProcessDataVlsp
from utils.settings import DATA_MODEL_DIR
from os.path import join
import numpy as np

pdv = ProcessDataVlsp()


# path data
path_data_train = join(DATA_MODEL_DIR, "vlsp/train")
path_vocab = join(DATA_MODEL_DIR, "vlsp/vocab_vlsp_punt_special.json")
file_feature_e_b = join(DATA_MODEL_DIR, "vlsp/feature_not_independent/feature_enhance_not_idp_punt_special_B.json")
file_feature_e_i = join(DATA_MODEL_DIR, "vlsp/feature_not_independent/feature_basic_not_idp_punt_special_I.json")

#load data
helper = Helper()
data = pdv.load_data_path(path_data_train)
vocab_number = helper.loadfile_data_json(path_vocab)
syllables_vn = helper.load_dictionary("syllables_dictionary_1.txt")
punt = helper.load_punctuation()
result = []
result_0 = []
# print(data[1])
for doc in data:
        result.extend(pdv.convert_doc_to_number(doc, vocab_number, syllables_vn, punt, True))
for i in result:
    if i != []:
        result_0.append(i)

# set model
states = [0, 1]
diction = Dictionary()
vocab_feature_e = diction.load_file_feature_e(file_feature_e_b, file_feature_e_i)
vocab_e = diction.covert_feature_to_array(vocab_feature_e, 2*len(vocab_number)+6)
vocab_t = diction.gen_feature_basic_t()
start_probabilities = [1, 0]
W_emission = np.random.rand(2*len(vocab_number)+6)
# print(W_emission.shape)
W_transition = np.array([0.8, 0.4, 0.8, 0.2], dtype=np.float64)

hmm = HiddenMarkovModel(states, W_transition, W_emission, start_probabilities, vocab_e, vocab_t, vocab_number)
print(hmm.get_matrix_transition())
print(hmm.get_matrix_emission())
# print(result_0[0])
hmm.baum_welch_algorithm(result_0, 1)

# save model
hmm.save_model("model_enhance_B_not_idp_punt_special.pickle")

print(hmm.get_matrix_emission())
print(hmm.get_matrix_transition())
