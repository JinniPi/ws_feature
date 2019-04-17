from HMM_add_Feature.Hmm import HiddenMarkovModel
from HMM_add_Feature.document import Document
from HMM_add_Feature.Helper import Helper
from HMM_add_Feature.Dictionary import Dictionary
from utils.settings import DATA_MODEL_DIR
from os.path import join
import numpy as np

helper = Helper()
doc = helper.load_data_vlsp_path("/home/trang/Downloads/job_rabiloo/Word_Tokenizer/data/vlsp/train")
DOC = Document()
vocab_number = helper.loadfile_data_json("/home/trang/Downloads/job_rabiloo/Word_Tokenizer/data/vlsp/vocab_vlsp.json")
syllables_vn = helper.load_dictionary("syllables_dictionary_1.txt")
punt = helper.load_punctuation()
print(len(doc))
# print(doc[2])
result = []
result_0 = []
for doc_i in doc:
        result.extend(DOC.convert_doc_to_number(doc_i, vocab_number, syllables_vn, punt))
for i in result:
    if i != []:
        result_0.append(i)
# print(len(result_0))
result = DOC.convert_doc_to_number(doc[0], vocab_number, syllables_vn, punt)
# print(result[0])
file_feature_e_b = join(DATA_MODEL_DIR, "vlsp/feature/feature_enhance_B.json")
file_feature_e_i = join(DATA_MODEL_DIR, "vlsp/feature/feature_enhance_I.json")
file_feature_t = join(DATA_MODEL_DIR, "vlsp/test/feature/feature_basic_transition.json")
states = [0, 1]
diction = Dictionary()
vocab_feature_e = diction.load_file_feature_e(file_feature_e_b, file_feature_e_i)
vocab_e = diction.covert_feature_to_array(vocab_feature_e)
vocab_t = diction.gen_feature_basic_t()
start_probabilities = [1, 0]
W_emission = np.random.rand(len(vocab_number)*2 + 7)
W_transition = np.array([6, 4, 7, 3], dtype=np.float64)

hmm = HiddenMarkovModel(states, W_transition, W_emission, start_probabilities, vocab_e, vocab_t, vocab_number)
# # print(hmm.get_matrix_emission())
hmm.baum_welch_algorithm(result_0, 1)
# # print(hmm.get_matrix_emission()[0])
#
#
print(hmm.get_matrix_emission()[0])
print(hmm.get_matrix_transition())
