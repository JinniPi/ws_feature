
from HMM_add_Feature.Helper import Helper
from  HMM_add_Feature.Hmm import HiddenMarkovModel
from utils.settings import DATA_MODEL_DIR
from os.path import join
from HMM_add_Feature.evaluate import Evalute
from HMM_add_Feature.Dictionary import Dictionary
from HMM_add_Feature.process_vlsp_data import ProcessDataVlsp
pdv = ProcessDataVlsp()
helper = Helper()


if __name__ == "__main__":

    path_data_test = join(DATA_MODEL_DIR, "vlsp/test/train")
    file_feature_e_b = join(DATA_MODEL_DIR, "vlsp/feature/feature_basic_B.json")
    file_feature_e_i = join(DATA_MODEL_DIR, "vlsp/feature/feature_basic_I.json")

    invert_dictionary_path = join(DATA_MODEL_DIR, "vlsp/vocab_invert_vlsp.json")
    occurrences_data_path = join(DATA_MODEL_DIR, "vlsp/data_pmi/occurrences_vlsp.pkl")


    # test with sub parameter
    bigram_path = join(DATA_MODEL_DIR, "vlsp/data_pmi/bigram_vlsp.pkl")
    invert_bigram_path = join(DATA_MODEL_DIR, "vlsp/data_pmi/invert_bigram_vlsp.pkl")

    # load data
    invert_bigram_hash = helper.load_obj(invert_bigram_path)
    # print(invert_bigram_hash)
    bigram_hash = helper.load_obj(bigram_path)
    invert_hmm_dictionary = helper.loadfile_data_json(invert_dictionary_path)
    statistic_bigram = helper.load_obj(occurrences_data_path)
    data = pdv.load_data_path(path_data_test)
    syllables_vn = helper.load_dictionary("syllables_dictionary_1.txt")
    punt = helper.load_punctuation()
    diction = Dictionary()
    vocab_feature_e = diction.load_file_feature_e(file_feature_e_b, file_feature_e_i)
    vocab_e = diction.covert_feature_to_array(vocab_feature_e)
    vocab_t = diction.gen_feature_basic_t()

    # loadmodel
    model = helper.load_model("model_basic_4.pickle")
    w_emission = model["emission"]
    w_transition = model["transition"]
    states = model["state"]
    vocab = model["vocab"]
    # print(vocab)
    # if "Trúng" in vocab:
    #     print("yes")
    start = model["start_probabilities"]
    # print(invert_hmm_dictionary)
    # print(invert_hmm_dictionary[2124])

    hmm = HiddenMarkovModel(states, w_transition, w_emission, start, vocab_e, vocab_t, vocab)
    print(hmm.get_matrix_emission())
    print(hmm.get_matrix_transition())

    ev = Evalute(hmm)
    list_sentence_unlabel = ev.convert_data(data, syllables_vn, punt)

    hmm = HiddenMarkovModel(states, w_transition, w_emission, start, vocab_e, vocab_t, vocab)
    ev = Evalute(hmm)
    list_sentence_label = pdv.get_tag_all(data)
    list_word = pdv.get_all_sentence(data)
    predict = ev.predict_tag(
        list_sentence_unlabel,
        using_sub_params=True,
        bigram_hash=bigram_hash,
        invert_bigram_hash=invert_bigram_hash,
        number_occurrences=statistic_bigram,
        invert_dictionary=invert_hmm_dictionary
    )
    trust = pdv.get_tag_all(data)
    # print(trust)
    # results = ce.conlleval(predict, trust, list_word, "result.txt")

    f = open("input_basic_pmi.txt", "w")
    for index, sentence in enumerate(predict):
        for index_tag , tag in enumerate(sentence):
            word = list_word[index][index_tag]
            tag_tmp = "x"
            tag_trust = trust[index][index_tag]+"-WS"
            line = word + "\t" + tag_tmp + "\t" +tag_trust+"\t"+tag+"-WS" +"\n"
            f.writelines(line)
        f.write("\n")
    f.close()




