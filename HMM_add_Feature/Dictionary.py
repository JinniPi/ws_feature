from HMM_add_Feature.Helper import Helper
from utils.settings import DATA_MODEL_DIR
from os.path import join
import numpy as np
class Dictionary:

    """
    class build Dictionary for doc of wiki and doc of VLSP
    build matrix feature for dictionary
    """

    def __init__(self):
        pass

    def build_vocab(self, path_folder, path_out, path_syllable_vn, option="wiki"):
        """
        function for build vocab , returns a set of syllables that appear in the data set.
        Default data according to xml structure.
        :param path_folder:
        :param path_syllable_vn
        :return:
        """
        helper = Helper()
        syllables_vn = helper.load_dictionary(path_syllable_vn)
        pun = helper.load_punctuation()
        vocab = set()
        if option == "vlsp":
            list_doc = helper.load_data_vlsp_path(path_folder)
        else:
            list_doc = helper.load_data_xml_path(path_folder)
        for doc in list_doc:
            list_syllable = helper.convert_doc_to_list(doc, option)
            for syllable in list_syllable:
                if helper.check_type_syllable(syllable, syllables_vn, pun) == "VIETNAMESE_SYLLABLE":
                    vocab.add(syllable)
                else:
                    continue
            vocab.add("PUNCT")
            vocab.add("CODE")
            vocab.add("NUMBER")
            vocab.add("FOREIGN_SYLLABLE")
        vocab_number = self.convert_vocab_to_number(vocab)
        helper.write_json(vocab_number, path_out)
        return vocab

    def gen_feature_basic(self, vocab):
        """
        function for generation feature basic given (tag z, token i)

        :param vocab:
        :return:
        """
        vocab_feature_basic_B = {}
        vocab_feature_basic_I = {}
        for syllable in vocab:
            feature_B = []
            feature_I = []
            index = vocab.get(syllable)
            feature_B.append(index)
            feature_I.append(len(vocab) + index)
            vocab_feature_basic_B[syllable] = feature_B
            vocab_feature_basic_I[syllable] = feature_I
        return vocab_feature_basic_B, vocab_feature_basic_I

    def add_enhance_to_feature(self, vocab_feature_basic, stop_word_path):
        list_stop_word = Helper().load_stop_word(stop_word_path)
        for vocab_feature_basic_state in vocab_feature_basic:
            len_vocab = len(vocab_feature_basic_state)
            print("len vocab", len_vocab)
            vocab_feature_enhance_state = {}
            for syllable in vocab_feature_basic_state:
                enhance_feature = self.gen_enhance_feature(syllable, list_stop_word, len_vocab)
                # print(enhance_feature)
                # print(type(enhance_feature))
                list_basic = vocab_feature_basic_state.get(syllable)
                vocab_feature_enhance_state[syllable] = list_basic.extend(enhance_feature)
        return vocab_feature_basic

    def gen_enhance_feature(self, syllable, list_stop_word, len_vocab):

        """
        function for generation feature (number, stopword, title, punction, ...)
        demension of vecto = 7
        +is Vietnamese_syllable: x[0]=1
        +is title: x[1] = 1
        +is in stop_word: x[2] = 1
        +is num : x[3] = 1
        +is code: x[4] = 1
        +is foreign_syllable: x[5] = 1
        +is punct : x[6] = 1

        :param syllable:
        :param list_stopword:
        :return:
        """
        index_0 = 2*len_vocab
        index = []
        if syllable == "PUNCT":
            i = index_0 + 6
            index.append(i)
        elif syllable == "FOREIGN_SYLLABLE":
            i = index_0 + 5
            index.append(i)
        elif syllable == "CODE":
            i = index_0 + 4
            index.append(i)
        elif syllable == "NUMBER":
            i = index_0 + 3
            index.append(i)
        elif syllable in list_stop_word:
            i = index_0 + 2
            index.append(index_0)
            index.append(i)
        elif syllable.istitle():
            i = index_0 + 1
            index.append(index_0)
            index.append(i)
        else:
            index.append(index_0)
        # print(index)
        return index

    @staticmethod
    def convert_vocab_to_number(vocab):
        """
        function to covert vocab in the dictionary to number
        :param path_vocab:
        :return:
        """
        dictionary = {}
        for i, syllable in enumerate(vocab):
            dictionary[syllable] = i
        return dictionary

    @staticmethod
    def covert_number_to_feature(vocab_number, stopword_path):

        """
        function help convert vocab to feature
        +is pun: 1(0)
        +is num : 1(0)
        +is Vietnamese_syllable: 1(0)
        +is stop_word. 1(0)
        +is code: 1
        +is foreign_syllable: 1(0)

        :param vocab_number:
        :param stopword_path:
        :return:
        """

        helper = Helper()
        list_stop_word = helper.load_stop_word(stopword_path)
        vocab_feature = {}
        for vocab in vocab_number:
            # print(vocab)
            if vocab == "CODE":
                vocab_feature[vocab_number[vocab]] = [0, 0, 0, 0, 1, 0]
                # print(vocab_feature[vocab_number[vocab]])
            elif vocab == "NUMBER":
                # print("vao NUMBER")
                vocab_feature[vocab_number[vocab]] = [0, 1, 0, 0, 0, 0]
                # print(vocab_feature[vocab_number[vocab]])
            elif vocab == "FOREIGN_SYLLABLE":
                vocab_feature[vocab_number[vocab]] = [0, 0, 0, 0, 0, 1]
            elif vocab == "PUNCT":
                vocab_feature[vocab_number[vocab]] = [1, 0, 0, 0, 0, 0]
            elif vocab in list_stop_word:
                # print("vao stop word")
                vocab_feature[vocab_number[vocab]] = [0, 0, 1, 1, 0, 0]
            else:
                vocab_feature[vocab_number[vocab]] = [0, 0, 1, 0, 0, 0]
            # print(vocab_feature)

        return vocab_feature


