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
                elif helper.check_type_syllable(syllable, syllables_vn, pun) == "PUNCT":
                    vocab.add(syllable)
                else:
                    continue
            vocab.add("CODE")
            vocab.add("NUMBER")
            vocab.add("FOREIGN_SYLLABLE")
        vocab_number = self.convert_vocab_to_number(vocab)
        helper.write_json(vocab_number, path_out)
        return vocab

    def gen_feature_basic_t(self):
        """
        state : B (0), I(1)
        return feature of transition.Each element in the array is a vector,with
        A[i,j] = 1 if t : state = i , t + 1 : state = j
        :return:
        """
        array_transiton_basic = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
        return np.array(array_transiton_basic)

    def gen_feature_basic_e(self, vocab):
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
            feature_I.append(index)
            vocab_feature_basic_B[syllable] = feature_B
            vocab_feature_basic_I[syllable] = feature_I
        return vocab_feature_basic_B, vocab_feature_basic_I

    def add_enhance_to_feature_e(self, vocab_feature_basic, stop_word_path):
        """
        give index of feature = 1
        :param vocab_feature_basic:
        :param stop_word_path:
        :return:
        """
        list_stop_word = Helper().load_stop_word(stop_word_path)
        for vocab_feature_basic_state in vocab_feature_basic:

            len_vocab = len(vocab_feature_basic_state)
            print("len vocab", len_vocab)
            vocab_feature_enhance_state = {}
            for syllable in vocab_feature_basic_state:
                enhance_feature = self.gen_enhance_feature_e(syllable, list_stop_word, len_vocab)
                list_basic = vocab_feature_basic_state.get(syllable)
                vocab_feature_enhance_state[syllable] = list_basic.extend(enhance_feature)

        return vocab_feature_basic

    def gen_enhance_feature_e(self, syllable, list_stop_word, len_vocab):

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
        index_0 = len_vocab
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
        elif syllable.lower() in list_stop_word:
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

    def load_file_feature_e(self, file_feature_b, file_feature_i):
        """
        load file feature emission return vocab_feature (type tuple)
        :param file_feature_b:
        :param file_feature_i:
        :return:
        """
        feature_b = Helper().loadfile_data_json(file_feature_b)
        feature_i = Helper().loadfile_data_json(file_feature_i)
        vocab_feature = (feature_b, feature_i)
        return vocab_feature

    def covert_feature_to_array(self, vocab_feature):
        """
        give vocab feature (index) covert to mumpy array
        :param vocab_feature:
        :return:
        """
        array_feature = []
        for vocab_feature_state in vocab_feature:
            vocab_feature_state_array = []
            for syllable in vocab_feature_state:
                feature_syllable = np.zeros(len(vocab_feature_state) + 7)
                index = vocab_feature_state.get(syllable)
                for i in index:
                    feature_syllable[i] = 1
                vocab_feature_state_array.append(feature_syllable)
            array_feature.append(vocab_feature_state_array)

        return np.array(array_feature, dtype=np.float64)
