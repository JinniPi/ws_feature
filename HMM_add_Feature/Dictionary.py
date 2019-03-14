from HMM_add_Feature.Helper import Helper
from utils.settings import DATA_MODEL_DIR
from os.path import join

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


if __name__ == "__main__":
    path_stopword = join(DATA_MODEL_DIR, "stop_word/c_e_l_viettreebank.txt")
    path_vlsp =join(DATA_MODEL_DIR, "vlsp/train")
    path_syllable_vn = join(DATA_MODEL_DIR, "syllable_vn/syllables_dictionary_1.txt")
    dic = Dictionary()
    dic.build_vocab(path_vlsp, "vocab_1.json", path_syllable_vn, "vlsp")
    # vocab_number = Helper().loadfile_data_json("vocab_1.json")
    # vocab_feature = dic.covert_number_to_feature(vocab_number, path_stopword)
    # print(vocab_number)
    # print(vocab_feature)
