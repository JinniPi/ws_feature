from HMM_add_Feature.Helper import Helper

helper = Helper()

class ProcessDataVlsp:

    def load_data_path(self, path):
        list_file = helper.GetFiles(path)
        list_doc = []
        for file in list_file:
            list_doc.append(self.load_data(file))
        return list_doc

    def load_data(self, file_name):
        content = helper.load_data_vlsp(file_name)
        return content

    def get_tag_sentence(self, sentence):
        word_array = sentence.split()
        tag_array = []
        tag_array_index = []
        for word in word_array:
            word_split = word.split("_")
            for index, syllable in enumerate(word_split):
                if index == 0:
                    tag_array.append("B")
                    tag_array_index.append(0)
                else:
                    tag_array.append("I")
                    tag_array_index.append(1)
        return tag_array, tag_array_index

    def get_sentence(self, sentence):
        syllable_array = []
        word_array = sentence.split()
        for word in word_array:
            word_split = word.split("_")
            for i in word_split:
                syllable_array.append(i)
        return syllable_array

    def get_word_array(self, sentence):
        word_array = sentence.split()
        return word_array

    def get_word_all_sentence(self, list_doc):
        list_sentence = []
        for doc in list_doc:
            for sentence in doc:
                list_sentence.append(self.get_word_array(sentence))
        return list_sentence

    def get_all_sentence(self, list_doc):
        list_sentence = []
        for doc in list_doc:
            for sentence in doc:
                list_sentence.append(self.get_sentence(sentence))
        return list_sentence

    def get_tag_doc(self, doc):
        tag_array = []
        for sentence in doc:
            tag_array.append(self.get_tag_sentence(sentence)[0])
        return tag_array

    def get_tag_all(self, list_doc):

        tag_array = []
        for doc in list_doc:
            tag_array.extend(self.get_tag_doc(doc))
        return tag_array

    def convert_doc_to_number(self, doc, vocab_number, syllables_vn, punctuation, punt=False):
        """
        function return list sentence, each sentence is list dict, each dict is a syllable
        :param doc:
        :param vocab_number:
        :param syllables_vn:
        :param punctuation:
        :param punt: if =False sẽ không đưa dấu câu về cùng 1 âm tiết
        :return:
        """
        helper = Helper()
        list_sentence = doc
        syllables_appear = vocab_number.keys()
        list_sentence_convert_to_number = []
        for sentence in list_sentence:
            list_syllable_sentence = self.get_sentence(sentence)
            list_syllable_number_sentence = []
            for syllable in list_syllable_sentence:
                syllable_number = {}
                type_syllable = helper.check_type_syllable(syllable, syllables_vn, punctuation)
                if type_syllable == "VIETNAMESE_SYLLABLE" and syllable in syllables_appear:
                    syllable_number[syllable] = vocab_number.get(syllable)
                elif type_syllable == "PUNCT":
                        if punt == False:
                            syllable_number[syllable] = vocab_number.get(syllable)
                        else:
                            syllable_number[syllable] = vocab_number.get(type_syllable)
                elif type_syllable == "VIETNAMESE_SYLLABLE" and syllable not in syllables_appear:
                    syllable_number[syllable] = vocab_number.get("FOREIGN_SYLLABLE")
                else:
                    syllable_number[syllable] = vocab_number.get(type_syllable)
                list_syllable_number_sentence.append(syllable_number)
            list_sentence_convert_to_number.append(list_syllable_number_sentence)
        return list_sentence_convert_to_number

    def convert_string_to_number(self, sentence):
        """

        :param sentence
        """
        strings = []
        index = []
        for syllable in sentence:
            strings.append(list(syllable.keys())[0])
            index.append(list(syllable.values())[0])
        return strings, index

if __name__ == "__main__":

    list_doc = ProcessDataVlsp().load_data_path("/home/trang/Downloads/job_rabiloo/Word_Tokenizer/HMM_add_Feature/data")
    all_tag = ProcessDataVlsp().get_tag_all(list_doc)
    print(all_tag)