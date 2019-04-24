
import json
import glob
import csv
import os
import re
import pickle
from os.path import join
from utils.settings import DATA_MODEL_DIR
FJoin = os.path.join
class Helper:
    """class help read file and load data """
    def __init__(self):
        pass

    @staticmethod
    def GetFiles(path):
        """Output: file_list là danh sách tất cả các file trong path và trong tất cả các
           thư mục con bên trong nó. dir_list là danh sách tất cả các thư mục con
           của nó. Các output đều chứa đường dẫn đầy đủ."""

        file_list, dir_list = [], []
        for dir, subdirs, files in os.walk(path):
            file_list.extend([FJoin(dir, f) for f in files])
            dir_list.extend([FJoin(dir, d) for d in subdirs])
        return file_list

    @staticmethod
    def loadfile_data_json(filename):
        with open(filename, encoding='utf-8') as json_data:
            data = json.load(json_data)
        return data

    @staticmethod
    def load_data_vlsp(filename):
        xml_file = open(filename, 'r')
        doc_array = []
        new_doc = []
        for line in xml_file:
            if ('pBody' in line) and ('/' not in line):
                new_doc = []
            elif ('pBody' in line) and ('/' in line):
                doc_array.extend(new_doc)
            else:
                new_doc.append(line)
        return doc_array

    def load_data_vlsp_path(self, path_folder):
        list_file = self.GetFiles(path_folder)
        doc_array = []
        for file in list_file:
            posts = self.load_data_vlsp(file)
            doc = '\n'.join(posts)
            doc = doc.replace("_", " ")
            doc_array.append(doc)
        return doc_array

    def load_data_vlsp_raw(self, path_folder):
        list_file = self.GetFiles(path_folder)
        doc_array = []
        for file in list_file:
            posts = self.load_data_vlsp(file)
            doc = '\n'.join(posts)
            doc_array.append(doc)
        return doc_array

    @staticmethod
    def load_data_xml(filename):
        xml_file = open(filename, 'r')
        doc_array = []
        new_doc = ''
        for line in xml_file:
            if line[:4] == '<doc':
                new_doc = ''
            elif line[:5] == '</doc':
                doc_array.append(new_doc)
            else:
                new_doc += line
        # print(type(doc_array[2]))
        return doc_array

    def load_data_xml_path(self, path_folder):

        list_file = self.GetFiles(path_folder)
        print(list_file)
        doc_array = []
        for file in list_file:
            doc_array.extend(self.load_data_xml(file))
        print(len(doc_array))
        return doc_array

    @staticmethod
    def load_dictionary(path_to_dict):

        dictionary = set()
        dict_file = open(path_to_dict, 'r')
        for line in dict_file:
            word = line.split('\t')[0]
            word = word.replace('\n', '')
            dictionary.add(word)
        return dictionary

    # @staticmethod
    # def load_vocab(path_vocab):
    #     file = open(path_vocab, "r")
    #     vocab = set()
    #     for line in file:
    #         syllable = line.replace("\n", "")
    #         vocab.add(syllable)
    #     return vocab

    @staticmethod
    def load_punctuation():
        punctuation = {'(', ')', ':', '-', ',', '.', '?', '...', '[', ']', '"', ';', '!'}
        return punctuation

    @staticmethod
    def load_stop_word(file_name):
        file = open(file_name, "r")
        list_stop_word = set()
        for line in file:
            word = line[:-1]
            list_stop_word.add(word)
        return list_stop_word

    @staticmethod
    def split_syllable(word, pun):
        if len(word) >= 2 and word.isalpha() is False and word.isdigit() is False:
            if word[0] in pun and word[1] in pun and word[-1] in pun and word[-2] in pun:
                word = word.replace(word[0], word[0]+" ").replace(word[1], word[1]+" ")\
                    .replace(word[-1], " "+word[-1]).replace(word[-2], " "+word[-2]).split()
                return word
            elif word[0] in pun and word[1]:
                word = word.replace(word[0], word[0] + " ").replace(word[1], word[1] + " ").split()
                return word
            elif word[-1] in pun and word[-2] in pun:
                word = word.replace(word[-1], " "+word[-1]).replace(word[-2], " "+word[-2]).split()
                return word
            elif word[0] in pun:
                word = word.replace(word[0], word[0] + " ").split()
                return word
            elif word[-1] in pun:
                word = word.replace(word[-1], " "+word[-1]).split()
                return word
        return word.split()

    def convert_doc_to_list(self, doc, option="wiki"):
        # pun = self.load_punctuation()
        if option == 'vlsp':
            list_syllable = doc.split()
            return list_syllable
        else:
            doc = self.clear_str(doc)
            list_syllable = doc.split()
            return list_syllable

    @staticmethod
    def check_type_syllable(syllable, syllables_dictionary, punctuation):

        if syllable.lower() in syllables_dictionary:
            return 'VIETNAMESE_SYLLABLE'
        elif syllable in punctuation:
            return 'PUNCT'
        elif syllable.isdigit():
            return 'NUMBER'
        elif syllable.isalpha() is False and syllable.isdigit() is False:
            return 'CODE'
        else:
            return 'FOREIGN_SYLLABLE'

    @staticmethod
    def clear_str(string):
        char_special = '\.|\,|\;|\(|\)|\>|\<|\'|\"|\-|\/|\:|\?|\!|\[|\]|\{|\}'
        str_clean = re.sub('([' + char_special + '])', r' \1 ', string)
        # str_clean = re.sub('[.]', ' ', str_clean)
        str_clean = str_clean.strip()
        str_clean = ' '.join(str_clean.split())
        return str_clean

    @staticmethod
    def covert_dict_to_array(vocab_feature):
        pass

    @staticmethod
    def write_json(data, path_out):
        with open(path_out, "w", encoding="utf-8") as file:
            data = json.dumps(data, ensure_ascii=False)
            file.write(data)
        return path_out

    @staticmethod
    def write_object(data, path_out):
        with open(path_out, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return path_out

    @staticmethod
    def load_model(file_model):
        with open(file_model, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
        return data



if __name__ == "__main__":
   doc = Helper.load_data_vlsp("/home/trang/Downloads/job_rabiloo/Word_Tokenizer/HMM_add_Feature/data/53.seg")
   print(doc[4].split())
