from HMM_add_Feature.Hmm import HiddenMarkovModel
from HMM_add_Feature.process_vlsp_data import ProcessDataVlsp
from sklearn.metrics import classification_report
import numpy as np
pdv = ProcessDataVlsp()

class Evalute:

    def __init__(self, model: HiddenMarkovModel):
        self.model = model

    def convert_data(self, list_doc, syllables_vn, punt, pun_option=False):
        tmp = []
        list_sentence = []
        list_syllable = []
        list_index = []
        vocab = self.model.vocab_number
        for doc in list_doc:
            tmp.extend(pdv.convert_doc_to_number(doc, vocab, syllables_vn, punt, pun_option))

        for i in tmp:
            if i != []:
                list_sentence.append(i)
        for sentence in list_sentence:
            convert = pdv.convert_string_to_number(sentence)
            list_syllable.append(convert[0])
            list_index.append(np.array(convert[1]))
        return list_syllable, list_index

    def predict_tag(self, list_sequence, using_sub_params=False,
        bigram_hash=None, invert_bigram_hash=None, number_occurrences=None,
        invert_dictionary=None):
        # print(list_sequence)

        # transition = self.model.get_matrix_transition()
        # emission = self.model.get_matrix_emission()
        # start = self.model.get_start_probabilities()

        tag_result = []
        for index, sequence in enumerate(list_sequence[1]):
            # print(index)
            # print(list_sequence[0][index])
            print(sequence)
            tag_sequence = self.model.veterbi_algorithm(
                                                        sequence,
                                                        using_sub_params,
                                                        bigram_hash,
                                                        invert_bigram_hash,
                                                        number_occurrences,
                                                        invert_dictionary
                                                    )[1]
            print(tag_sequence)
            tag_result.append(tag_sequence)
        return tag_result

    def evalute(self, predict, trust, target_name):
        """
        folow tags B I
        :param predict:
        :param trust:
        :param target_name:
        :return:
        """
        predict = np.array(predict)
        trust = np.array(trust)
        return classification_report(trust, predict, target_names=target_name, output_dict=True)

    def caculate_precesion(self, unlabeled_sequence, sequence_unlabel_index, labeled_sequence):
        """

        :param unlabeled_sequence: list
        :param sequence_unlabel_index
        :param labeled_sequence: list word in sentence of vlsp
        :return:
        """
        try:
            # predict_tags = self.predict(unlabeled_sequence)
            # print(sequence_unlabel_index)
            print("1. ", labeled_sequence)
            predict_tags = self.model.veterbi_algorithm(sequence_unlabel_index)[0]

            # predict_tags = predict_tags[1]
            print("2 .", predict_tags)

            word_array = []
            new_word = []
            for index, predict_tag in enumerate(predict_tags):
                if predict_tag == 0:
                    if new_word:
                        word_array.append('_'.join(new_word))
                    new_word = [unlabeled_sequence[index]]
                else:
                    new_word.append(unlabeled_sequence[index])
            word_array.append('_'.join(new_word))
            print("3. ", word_array )
            print("\n")
            number_predict_true = 0
            for word in word_array:
                if word in labeled_sequence:
                    number_predict_true += 1

            return {
                'success': True,
                'object': float(number_predict_true) / len(predict_tags),
                'number_predict_true': number_predict_true,
                'number_predict': len(word_array),
                'number_destination_word': len(labeled_sequence)
            }
        except Exception as error:
            return {
                'success': False,
                'message': str(error)
            }

    def report_precision(self, list_sentence_unlabel, list_sentence_label):
        """

        :param list_sentence_unlabel:
        :param list_sentence_label:
        :return:
        """
        total_precision = 0
        max_precision = 0
        min_precision = 100
        number_predict_true = 0
        number_predict = 0
        number_destination_word = 0
        for index_sentence, sentence in enumerate(list_sentence_unlabel[0]):
            sentence_label = list_sentence_label[index_sentence]
            eval_precision = self.caculate_precesion(sentence, list_sentence_unlabel[1][index_sentence], sentence_label)
            if not eval_precision['success']:
                print('Has an exception: %s' % str(eval_precision['message']))
                continue
            precision = eval_precision['object']
            number_predict_true += eval_precision['number_predict_true']
            number_predict += eval_precision['number_predict']
            number_destination_word += eval_precision['number_destination_word']
            total_precision += precision
            if precision > max_precision:
                max_precision = precision
            if precision < min_precision:
                min_precision = precision
            # print('Precesion for sentence %i: %f' % (index_sentence, precision))

        avg_precision = total_precision / len(list_sentence_unlabel[0])
        precision_words = float(number_predict_true) / number_predict
        recall = float(number_predict_true) / number_destination_word
        print('Average precision: %f' % avg_precision)
        print('Max precision: %f' % max_precision)
        print('Min precision: %f' % min_precision)
        print('Precision in words: %f' % precision_words)
        print('Recall: %f' % recall)

        return {"avg_precision": avg_precision,
                "Max precision": max_precision,
                "Min precision": min_precision,
                "Precision in words": precision_words,
                "recall": recall
                }
