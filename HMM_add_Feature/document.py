from HMM_add_Feature.Helper import Helper
from HMM_add_Feature.Dictionary import Dictionary
from utils.settings import DATA_MODEL_DIR
from os.path import join
import re
class Document:
	"""class for process doc to list syllable and to number"""

	@staticmethod
	def detect_paragraph(doc):
		paragraphs = doc.splitlines()
		return paragraphs

	@staticmethod
	def split_sentences(paragraphs_array):
		array_pharagraph_with_sentence = []
		for paragraph in paragraphs_array:
			if paragraph:
				paragraph = paragraph.replace(". ", ".. ")
				sentences = paragraph.split(". ")
				for sentence in sentences:
					if sentence:
						sentences = Helper().clear_str(sentence)
					array_pharagraph_with_sentence.append(sentences)
		return array_pharagraph_with_sentence

	def convert_doc_to_number(self, doc, vocab_number, syllables_vn, punctuation):
		"""
		function return list sentence, each sentence is list dict, each dict is a syllable
		:param doc:
		:param vocab_number:
		:param syllables_vn:
		:param punctuation:
		:return:
		"""
		helper = Helper()
		list_pharagraph = self.detect_paragraph(doc)
		list_sentence = self.split_sentences(list_pharagraph)
		syllables_appear = vocab_number.keys()
		list_sentence_convert_to_number = []
		for sentence in list_sentence:
			list_syllable_sentence = sentence.split()
			list_syllable_number_sentence = []
			for syllable in list_syllable_sentence:
				syllable_number = {}
				type_syllable = helper.check_type_syllable(syllable, syllables_vn, punctuation)
				if type_syllable == "VIETNAMESE_SYLLABLE" and syllable.lower() in syllables_appear:
					syllable_number[syllable] = vocab_number.get(syllable.lower())
				elif type_syllable == "VIETNAMESE_SYLLABLE" and syllable.lower() not in syllables_appear:
					syllable_number[syllable] = vocab_number.get("FOREIGN_SYLLABLE")
				else:
					syllable_number[syllable] = vocab_number.get(type_syllable)
				list_syllable_number_sentence.append(syllable_number)
			list_sentence_convert_to_number.append(list_syllable_number_sentence)
		return list_sentence_convert_to_number


if __name__ == "__main__":
	helper = Helper()
	doc = helper.load_data_xml("wiki_00")[1]
	DOC = Document()
	vocab_number = helper.loadfile_data_json("vocab_1.json")
	syllables_vn = helper.load_dictionary("syllables_dictionary_1.txt")
	punt = helper.load_punctuation()
	result = DOC.convert_doc_to_number(doc, vocab_number, syllables_vn, punt)
	print(result)
