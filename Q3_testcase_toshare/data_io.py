# COMP6714 Project
# DO NOT MODIFY THIS FILE!!!
import numpy as np
import math
import random
import torch

UNKNOWN_WORD = "<UNK_WORD>"
PAD = '<PAD>'
UNKNOWN_CHAR = "<UNK_CHAR>"


# read tag vocabulary from given file (e.g., tags.txt)
def read_tag_vocab(file_name):
	tag_dict = {}
	with open(file_name) as f:
		for line in f:
			line = line.strip()
			if line not in tag_dict:
				tag_dict[line] = len(tag_dict)
	return tag_dict


# generate a dictionary of embeddings from given file (e.g., embeddings_all.txt)
# the embeddings are stored as string in this stage
# this function will be called by gen_embedding_from_file()
def gen_dict_from_file(file_name):
	embedding_keys = []
	embedding_dict = {}
	with open(file_name) as f:
		for line in f:
			line = line.strip()
			current_word = line.split()[0]
			embedding_keys.append(current_word)
			embedding_dict[current_word] = line.split()[1:]
	return embedding_keys, embedding_dict


# generate an index of embeddings from given file (e.g., embeddings_all.txt)
# key_index_dict stores words (chars) and its embedding index
# embedding is an n*d matrix, n is number of words (chars) and d is the dimensionality of embeddings
def gen_embedding_from_file(embedding_file, embeds_dim):
	key_list, key_dict = gen_dict_from_file(embedding_file)
	key_index_dict = {PAD: 0}
	embedding = np.zeros(shape=(len(key_dict) + len(key_index_dict), embeds_dim))
	for key in key_list:
		key_index_dict[key] = len(key_index_dict)
		assert len(key_dict[key]) == embeds_dim
		embedding[key_index_dict[key], :] = np.asarray(key_dict[key], dtype=np.float64)
	return embedding, key_index_dict


# read sentences and BIO tags from given file (e.g., train.txt)
# returns a list of sentences (sentence are formed as a list of words) and a list of tag sequences
def read_sentences_and_tags(file_name):
	sentence_list, tag_sequence_list = [], []
	with open(file_name) as f:
		sentence, tag_sequence = [], []
		for line in f:
			line = line.strip()
			word_tag_pair = line.split()
			if len(word_tag_pair) == 2:
				sentence.append(word_tag_pair[0])
				tag_sequence.append(word_tag_pair[1])
			else:  # sentences are split by an empty line
				if len(sentence) > 0:
					sentence_list.append(sentence)
					tag_sequence_list.append(tag_sequence)
					sentence, tag_sequence = [], []
	return sentence_list, tag_sequence_list


class DataReader(object):
	def __init__(self, config, file_name, input_word_dict, input_char_dict, output_tag_dict, batch_size, is_train=False):
		self.config = config
		self.input_word_dict = input_word_dict
		self.input_char_dict = input_char_dict
		self.output_tag_dict = output_tag_dict
		self.is_train = is_train
		self.batch_size = batch_size
		self.instance_count = 0
		self.read_data(file_name)

	# return the id of w, which can be used to query the embedding of w
	def get_word_ids(self, w):
		word = w.lower()
		if word in self.input_word_dict:
			return self.input_word_dict[word]
		else:
			return self.input_word_dict[UNKNOWN_WORD]

	# return the id of c, which can be used to query the embedding of c
	def get_char_ids(self, c):
		if c in self.input_char_dict:
			return self.input_char_dict[c]
		else:
			return self.input_char_dict[UNKNOWN_CHAR]

	# read and process the whole dataset (e.g., training set, test set)
	def read_data(self, file_name):
		sentence_list, tag_sequence_list = read_sentences_and_tags(file_name)
		assert len(tag_sequence_list) == len(sentence_list)
		self.instance_count = len(tag_sequence_list)
		word_index_lists = [[self.get_word_ids(word) for word in sentence] for sentence in sentence_list]
		char_index_matrices = [[[self.get_char_ids(char) for char in word] for word in sentence] for sentence in sentence_list]
		tag_index_lists = [[self.output_tag_dict[tag] for tag in tag_sequence] for tag_sequence in tag_sequence_list]
		self.dataset = list(zip(word_index_lists, char_index_matrices, tag_index_lists))
		self.current_batch_index = 0

	def has_next(self):
		if self.current_batch_index >= len(self):
			self.current_batch_index = 0
			return False
		else:
			return True

	def __iter__(self):
		return self

	def next(self):
		return self.__next__()

	# generate batches based on the length of sentences
	def generate_batches(self):
		if self.is_train:
			self.dataset = sorted(self.dataset, key=lambda x: (len(x[1]), random.random()))
		self.batch_list = []
		for i in range(len(self)):
			batch_data = self.dataset[i * self.batch_size: (i + 1) * self.batch_size]
			self.batch_list.append(batch_data)

	# read a batch of sentences
	def __next__(self):
		if not self.has_next():
			raise StopIteration()

		if self.current_batch_index == 0:
			self.generate_batches()

		[input_word_index_lists, input_char_index_matrices, input_tag_index_lists] = list(zip(*self.batch_list[self.current_batch_index]))

		batch_sentence_len_list = [len(x) for x in input_word_index_lists]
		input_word_len_lists = [[len(word) for word in sentence] for sentence in input_char_index_matrices]

		batch_word_index_lists = np.zeros((len(input_word_index_lists), max(batch_sentence_len_list)), dtype=int)
		batch_word_mask = np.zeros((len(input_word_index_lists), max(batch_sentence_len_list)), dtype=int)
		for i, (input_word_index_list, sent_len) in enumerate(zip(input_word_index_lists, batch_sentence_len_list)):
			batch_word_index_lists[i, :sent_len] = input_word_index_list
			batch_word_mask[i, :sent_len] = 1

		batch_char_index_matrices = np.zeros((len(input_word_index_lists), max(batch_sentence_len_list), max(map(max, input_word_len_lists))), dtype=int)
		batch_char_mask = np.zeros((len(input_word_index_lists), max(batch_sentence_len_list),max(map(max, input_word_len_lists))), dtype=int)
		for i, (input_char_index_matrix, word_len_list) in enumerate(zip(input_char_index_matrices, input_word_len_lists)):
			for j in range(len(word_len_list)):
				batch_char_index_matrices[i, j, :word_len_list[j]] = input_char_index_matrix[j]
				batch_char_mask[i, j, :word_len_list[j]] = 1

		batch_tag_index_list = np.zeros((len(input_word_index_lists), max(batch_sentence_len_list)), dtype=int)
		for i, (input_tag_index_list, sent_len) in enumerate(zip(input_tag_index_lists, batch_sentence_len_list)):
			batch_tag_index_list[i, :sent_len] = input_tag_index_list[:sent_len]

		batch_word_len_lists = np.ones((len(input_word_index_lists), max(batch_sentence_len_list)), dtype=int) # cannot set default value to 0
		for i, (word_len, sent_len) in enumerate(zip(input_word_len_lists, batch_sentence_len_list)):
			batch_word_len_lists[i, :sent_len] = word_len

		batch_sentence_len_list = torch.from_numpy(np.array(batch_sentence_len_list))
		batch_word_index_lists = torch.from_numpy(batch_word_index_lists).long()
		batch_char_index_matrices = torch.from_numpy(batch_char_index_matrices).long()
		batch_word_len_lists = torch.from_numpy(batch_word_len_lists).long()
		batch_tag_index_list = torch.from_numpy(batch_tag_index_list).long()
		batch_word_mask = torch.from_numpy(batch_word_mask).float()
		batch_char_mask = torch.from_numpy(batch_char_mask).float()

		self.current_batch_index += 1
		return batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list

	def __len__(self):
		return math.ceil(float(len(self.dataset)) / self.batch_size)