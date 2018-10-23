import torch
import torch.nn.functional as F
from config import config
import numpy as np

_config = config()


def get_type(data):
	O_TYPE = 'O'
	return O_TYPE if data == O_TYPE else data[-3:]

"""
check number of continue word in given list before first O_TYPE, list could be partial list
"""
def type_finder(g_data, p_data):
	O_TYPE = 'O'
	g_label = 0
	p_label = 0
	match = 0

	if not g_data:
		return g_label, p_label, match 
	g_word = g_data[0]
	p_word = p_data[0]
	## handle single char
	if len(g_data) == 1:
		g_label = 1 if g_word != O_TYPE else 0
		p_label = 1 if p_word != O_TYPE else 0
		match = 1 if (g_word != O_TYPE and g_word == p_word) else 0
		return g_label, p_label, match
	## mutliple chars
	bucket_list = [0]
	## golden list
	g_data.append(O_TYPE)
	prev_type = get_type(g_word)
	for i in range(1, len(g_data)):
		g_word = g_data[i]
		w_type = get_type(g_word)
		if w_type != prev_type:
			if prev_type != O_TYPE:
				g_label += 1
			prev_type = w_type
			bucket_list.append(i)

	## prediction list
	p_data.append(O_TYPE)
	prev_type = get_type(p_word)
	for i in range(1, len(p_data)):
		p_word = p_data[i]
		w_type = get_type(p_word)
		if w_type != prev_type:
			if prev_type != O_TYPE:
				p_label += 1
			prev_type = w_type

	## match
	for i in range(len(bucket_list) - 1):
		beg = bucket_list[i]
		end = bucket_list[i + 1]
		if (g_data[beg:end] == p_data[beg:end]) and (O_TYPE not in g_data[beg:end]):
			match += 1
	return g_label, p_label, match



'''
True positive: index match and bondary match on both golden and predicted
False positive: golden index and predicted matches but boundary doesn't match or golden index is O_TYPE but predicted is not
False negative: golden is not O_TYPE but predicted doesn't match
'''
def evaluate(golden_list, predict_list, debug_mode=False):

	final_g_label = 0
	final_p_label = 0
	final_match = 0
	combo_list = zip(golden_list, predict_list)

	for current_golden_list, current_predict_list in combo_list:
		assert len(current_golden_list) == len(current_predict_list), "Error:golden_list has different size to predict_list!"

		g_label, p_label, match = type_finder(current_golden_list, current_predict_list)
		final_g_label += g_label
		final_p_label += p_label
		final_match += match

	# precision = 1.0* tp/(tp+fp)
	# recall = 1.0* tp/(tp+ fn)
	try:
		precision = 1.0*final_match/final_p_label
		recall = 1.0*final_match/final_g_label
		f1 = (2*precision*recall)/(precision+recall)

		if debug_mode:
			print("final_g_label: {}, final_p_label: {}, final_match: {}".format(final_g_label, final_p_label, final_match))
			print("precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(precision, recall, f1))

	except:
		f1 = 0
	
	return round(f1,3)



def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
	## original code from torch.nn._functions.rnn.LSTMCell
	if input.is_cuda:
		igates = F.linear(input, w_ih)
		hgates = F.linear(hidden[0], w_hh)
		state = fusedBackend.LSTMFused.apply
		return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

	### don't modify below 3 lines of codes ====================================
	hx, cx = hidden
	gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
	ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
	### don't modify above 3 lines of codes ====================================

	ingate = F.sigmoid(ingate)
	forgetgate = F.sigmoid(forgetgate)
	cellgate = F.tanh(cellgate)
	outgate = F.sigmoid(outgate)

	#cy = (forgetgate * cx) + (ingate * cellgate)			## before modification
	cy = (forgetgate * cx) + ((1-forgetgate) * cellgate)	## after modification
	hy = outgate * F.tanh(cy)

	return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
	# Given an input of the size [2,7,14], we will convert it a minibatch of the shape [14,14] to 
	# represent 14 words(7 in each sentence), and 14 characters in each word.
	## NOTE: Please DO NOT USE for Loops to iterate over the mini-batch.
	char_size = batch_char_index_matrices.size()
	mini_batch = batch_char_index_matrices.view(char_size[0]*char_size[1], char_size[2])

	# Get corresponding char_Embeddings, we will have a Final Tensor of the shape [14, 14, 50]
	char_Embeddings = model.char_embeds(mini_batch)

	# Sort the mini-batch wrt word-lengths, to form a pack_padded sequence.
	# Feed the pack_padded sequence to the char_LSTM layer.
	batch_word_lengths = batch_word_len_lists.view(-1)
	perm_idx, sorted_batch_word_len_lists = model.sort_input(batch_word_lengths)
	sorted_input_embeds = char_Embeddings[perm_idx]

	# Get hidden state of the shape [2,14,50].
	_, desorted_indices = torch.sort(perm_idx, descending=False)
	outputs = pack_padded_sequence(sorted_input_embeds, lengths = sorted_batch_word_len_lists.data.tolist(), batch_first=True)
	outputs, hidden_state = model.char_lstm(outputs)

	# Recover the hidden_states corresponding to the sorted index.
	result = torch.cat([hidden_state[0][0], hidden_state[0][1]], dim=-1)
	result = result[desorted_indices]

	# Re-shape it to get a Tensor the shape [2,7,100].
	r_size = result.size()
	result = result.view(2, int(r_size[0]/2), r_size[-1])

	return result

def flip(x, dim):
	print('x.dim ==>', x.dim())
	print('x.shape ==>', x.shape)
	indices = [slice(None)] * x.dim()
	print('indices ==>',indices)
	indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
	#indices = indices[dim][-3:] + indices[dim][3:]
	print('indices after ==>',indices)
	return x[tuple(indices)]


if __name__ == "__main__":
	O_type = 'O'
	BT = 'B-TAR'
	IT = 'I-TAR'
	BH = 'B-HYP'
	IH = 'I-HYP'
	# a=[[BT,IT,O_type, BH], [BT,O_type,O_type, BH]]
	# b=[[BT,O_type,O_type,O_type],[BT,O_type,BH,IH]]

	# a=[['B-TAR', 'I-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
	# b = [['O', 'O', 'O', 'O'], ['B-TAR', 'O', 'O', 'O']]

	a = [[BT, IT, IT, IT, O_type, BH]]
	b = [[BT, IT, BH, IH, O_type, BH]]
	evaluate(a, b, True)
