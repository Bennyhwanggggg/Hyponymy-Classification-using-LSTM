import torch
import torch.nn.functional as F
from config import config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

_config = config()


def evaluate(golden_list, predict_list):

	if all(not g for g in golden_list) and all(not p for p in predict_list):
		return 1

	fp, fn, tp = 0, 0, 0
	Blist, Ilist = ['B-TAR', 'B-HYP'], ['I-TAR', 'I-HYP']

	for j in range(len(golden_list)):
		for i in range(len(golden_list[j])):
			if predict_list[j][i] in Blist and golden_list[j][i] != predict_list[j][i]:
				fp += 1
			if golden_list[j][i] in Blist:
				if golden_list[j][i] != predict_list[j][i]:
					fn += 1
				else:
					if i != len(golden_list[j]) -1:
						for n in range(i+1, len(golden_list[j])):
							if golden_list[j][n] not in Ilist and predict_list[j][n] not in Ilist:
								tp+=1
								break
							elif golden_list[j][n] != predict_list[j][n]:
								fn += 1
								fp += 1
								break
							elif n==len(golden_list[j])-1 and golden_list[j][n] == predict_list[j][n]:
								tp += 1
					else:
						tp += 1

	try:
		f1 = (2*tp)/(2*tp + fn + fp)
	except:
		f1 = 0
	return f1

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
	result = result.view(char_size[0], int(r_size[0]/char_size[0]), r_size[-1])

	return result


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
	# a = [[]]
	# b= [[]]
	print(evaluate(a, b))
