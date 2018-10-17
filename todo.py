import torch
import torch.nn.functional as F
from config import config

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
	m_shape = batch_char_index_matrices.shape
	m = batch_char_index_matrices.clone()
	for i in range(len(batch_word_len_lists)):
		# print('****')
		for j in range(len(batch_word_len_lists[i])):
			# print('--')
			# print(batch_word_len_lists[i][j])
			length = batch_word_len_lists[i][j]
			for k in range(length-1, -1, -1):
				character = batch_char_index_matrices[i][j][k]
				# result = torch.zeros(batch_char_index_matrices.shape[0], batch_char_index_matrices.shape[1], m.shape[0])
				m[i][j][length-1-k] = character


	
	# m =  model.char_embeds(batch_char_index_matrices)
	#batch_reverse_char_index_lists = batch_char_index_matrices
	forward =  model.char_embeds(batch_char_index_matrices)[:,:,-1,:]
	backward = model.char_embeds(m)[:,:,-1,:]
	output_char_seq = torch.cat([forward, backward], dim=-1)


	# if m_shape[0] == 10 and m_shape[1] == 9 and m_shape[2] == 11:
	# 	# print(">>>>", output_char_seq.shape)
	# 	# print("!!!!", packed.shape)
	# 	#print("<<<<", batch_word_len_lists.shape, batch_word_len_lists)
	# 	#print(char_embeds)
	# 	print(batch_char_index_matrices)
	# 	print(">>>>>>==============")

	# 	print(m)
	# 	sys.exit()
	



	return output_char_seq

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
