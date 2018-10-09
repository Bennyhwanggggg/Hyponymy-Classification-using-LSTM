#import torch
from config import config

_config = config()



"""
check number of continue word in given list before first O_TYPE, list could be partial list
"""
def continue_words(data):
	O_TYPE = 'O'
	word_len = 0
	for w in data:
		if w == O_TYPE:
			return word_len
		word_len += 1
	return word_len



'''
True positive: index match and bondary match on both golden and predicted
False positive: golden index and predicted matches but boundary doesn't match or golden index is O_TYPE but predicted is not
False negative: golden is not O_TYPE but predicted doesn't match
'''
def evaluate(golden_list, predict_list, debug_mode=False):

	tp, fp, fn = 0, 0, 0
	O_TYPE = 'O'
	combo_list = zip(golden_list, predict_list)

	for current_golden_list, current_predict_list in combo_list:
		assert len(current_golden_list) == len(current_predict_list), "Error:golden_list has different size to predict_list!"
		for i in range(len(current_golden_list)):

			## B-t, I-t
			## B-t, O
			len_g = continue_words(current_golden_list[i:])
			len_p = continue_words(current_predict_list[i:])

			if len_g == len_p == 0:
				## O -> O
				continue
			elif len_p == 0:
				## Bt -> O
				fn += 1
			elif len_g == 0:
				## O -> Bt
				fp += 1
			else:
				if len_g != len_p:
					## Bt It -> Bt O
					fp += 1
				else:
					if current_golden_list[i: i+len_g] != current_predict_list[i:i+len_p]:
						## Bt It -> Bt Ih
						fn += 1
					else:
						## Bt It -> Bt It
						tp += 1

			# ## tn case -> no need to consider this index when both O_TYPE
			# if current_golden_list[i] == O_TYPE and current_predict_list[i] == O_TYPE: 
			# 	continue
			# ## edge case when list only has one value
			# if len(current_golden_list) == 1: 
			# 	# tp case
			# 	if (current_golden_list[i] != O_TYPE) and (current_golden_list[i] == current_predict_list[i]):
			# 		tp += 1
			# 	# fn case
			# 	elif (current_golden_list[i] != O_TYPE) and (current_predict_list[i] != current_golden_list[i]):
			# 		fn += 1
			# 	# fp case
			# 	elif (current_golden_list[i] == O_TYPE) and (current_predict_list[i] != O_TYPE):
			# 		fp += 1
				
			# elif i == 0: # case when at begining of the list
			# 	# tp case
			# 	if current_golden_list[i] != O_TYPE and current_golden_list[i] == current_predict_list[i] \
			# 		and current_golden_list[i+1] == current_predict_list[i+1]:
			# 		tp += 1
			# 	# fp case
			# 	elif (current_golden_list[i] == O_TYPE and current_predict_list[i] != O_TYPE) or \
			# 		(current_golden_list[i] == current_predict_list[i] and current_golden_list[i+1] != current_predict_list[i+1]):
			# 		fp += 1
			# 	# fn case
			# 	elif (current_golden_list[i] != O_TYPE and current_predict_list[i] != current_golden_list[i]):
			# 		fn += 1

			# elif i == len(current_golden_list)-1: # case when at end of the list
			# 	# tp case
			# 	if current_golden_list[i] != O_TYPE and current_golden_list[i] == current_predict_list[i] \
			# 		and current_golden_list[i-1] == current_predict_list[i-1]:
			# 		tp += 1
			# 	# fp case
			# 	elif (current_golden_list[i] == O_TYPE and current_predict_list[i] != O_TYPE) or \
			# 		(current_golden_list[i] == current_predict_list[i] and current_golden_list[i-1] != current_predict_list[i-1]):
			# 		fp += 1
			# 	# fn case
			# 	elif (current_golden_list[i] != O_TYPE and current_predict_list[i] != current_golden_list[i]):
			# 		fn += 1
			# else:
			# 	# tp case
			# 	if current_golden_list[i] != O_TYPE and current_golden_list[i] == current_predict_list[i] \
			# 		and current_golden_list[i-1] == current_predict_list[i-1] \
			# 		and current_golden_list[i+1] == current_predict_list[i+1]:
			# 		tp += 1
			# 	# fp case
			# 	elif (current_golden_list[i] == O_TYPE and current_predict_list[i] != O_TYPE) or \
			# 		(current_golden_list[i] == current_predict_list[i] and \
			# 		(current_golden_list[i-1] != current_predict_list[i-1] or current_golden_list[i+1] != current_predict_list[i+1])):
			# 		fp += 1
			# 	# fn case
			# 	elif (current_golden_list[i] != O_TYPE and current_predict_list[i] != current_golden_list[i]):
			# 		fn += 1
	# precision = tp/(tp + fp)
	# recall = tp/(tp + fn)

	precision = 1.0* tp/(tp+fp)
	recall = 1.0* tp/(tp+ fn)

	f1 = (2*precision*recall)/(precision+recall)

	if debug_mode:
		print("tp: {}, fp: {}, fn: {}".format(tp, fp, fn))
		print("precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(precision, recall, f1))

	return f1



def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
	## original code from torch.nn._functions.rnn.LSTMCell
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
  pass;


if __name__ == "__main__":
	O_type = 'O'
	# a=[[1,2,O_type, 3], [1,O_type,O_type, 3]]
	# b=[[1,O_type,O_type,O_type],[1,O_type,3,4]]
	a=[['B-TAR', 'I-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
	b = [['O', 'O', 'O', 'O'], ['B-TAR', 'O', 'O', 'O']]
	evaluate(a, b, True)
