import torch
from config import config

_config = config()


'''
True positive: index match and bondary match on both golden and predicted
False positive: golden index and predicted matches but boundary doesn't match or golden index is 'O' but predicted is not
False negative: golden is not 'O' but predicted doesn't match
'''
def evaluate(golden_list, predict_list):
	tp, fp, fn = 0, 0, 0
	for i in range(len(golden_list)):
		current_golden_list = golden_list[i]
		current_predict_list = predict_list[i]
		for j in range(len(current_golden_list)):
			if current_golden_list[j] == 'O' and current_predict_list[j] == 'O': # no need to consider this index when both 'O'
				continue
			if len(current_predict_list) == 1 and len(current_golden_list) == 1: # edge case when list only has one value
				# tp case
				if current_golden_list[j] != 'O' and current_golden_list[j] == current_predict_list[j]:
					tp += 1
				# fp case
				elif current_golden_list[j] == 'O' and current_predict_list[j] != 'O':
					fp += 1
				# fn case
				elif (current_golden_list[j] != 'O' and current_predict_list[j] != current_golden_list[j]):
					fn += 1
			elif j == 0: # case when at begining of the list
				# tp case
				if current_golden_list[j] != 'O' and current_golden_list[j] == current_predict_list[j] \
					and current_golden_list[j+1] == current_predict_list[j+1]:
					tp += 1
				# fp case
				elif (current_golden_list[j] == 'O' and current_predict_list[j] != 'O') or \
					(current_golden_list[j] == current_predict_list[j] and current_golden_list[j+1] != current_predict_list[j+1]):
					fp += 1
				# fn case
				elif (current_golden_list[j] != 'O' and current_predict_list[j] != current_golden_list[j]):
					fn += 1
			elif j == len(current_golden_list)-1: # case when at end of the list
				# tp case
				if current_golden_list[j] != 'O' and current_golden_list[j] == current_predict_list[j] \
					and current_golden_list[j-1] == current_predict_list[j-1]:
					tp += 1
				# fp case
				elif (current_golden_list[j] == 'O' and current_predict_list[j] != 'O') or \
					(current_golden_list[j] == current_predict_list[j] and current_golden_list[j-1] != current_predict_list[j-1]):
					fp += 1
				# fn case
				elif (current_golden_list[j] != 'O' and current_predict_list[j] != current_golden_list[j]):
					fn += 1
			else:
				# tp case
				if current_golden_list[j] != 'O' and current_golden_list[j] == current_predict_list[j] \
					and current_golden_list[j-1] == current_predict_list[j-1] \
					and current_golden_list[j+1] == current_predict_list[j+1]:
					tp += 1
				# fp case
				elif (current_golden_list[j] == 'O' and current_predict_list[j] != 'O') or \
					(current_golden_list[j] == current_predict_list[j] and \
					(current_golden_list[j-1] != current_predict_list[j-1] or current_golden_list[j+1] != current_predict_list[j+1])):
					fp += 1
				# fn case
				elif (current_golden_list[j] != 'O' and current_predict_list[j] != current_golden_list[j]):
					fn += 1
	# precision = tp/(tp + fp)
	# recall = tp/(tp + fn)
	precision = tp/(tp+fp)
	recall = tp/(tp+ fn)

	f1 = (2*precision*recall)/(precision+recall)
	return f1



def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
  pass;


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
  pass;

