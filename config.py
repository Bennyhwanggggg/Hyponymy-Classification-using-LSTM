# You can change the paths/hyper-parameters in this file to experiment your implementation
class config:
	use_f1 = True
	use_char_embedding = True
	use_modified_LSTMCell = True

	train_file = 'data/train.txt'
	dev_file = 'data/dev.txt'
	test_file = 'data/test.txt'
	output_tag_file = 'data/tags.txt'
	char_embedding_file = 'data/char_embeddings.txt'
	word_embedding_file = 'data/word_embeddings.txt'
	model_file = 'data/result_model.pt'

	word_embedding_dim = 50
	char_embedding_dim = 50
	char_lstm_output_dim = 50
	batch_size = 10
	hidden_dim = 50
	nepoch = 50
	dropout = 0.5

	nwords = 0
	nchars = 0
	ntags = 0