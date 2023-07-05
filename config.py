# BPE
bpe_num_ops = 7000

# BLEU
bleu_big_n = 4

# batching
window_size = 3
batch_size = 64

# dictionary
index_sentence_start = 0 # <s>
sentence_start = '<s>'
index_sentence_end = 1 # </s>
sentence_end = '</s>'
index_unknown = 2 # <UNK>
unknown = '<UNK>'
index_padding = 3
padding = '<pad>'

# hyper-parameters:

# training
number_epochs = 7 # total number of epoches to iterated over
learning_rate = 0.0001 # tuning parameter, determines the pace at the model updates the values of the weights estimated
# A large learning rate can miss the global minimum 
# and in extreme cases can cause the model to diverge completely from the optimal solution.

# for adjusting learning rate
accuracy_threshold = 0.5 # difference of accuracy between the i-th and (i+1)-th epoch
perplexity_threshold = 5 # difference of perplexity between the i-th and (i+1)-th epoch

# feedforward
# arbitrarily chosen
embedding_dimension = 128
hidden_size_1 = 500 # output dim of fully connect src and fully connect tgt
hidden_size_2 = 750 # output dim of layer one and two

# decoding
max_length = 48 # longestes possible sentence?
beam_size = 5

# transformer
NUM_LAYERS = 6
DIM_MODEL = 512
DIM_FF = 2048
NUM_HEADS = 8
DROPOUT = 0.1
TRANSFORMER_BATCH_SIZE = 32
MAX_PADDING = 72