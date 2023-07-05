from dictionary import Dictionary
import textloader as dl
import numpy as np
import pandas as pd
from config import index_padding, padding
from transformer import subsequent_mask

class Batch:
    S = [[]]
    T = [[]]
    L = []

    def __init__(self, batchSize: int, windowSize: int) -> None:
        # is initialization with zero matrix even necessary?
        self.S = [[0 for _ in range(2*windowSize+1)] for _ in range(batchSize)]
        self.T = [[0 for _ in range(windowSize)] for _ in range(batchSize)]
        self.L = [0 for _ in range(batchSize)]

def alignment(target_pos: int, targetLength: int, sourceLength: int) -> int:
    # calculates the alignment of a target position in a source sentence (b_i)
    # first calculates the ratio of source sentence length to target sentence length 
    # then multilplies that ratio by the target position
    # returns the rounded result to the nearest integer
    return int(round((float(sourceLength)/float(targetLength))*target_pos))
    
def get_dictIndex_of_wordInText(pos: int, dictionary: Dictionary, sentence: list[str]) -> int:
    """finds the corresponding index in the dictionary of the word at the given position in sentence

    Args:
        pos: position of the word w in sentence
        dictionary: dictionary of the language the sentence is written in
        sentence: sentence as a list of words

    Returns:
        index of the word w in the dictionary
    """
    if pos < 0:
        # start of sentence if < start of index
        # index of words in sentence starts at 1
        return dictionary.get_index('<s>')
    elif pos >= len(sentence):
        # end of sentence if > length of target
        return dictionary.get_index('</s>')
    else:
        # print(f"pos={pos}")
        # print(f"len(sentence)={len(sentence)}")
        # print(f"word={sentence[pos]}")
        # print(f"index={dictionary.get_index(sentence[pos])}")
        # print(f"pos={pos}, word={sentence[pos]}, index={dictionary.get_index(sentence[pos])}")
        return dictionary.get_index(sentence[pos]) # index of words in sentence starts at 1
    
def text_to_array(text: list[list[str]], dict: Dictionary) -> list[list[int]]:

    list_of_indices = []

    for sentence in text:
        sentence_as_indices = []
        for word in sentence:
            sentence_as_indices.append(dict.get_index(word))
        list_of_indices.append(sentence_as_indices)

    return list_of_indices

def array_to_text(arrays: list[list[int]], dict: Dictionary) -> list[list[str]]:
    list_of_words = []

    for array in arrays:
        sentence = []
        for idx in array:
            sentence.append(dict.get_word(idx))
        list_of_words.append(sentence)

    return list_of_words
    
def batching(batchSize: int, windowSize: int, 
             sourceText: list[list[str]], targetText: list[list[str]]) -> list[Batch]:
    """generates batches of a given batch and window size from source and target text

    Args:
        batchSize: given batch size (depends on the hardware)
        windowSize: given window size
        sourceText: given source text
        targetText: given target text

    Returns:
        a list of batches
    """
    # initialize batches
    batches = []

    # build dicts for source and target language from training data
    source_dict = Dictionary(sourceText)
    target_dict = Dictionary(targetText)

    # calculate length of target text for the outer iteration
    length_target_text = 0
    for target_sentence in targetText:
        length_target_text += len(target_sentence)   
    
    # keep track of which sentence e_i is in, and where e_i is in the sentence
    index_sentence = 0
    index_word = 0

    # each iteration creates a new batch
    for _ in range(0, length_target_text, batchSize):
        cur_batch = Batch(batchSize, windowSize)

        # calculate S, T, and L
        for batch_row in range(batchSize):
            if index_sentence == len(targetText):
                # if reached end of text but not end of batch
                cur_batch.S[batch_row] = [source_dict.get_index('</s>') for _ in range(-windowSize, +windowSize+1)]
                cur_batch.T[batch_row] = [source_dict.get_index('</s>') for _ in range(-windowSize, 0)]
                cur_batch.L[batch_row] = [source_dict.get_index('</s>')]
                continue
            
            if index_word == len(targetText[index_sentence])+1:
                # if reached end of sentence </s>
                # switch to next sentence
                index_sentence = min(index_sentence+1, len(targetText)-1)
                index_word = 0
            
            # calculate alignment
            I = len(targetText[index_sentence])
            J = len(sourceText[index_sentence])
            alignment = cur_batch.alignment(index_word, I, J)
            
            # calculate the batch_row-th row of S, T, and L
            cur_batch.S[batch_row] = [get_dictIndex_of_wordInText(m, source_dict, sourceText[index_sentence])
                                        for m in range(alignment-windowSize, alignment+windowSize+1)]
            
            cur_batch.T[batch_row] = [get_dictIndex_of_wordInText(m, target_dict, targetText[index_sentence])
                                        for m in range(index_word-windowSize, index_word)]
            
            cur_batch.L[batch_row] = [get_dictIndex_of_wordInText(index_word, target_dict, targetText[index_sentence])]
            
            # update index_word
            index_word += 1

        # # S : batchSize x (2w + 1)
        # cur_batch.S = [[get_dictIndex_of_wordInText(m, source_dict, sourceText)
        #                for m in range(n-windowSize, n+windowSize+1)]          # m = b_i - w, ..., b_i + w
        #                for n in range(b, b+batchSize)]                          # n = b_1, ..., b_{I+1}

        # # T : batchSize x w
        # cur_batch.T = [[get_dictIndex_of_wordInText(m, target_dict, targetText)
        #                for m in range(n-windowSize, n)]                        # m = 1-w, ..., I
        #                for n in range(i, i+batchSize)]                          # n = 1, ..., I+1

        # # L : batchSize x 1
        # cur_batch.L = [[get_dictIndex_of_wordInText(n, target_dict, targetText)]
        #                for n in range(i, i+batchSize)]                          # n = 1, ..., I+1

        batches.append(cur_batch)

    return batches

def print_batches_index(batches: list[Batch]) -> None:
    """print batches with elemtents of matrices in the form of index of words in dict

    Args:
        batches: a list of batches to be printed
    """
    for batch in batches:
        # np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
        # print(np.hstack((batch.S, batch.T, batch.L)))
        print(pd.DataFrame({"S": batch.S, "T": batch.T, "L": batch.L}))
    
def print_batches_string(batches: list[Batch], sourceDict: Dictionary, targetDict: Dictionary) -> None:
    """print batches with elements of matrices in the form words

    Args:
        batches: a list of batches to be printed
        sourceDict: dictionary for the source text
        targetDict: dictionary for the target text
    """
    for batch in batches:
        S_string = [[sourceDict.get_word(x) for x in row] for row in batch.S]
        T_string = [[targetDict.get_word(x) for x in row] for row in batch.T]
        L_string = [[targetDict.get_word(x) for x in row] for row in batch.L]
        print(pd.DataFrame({"S": S_string, "T": T_string, "L": L_string}))

class Seq2SeqBatch:
    """Object for holding a batch of data with mask during training"""

    def __init__(self, src, tgt=None, pad=index_padding):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1] # target context
            self.tgt_y = tgt[:, 1:] # what we want to predict
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

if __name__ == '__main__':

    source_text = dl.loadData_of_specific_lines("./data/bpe_de_15000.txt", 1100, 1200)
    target_text = dl.loadData_of_specific_lines("./data/bpe_en_15000.txt", 1100, 1200)
    # batches = batching(200, 3, source_text, target_text)

    # # settings for the matrix output
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_colwidth',None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.expand_frame_repr', False)
    # # pd.describe_option()

    # # output as matrix of indices
    # # print_batches_index(batches)
    
    # # output as string
    source_dict = Dictionary(source_text)
    target_dict = Dictionary(target_text)
    # print_batches_string(batches, source_dict, target_dict)
    src = text_to_array(source_text, source_dict)
    print(src)