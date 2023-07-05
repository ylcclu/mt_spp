#from feedforward import FeedForwardNet
import torch
import torch.nn as nn
import numpy as np
import batch # for preprocessing src sentence string
from bpe import BPE # for posrprocessing
import textloader as tl
import config

# calculates alignment sentence wise
def alignment(training_src, training_tgt) -> int:
    # fixed average alignment over the training set
    align = 0

    # src and tgt are sentences as list of words
    for src, tgt in zip(training_src, training_tgt):
        align += float(len(src))/float(len(tgt))

    # returns rounded word per sentences
    return int(round(align/len(training_src)))


def greedy_search(model, src_sentence, alignment):
    """

    Args:
        model: pretrained
        src_sentence: list of word-indices (only one sentence)
        max_length: length of generated sentence
        alignment: calculated in def alignment()

    Returns: list of indices (translation of the input sentence)

    """
    # max length of translated sentence
    max_length = config.max_length

    # return value with indices of translated words
    path = []

    # sets the model to evaluation mode to ensure consistent predictions
    # deactivates certain layers that are only needed for training and behave differently in evaluation mode
    model.eval()

    # deactivates SGD
    with torch.no_grad():
        idx = 0

        while True:

            # fixed alignment / position of target
            align = alignment*idx

            # create src_win only for one sentences
            # every el. in list gets one line
            src_win = [src_sentence[m] if m in range(0,len(src_sentence)) 
                    else config.index_sentence_start if m < 0
                    else config.index_sentence_end
                    for m in range(align-config.window_size, align+config.window_size+1)]

            # tgt_win filled with sentence_start if there are no previous translations, else: previous translation
            tgt_win = [path[i+idx] if i + idx >= 0
                    else config.index_sentence_start for i in range(-config.window_size, 0)]

            # transforms windows into tensors
            # view changes the shape of the tensor to a 1-row tensor
            # makes sure dimension is correct
            src_win = torch.tensor(src_win).view(1, -1)
            tgt_win = torch.tensor(tgt_win).view(1, -1)

            # find the next most probable word
            # -1 refers to the last dimension of the tensor
            output = torch.log(torch.softmax(model(src_win, tgt_win), dim=-1))
            next = torch.argmax(output, dim=-1).item()

            path.append(next)

            if path[idx] is config.index_sentence_end or idx > max_length:
                break

            idx += 1

    return path

def beam1_search(model, src_sentence, alignment):
    """

    Args:
        model: trained model
        src_sentence: list of word-indices (only one sentence)
        beam_size: number of candidates
        max_length: length of generated sentence
        alignment: calculated in def alignment()

    Returns:

    """
    beam_size = config.beam_size
    max_length = config.max_length

    # src_sentence is a list of indices
    # original beam_size = k
    og_beam_size = beam_size

    candidates = [] # list of list of word indices
    cand_probs = [] # probability of those lists of words as a subsentence
    best = {} # contains complete sentences; key is list of word indices, value is joint prob

    # sets the model to evaluation mode to ensure consistent predictions
    # deactivates certain layers that are only needed for training and behave differently in evaluation mode
    model.eval()

    # deactivates SGD
    with torch.no_grad():
        for idx in range(max_length):

            # alignment of fixed proportion
            align = alignment*idx

            # generate src_win and tgt_win

            # initialize / batching
            # if no candidates, we have dim_0 = 1
            src_win = [src_sentence[m] if m in range(0,len(src_sentence)) 
                    else config.index_sentence_start if m < 0
                    else config.index_sentence_end 
                    for m in range(align-config.window_size, align+config.window_size+1)]

            # no context thus initialize with sentence_start
            tgt_win = [config.index_sentence_start for _ in range(config.window_size)]

            if candidates:
                # dim_0 = beam_size = #candidates
                # sry_win for every candidate
                src_win = [src_win for _ in range(beam_size)]

                # create tgt_win for each candidate
                tgt_win = [[path[i+idx] if i+idx >= 0
                            else config.index_sentence_start for i in range(-config.window_size, 0)] 
                            for path in candidates]

            else:
                # make batch_size = 1 to make size fit, otherwise cannot multiply in Linear layer
                src_win = [src_win]
                tgt_win = [tgt_win]

            # transform to tensor
            src_win = torch.tensor(src_win)
            tgt_win = torch.tensor(tgt_win)

            # expand search tree to k*k candidates
            output = torch.log(torch.softmax(model(src_win, tgt_win), dim=-1))

            # topk gives the k (=beam size) highest probs and the corresponding words
            # probability of the words are in probs, their indices in tgt_dict are in words
            probs, words = torch.topk(output, beam_size, dim=-1)


            # prune search tree => again only beam_size number of candidates (with highest prob) are left
            # add prev prob to the word prob
            # i = rows of prob tensor
            for i, prob_i in enumerate(probs):

                if cand_probs:

                    # multiplication (addition in log) to calculate conditional dependencies
                    # one row in prob is one product = one tensor in can_probs
                    probs[i] = torch.add(prob_i, cand_probs[i])
            # find the position of the k best probabilities in the 2-dimensional tensor probs
            # 'flatten()' reshapes to 1-dimensional tensor
            # topk(), sorts tensor (by default) so that largest element (here: prob) is first.
            value, position = torch.topk(probs.flatten(), beam_size)

            # transforms position in a beam size x 2 tensor
            # the positions are in descending order with regards to the probabilities
            position = torch.from_numpy(np.array(np.unravel_index(position.numpy(), probs.shape)).T)


            new_candidates = []
            new_probs = []
            # enumerate over position to update candidate (and best)
            for i, pos in enumerate(position):
                # for the word in pos = (x, y)
                # its previous string is the x-th candidate
                # it has index y in tgt_dict
                if candidates:
                    prev = [x for x in candidates[pos[0]]]
                    prev.append(words[pos[0], pos[1]].item())
                    new_candidate = prev
                else:
                    new_candidate = [words[pos[0], pos[1]].item()]


                if words[pos[0], pos[1]].item() != config.index_sentence_end:
                    new_candidates.append(new_candidate)
                    new_probs.append(value[i]/(idx+1)) # length normalization

                else:
                    # special case: the generated word is </s>
                    # i.e., a complete sentence is generated
                    # add sentence to best and decrement beam_size (expand only k-1 other candidates)
                    best[tuple(new_candidate)] = probs[pos[0], pos[1]].item()
                    beam_size -= 1
            
            candidates = new_candidates
            cand_probs = new_probs

            # if idx == 0:
            #     print(f"first k candidates {candidates}")

        # if after max_length iterations best has less than k entries
        # fill with top candidates

        for i in range(og_beam_size-len(best)):

            # best is a dictionary with hypothesis with highest prob
            best[tuple(candidates[i])] = (cand_probs[i]/max_length).item() # also length normalization

    best = dict(sorted(best.items(), key=lambda item: item[1], reverse=True))
    return best

    ####################################################################################################################
    # new version of beam search
    ####################################################################################################################


def find_k_highest_values(tensors, k):
    # brute force find the position and value of the top k values in the given tensors
    highest_values = []

    for tensor_idx, tensor in enumerate(tensors):
        flattened_tensor = tensor.flatten()
        topk_values, topk_indices = torch.topk(flattened_tensor, k)

        # Update the list of k highest values
        for idx, val in zip(topk_indices, topk_values):
            idx = idx.item()
            val = val.item()
            if len(highest_values) < k:
                highest_values.append([tensor_idx, idx, val])
            else:
                min_val = min(highest_values, key=lambda x: x[2])
                if val > min_val[2]:
                    highest_values.remove(min_val)
                    highest_values.append([tensor_idx, idx, val])

    return highest_values


def beam_search(model, src_sentence, alignment):
    """

    Args:
        model: trained model
        src_sentence: list of word-indices (only one sentence)
        alignment: calculated in def alignment()

    Returns:
        a dict (best) containing complete sentences; key is list of word indices, value is joint prob
    """
    beam_size = config.beam_size

    candidates = []  # list of list of word indices, candidates has beam_size many elements (= paths)
    cand_probs = []  # probability of those lists of word indices as a subsentence
    best = {}  # contains complete sentences; key is list of word indices, value is joint prob

    # sets the model to evaluation mode to ensure consistent predictions
    # deactivates certain layers that are only needed for training and behave differently in evaluation mode
    model.eval()

    idx = 0 # index of the word generated
    with torch.no_grad(): 
    # deactivates SGD
        while beam_size != 0:

            # alignment of fixed proportion
            align = alignment * idx

            # generate src_win and tgt_win
            # initialize / batching
            # if no candidates, we have dim_0 = 1
            src_win = [src_sentence[m] if m in range(0, len(src_sentence))
                       else config.index_sentence_start if m < 0
                       else config.index_sentence_end
                       for m in range(align - config.window_size, align + config.window_size + 1)]

            # no context thus initialize with sentence_start
            tgt_win = [config.index_sentence_start for _ in range(config.window_size)]

            if candidates:
                # dim_0 = beam_size = #candidates
                # sry_win for every candidate are the same
                src_win = [src_win for _ in range(beam_size)]

                # create tgt_win for each candidate
                tgt_win = [[path[i + idx] if i + idx >= 0
                            else config.index_sentence_start for i in range(-config.window_size, 0)]
                           for path in candidates]

            else:
                # make batch_size = 1 to make size fit, otherwise cannot multiply in Linear layer
                src_win = [src_win]
                tgt_win = [tgt_win]

            #print(tgt_win)

            # transform to tensor
            src_win = torch.tensor(src_win)
            tgt_win = torch.tensor(tgt_win)

            """
            generate a softmax over the entire vocabulary to extend the hypothesis to every possible next token. 
            Each of these k*V hypotheses is scored by the product of the probability of current word
            choice multiplied by the probability of the path that led to it
            """

            output = torch.log(torch.softmax(model(src_win, tgt_win), dim=-1))

            # topk returns the k (=beam size) highest probs and the corresponding words as tuple
            # for each of the candidate topk, according to the probability distribution in dim=-1
            # probability of the words are in probs, their indices in tgt_dict are in words
            probs, words = torch.topk(output, beam_size, dim=-1)

            #print("words:", words)
            #print("probs:", probs)

            if idx == 0:
            # first iteration: sentence prob is the prob of the first word
                cand_probs = [x for x in probs]

            # add prev prob to the word prob (log space)
            # i = rows of prob tensor
            for i, prob_i in enumerate(probs):
                # multiplication (addition in log) to calculate conditional dependencies
                # one row in prob is one product = one tensor in cand_probs
                if candidates:
                    cand_probs[i] = torch.add(cand_probs[i], prob_i)

            # find the topk amongst all the topks of the candidates
            topks = find_k_highest_values(cand_probs, beam_size)
            
            # update path
            new_candidates = []
            new_probs = []
            for pos in topks:

                if candidates:
                    # pos[0] is the position 0f the tensor thus the index in candidates
                    prev = [x for x in candidates[pos[0]]]
                    prev.append(words[pos[0], pos[1]].item())
                    new_candidate = prev
                else:
                    # initially, new_candidates is just a list with beam size many words with the highest prob
                    new_candidate = [words[pos[0], pos[1]].item()]

                if words[pos[0], pos[1]].item() != config.index_sentence_end:
                    new_candidates.append(new_candidate)
                    new_probs.append(pos[2])
                else:
                    # special case: the generated word is </s>
                    # i.e., a complete sentence is generated
                    # add sentence to best and decrement beam_size (expand only k-1 other candidates)
                    # length normalization only here when end of sentence is reached
                    best[tuple(new_candidate)] = pos[2] / (idx + 1)
                    #best[tuple(new_candidate)] = pos[2]
                    beam_size -= 1

            candidates = new_candidates
            #print("candidates:", candidates)
            cand_probs = new_probs
            #print("new_cand_probs", cand_probs)


            idx += 1

    best = dict(sorted(best.items(), key=lambda item: item[1], reverse=True))
    
    return best


def search(model, model_path, src_sentence, align, src_dict, tgt_dict, mode='beam', n_best=1, remove_bpe=False) -> list[list[str]]:
    """
    Args:
        model: initialized model
        model_path: path to a checkpoint
        src_sentence: a list of str
        align: _description_
        mode: 'beam' oder'greedy'. Defaults to 'beam'.
        n_best: number of hypos in the output list. Defaults to 1.
        remove_bpe: whether to remove bpe tokenization. Defaults to False.

    Returns:
        a list of hypotheses (list of str)
    """
    # preprocessing the string
    # list of *a* list of indices
    src_sentence = batch.text_to_array([src_sentence.split()], src_dict)

    # load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # apply search
    model.eval()
    with torch.no_grad():
        tgt_sentences = []
        if mode == 'beam':
            candidates_dict = beam_search(model, src_sentence[0],
                                      alignment=align)

            # transforms dict.keys to list of lists, each list contains one possible translation
            # (values are probs that are no longer needed)

            candidates = [list(cand) for cand in list(candidates_dict.keys())]

            tgt_sentences = candidates[:n_best]

        else:
            tgt_sentences = [greedy_search(model, src_sentence[0],
                                       alignment=align)]

    # transforms list of lists of indices to list of lists of strings
    # each list of list contains a sentence
    tgt_sentences = batch.array_to_text(tgt_sentences, tgt_dict)

    # optional reversal of BPE tokenization
    if remove_bpe == True:
        tgt_sentences = BPE.bpe_reverse(tgt_sentences)

    return tgt_sentences


def generate_translation(model: nn.Module, checkpoint_path: str, src: list[list[str]], align, src_dict, tgt_dict, mode='beam') -> list[list[str]]:
    translation = []
    for src_sentence in src:
        hypo = search(model, checkpoint_path, ' '.join(src_sentence),
                                        align, src_dict, tgt_dict, mode=mode, n_best=1, remove_bpe=True)
        #print(hypo, "\n\n")
        translation.append(hypo[0])
    return translation


if __name__ == '__main__':
    training_src = tl.loadData('data/bpe_de_7000.txt')
    training_tgt = tl.loadData('data/bpe_en_7000.txt')

    align = alignment(training_src, training_tgt)
