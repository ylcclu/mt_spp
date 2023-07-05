#from feedforward import FeedForwardNet
import torch
import torch.nn as nn
import numpy as np
import batch # for preprocessing src sentence string
from bpe import BPE # for posrprocessing
import textloader as tl
import config


def greedy_search(encoder, decoder, src_sentence):
    encoder.eval()
    decoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_tensor = torch.tensor([src_sentence])

        encoder_outputs, encoder_hidden = encoder(input_tensor.to(device=device))
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == config.index_sentence_end:
                break
            decoded_words.append(idx.tolist())

    return decoded_words



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


def beam_search(encoder, decoder, src_sentence):
    beam_size = config.beam_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = []  # list of list of word indices, candidates has beam_size many elements (= paths)
    cand_probs = []  # probability of those lists of word indices as a subsentence
    best = {}  # contains complete sentences; key is list of word indices, value is joint prob

    # sets the model to evaluation mode to ensure consistent predictions
    # deactivates certain layers that are only needed for training and behave differently in evaluation mode
    encoder.eval()
    decoder.eval()

    #  do we need this?
    src_tensor = torch.tensor([src_sentence])

    # index of the word generated
    idx = 0

    with torch.no_grad():
        # deactivates SGD

        # completely enroll the encoder
        encoder_outputs, encoder_hidden = encoder(src_tensor.to(device))

        while beam_size != 0:

            # initialize target tensor with SOS
            # has to be shape torch.Size([64, 1])
            target_tensor = torch.empty(config.batch_size, 1, dtype=torch.long, device=device).fill_(
                config.index_sentence_start)


            # start decoder with context in target_tensor
            decoder_outputs, _, _ = decoder.forward_step(
                target_tensor, encoder_hidden, encoder_outputs
            )

            #TODO Wieso wirft er einen Fehler, obwohl target_tensor die richtige Dim hat???

            if candidates:
                target_tensor_list = [target_tensor for path in candidates]


            # tensor with beam-size many probs
            _, topi = decoder_outputs.topk(beam_size)

            # update path
            new_candidates = []
            new_probs = []

            # for pos in topi:
            #
            # break

    # best = dict(sorted(best.items(), key=lambda item: item[1], reverse=True))

    # return best


def search(encoder, decoder, model_path, src_sentence, src_dict, tgt_dict, mode='beam', n_best=1, remove_bpe=False) -> list[list[str]]:
    """
    Args:
        model: initialized model
        model_path: path to a checkpoint
        src_sentence: a list of str
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
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # apply search
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        tgt_sentences = []
        if mode == 'beam':
            pass
            #candidates_dict = beam_search(e, src_sentence[0],
             #                         alignment=align)

            # transforms dict.keys to list of lists, each list contains one possible translation
            # (values are probs that are no longer needed)

            #candidates = [list(cand) for cand in list(candidates_dict.keys())]

            #tgt_sentences = candidates[:n_best]

        else:
            tgt_sentences = [greedy_search(encoder, decoder, src_sentence[0])]

    # transforms list of lists of indices to list of lists of strings
    # each list of list contains a sentence
    tgt_sentences = batch.array_to_text(tgt_sentences, tgt_dict)

    # optional reversal of BPE tokenization
    if remove_bpe == True:
        tgt_sentences = BPE.bpe_reverse(tgt_sentences)

    return tgt_sentences


def generate_translation(encoder, decoder, checkpoint_path: str, src: list[list[str]], src_dict, tgt_dict, mode='greedy') -> list[list[str]]:
    translation = []
    for src_sentence in src:
        hypo = search(encoder, decoder, checkpoint_path, ' '.join(src_sentence),
                                        src_dict, tgt_dict, mode=mode, n_best=1, remove_bpe=True)
        translation.append(hypo[0])
    return translation
