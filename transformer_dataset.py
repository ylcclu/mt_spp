import batch, config
from dictionary import Dictionary
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad

class TransformerDataset(Dataset):

    def __init__(self, src_text: list[list[str]], tgt_text: list[list[str]], 
                 src_dict: Dictionary, tgt_dict: Dictionary) -> None:
        # turn text into np array of indices in dictionary
        src_array = batch.text_to_array(src_text, src_dict)
        tgt_array = batch.text_to_array(tgt_text, tgt_dict)

        bs_id = torch.tensor([config.index_sentence_start])  # <s> token id
        eos_id = torch.tensor([config.index_sentence_end])  # </s> token id
        src_list = []
        tgt_list = []

        for src_, tgt_ in zip(src_array, tgt_array):
            # add <s> and </s> to src and tgt sentence  
            src = torch.cat(
                [
                    bs_id,
                    torch.tensor(src_, dtype=torch.int64),
                    eos_id
                ],
                0
            )
            tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(tgt_, dtype=torch.int64),
                    eos_id
                ],
                0
            )

            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(
                    src,
                    (
                        0,
                        config.MAX_PADDING - len(src),
                    ),
                    value=config.index_padding,
                )
            )
            tgt_list.append(
                pad(
                   tgt,
                    (0, config.MAX_PADDING - len(tgt)),
                    value=config.index_padding,
                )
            )
        
        self.src = torch.stack(src_list)
        self.tgt = torch.stack(tgt_list)

        self.n_samples = self.src.shape[0]

    
    def __getitem__(self, index):
        return self.src[index], self.tgt[index]
    
    def __len__(self) -> int:
        return self.n_samples
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    
def subsequent_mask(size):
    "Mask out subsequent positions during decoding"
    "to prevent leftward information flow in the decoder to preserve the auto-regressive property"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # triu returns an upper triangle
    return subsequent_mask == 0
    # returns a boolean tensor where every element above the diagonal is False
    #                                and every other is True

class Seq2SeqBatch:
    """Object for holding a batch of data with mask during training"""

    def __init__(self, src, tgt=None, pad=config.index_padding):
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
        
def greedy_search(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        print(f"subsequent_mask(ys.size(1)).type_as(src.data) {subsequent_mask(ys.size(1)).type_as(src.data)}")
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

def beam_search():
    return

def search(model, src, src_mask, max_len, start_symbol, mode='beam'):
    model.eval()
    with torch.no_grad():
        if mode == 'beam':
            return beam_search()
        elif mode == 'greedy':
            return greedy_search(model, src, src_mask, max_len, start_symbol)

from batch import array_to_text
from dictionary import remove_special_symbols

def generate_translation(trained_model, dev_loader, vocab_tgt: Dictionary):

    translations = []

    for (src, tgt) in dev_loader:
        batch = Seq2SeqBatch(src, tgt, config.index_padding)
        model_out = search(trained_model, batch.src, batch.src_mask, config.max_length, config.index_sentence_start, mode='greedy')[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_word(x) for x in model_out if x != config.index_padding]
            ).split(config.sentence_end, 1)[0]
            + config.sentence_end
        )
        translations.append(model_txt)

    return remove_special_symbols(translations)

