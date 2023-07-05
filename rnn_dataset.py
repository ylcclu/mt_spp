import batch, config
from dictionary import Dictionary
import textloader as tl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

class RNNDataset(Dataset):

    def __init__(self, src_text: list[list[str]], tgt_text: list[list[str]], 
                 src_dict: Dictionary, tgt_dict: Dictionary) -> None:
        # super().__init__()
        # turn text into np array of indices in dictionary
        src_array = batch.text_to_array(src_text, src_dict)
        tgt_array = batch.text_to_array(tgt_text, tgt_dict)

        # build S, T, and L as list of lists of the same length (2*w+1, w, or 1)
        # each list element is a row in the corresponding matrix
        S_lists, T_lists = [], []

        # import settings in config for the building of matrices
        for src, tgt in zip(src_array, tgt_array):
            s = [src[m] if m in range(0,len(src)) else config.index_sentence_end for m in range(len(src))]
            t = [tgt[m] if m in range(0,len(tgt)) else config.index_sentence_end for m in range(len(src))]
            S_lists.append(torch.tensor(s))
            T_lists.append(torch.tensor(t))
        # cast lists into np array, then into tensor
        self.S = pad_sequence(S_lists, batch_first=True, padding_value=config.index_padding)
        self.T = pad_sequence(T_lists, batch_first=True, padding_value=config.index_padding)
        # print(self.S)
        # print(self.T)
        # print(self.L)
        # print(f"shape of S: {self.S.shape}")
        # print(f"shape of T: {self.T.shape}")
        # print(f"shape of L: {self.L.shape}")

        # n_samples as dim of tensor
        self.n_samples = self.S.shape[0]

    
    def __getitem__(self, index):
        return self.S[index], self.T[index]
    
    def __len__(self) -> int:
        return self.n_samples
    
class TransformerDataset(Dataset):

    def __init__(self, src_text: list[list[str]], tgt_text: list[list[str]], 
                 src_dict: Dictionary, tgt_dict: Dictionary, device) -> None:
        # turn text into np array of indices in dictionary
        src_array = batch.text_to_array(src_text, src_dict)
        tgt_array = batch.text_to_array(tgt_text, tgt_dict)

        bs_id = torch.tensor([config.index_sentence_start], device=device)  # <s> token id
        eos_id = torch.tensor([config.index_sentence_end], device=device)  # </s> token id
        src_list = []
        tgt_list = []

        for src, tgt in zip(src_array, tgt_array):
            src = torch.cat(
                [
                    bs_id,
                    torch.tensor(src, dtype=torch.int64, device=device),
                    eos_id
                ],
                0
            )
            tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(tgt, dtype=torch.int64, device=device),
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
        
        self.src = src_list
        self.tgt = tgt_list

        self.n_samples = self.src.shape[0]

    
    def __getitem__(self, index):
        return self.src[index], self.tgt[index]
    
    def __len__(self) -> int:
        return self.n_samples


if __name__ == '__main__':

    src = tl.loadData('bpe_de_7000.txt')
    tgt = tl.loadData('bpe_en_7000.txt')
    src_dict = Dictionary(src)
    tgt_dict = Dictionary(tgt)
    textdataset = RNNDataset(src_dict, tgt, src_dict, tgt_dict)

    batch_size = config.batch_size
    dataloader = DataLoader(dataset=textdataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)

