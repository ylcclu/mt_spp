import batch, config
from dictionary import Dictionary

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):

    def __init__(self, src_text: list[list[str]], tgt_text: list[list[str]], 
                 src_dict: Dictionary, tgt_dict: Dictionary) -> None:
        # super().__init__()

        # turn text into np array of indices in dictionary
        src_array = batch.text_to_array(src_text, src_dict)
        tgt_array = batch.text_to_array(tgt_text, tgt_dict)

        # build S, T, and L as list of lists of the same length (2*w+1, w, or 1)
        # each list element is a row in the corresponding matrix
        S_lists, T_lists, L_lists = [], [], []

        # import settings in config for the building of matrices
        win_size = config.window_size
        start_idx = config.index_sentence_start
        end_idx = config.index_sentence_end

        for src, tgt in zip(src_array, tgt_array):
            for idx in range(len(tgt)+1):
                # iterate from 0 to len(tgt)
                # instead of to len(tgt)-1
                align = batch.alignment(idx, len(tgt), len(src))
                S_list = [src[m] if m in range(0,len(src)) else start_idx if m<0 else end_idx for m in range(align-win_size, align+win_size+1)]
                T_list = [tgt[m] if m in range(0,len(tgt)) else start_idx if m<0 else end_idx for m in range(idx-win_size, idx)]

                S_lists.append(S_list)
                T_lists.append(T_list)
                # L_lists.append([tgt[idx] if idx < len(tgt) else end_idx])

            # add e_{len(tgt)+1} to the target label
            tgt.append(end_idx)
            L_lists.extend(tgt)
                
        # cast lists into np array, then into tensor
        self.S = torch.tensor(S_lists)
        self.T = torch.tensor(T_lists)
        self.L = torch.tensor(L_lists)
        # print(self.S)
        # print(self.T)
        # print(self.L)
        # print(f"shape of S: {self.S.shape}")
        # print(f"shape of T: {self.T.shape}")
        # print(f"shape of L: {self.L.shape}")

        # n_samples as dim of tensor
        self.n_samples = self.S.shape[0]

    
    def __getitem__(self, index):
        return self.S[index], self.T[index], self.L[index]
    
    def __len__(self) -> int:
        return self.n_samples
    
if __name__ == '__main__':
    textdataset = TextDataset('data/bpe_de_7000.txt', 'data/bpe_en_7000.txt')

    batch_size = config.batch_size
    dataloader = DataLoader(dataset=textdataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
    
    dataiter = iter(dataloader)
    data = next(dataiter)
    src_win, tgt_win, tgt_lb = data
    # print(f"src_win: {src_win}, {src_win.shape}")
    # print(f"tgt_win: {tgt_win}, {tgt_win.shape}")
    # print(f"tgt_lb: {tgt_lb}, {tgt_lb.shape}")
