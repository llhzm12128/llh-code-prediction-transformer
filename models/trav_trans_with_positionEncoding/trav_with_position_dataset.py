
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from dataset import  BaseSetup, BaseVocab, TravTransPositionEncodingBaseDataset


class Setup(BaseSetup):
    def _create_vocab(self):
        return Vocab(self.filepaths["vocab"])

    def _create_dataset(self, fp, ids_fp, level_fp):
        return Dataset(fp, ids_fp, level_fp)


class Vocab(BaseVocab):
    def convert(self, line):
        dp, ext = line
        dp_conv = [
            self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
            for token in dp
        ]
        return [dp_conv, ext]


class Dataset(TravTransPositionEncodingBaseDataset):
    @staticmethod
    def collate(seqs, pad_idx):
        max_len = max(len(seq[0][0]) for seq in seqs)
        max_len = max(max_len, 2)
        input_seqs = []
        target_seqs = [] 
        extended = []
        level_seqs = []
        ids = {name: [] for name in seqs[0][1].keys()}

        
        for i, ((seq, ext), ids_lst, level_list) in enumerate(seqs):
            padding = [pad_idx] * (max_len - len(seq))
            level_padding = [999] * (max_len - len(seq))
            input_seqs.append(seq[:-1] + padding)
            level_seqs.append(level_list[:-1] + level_padding)
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)
            for name, lst in ids_lst.items():
                ids[name] += [j - 1 + (max_len - 1) * i for j in lst]    #减一操作是为了在eval时能够从y中获取对应的token（因为y相比于x去掉了根节点）

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended),
            "ids": ids,
            "level":torch.tensor(level_seqs)
        }
        
