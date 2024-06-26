#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
sys.path.append("C:/Users/llh/Desktop/ISCAS/llh-code-prediction-transformer")
from dataset import PathBaseDataset, BaseSetup, BaseVocab


class Setup(BaseSetup):
    def _create_vocab(self):
        return Vocab(self.filepaths["vocab"])

    def _create_dataset(self, fp, ids_fp,level_fp = None):
        return Dataset(fp, ids_fp)


class Vocab(BaseVocab):
    def convert(self, line):
        dp, ext, root_paths, leaf_type = line
        dp_conv = [
            self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
            for token in dp
        ]
        root_paths_conv = [
            [
                self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
                for token in path
            ]
            for path in root_paths
        ]
        return [dp_conv, ext, root_paths_conv, leaf_type]


class Dataset(PathBaseDataset):
        
    #将多个dp打包到一个batch中
    @staticmethod
    def collate(seqs, pad_idx, bos_idx=None):
        def combine_root_paths(root_paths, max_len, max_path_len):
            paths = []
            for path in root_paths:
                paths.append(path + [pad_idx] * (max_path_len - len(path)))
            #下面代码的目的是将 paths 序列进行填充，使其达到指定的最大长度 max_len
            # 如果 paths 的长度小于 max_len，则需要进行填充操作。    
            len_pad = torch.ones((max_len - len(paths), max_path_len)).long() 
            return torch.cat((torch.tensor(paths), len_pad))

        max_len = max(len(seq[0]) for seq in seqs) #多个dp中最大的叶子数量
        max_len = max(2, max_len)
        max_path_len = max(max(len(path) for path in seq[2]) for seq in seqs) #多个dp中最大的路径长度
        max_path_len = max(2, max_path_len)
        input_seqs = []
        target_seqs = []
        extended = []
        root_path_seqs = []
        #ids = {name: [] for name in seqs[0][1].keys()}

        for i, (seq, ext, root_paths,leaf_type) in enumerate(seqs):
            padding = [pad_idx] * (max_len - len(seq))
            #input删除最后一个元素，并使用padding进行尾部填充，使其长度达到max_len（叶子节点最多的序列长度）
            input_seqs.append(seq[:-1] + padding) 
            #target删除第一个元素，并使用padding进行尾部填充，使其长度达到max_len（叶子节点最多的序列长度）
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)
            #将每个 path 进行填充，使其达到指定的最大长度 max_path_len
            # 如果 path 的长度小于 max_len，则需要进行填充操作
            #然后对将 paths 序列进行填充，使其达到指定的最大长度 max_len
            root_path_seqs.append(combine_root_paths(root_paths, max_len, max_path_len))
            #paths删除第一个path
            new_root_path_seqs = torch.stack(root_path_seqs)[:, 1:, :]
            #input删除最后一个元素，target删除第一个元素，paths删除第一个path

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended),
            #将root_path_seqs去除第一个path
            "root_paths": new_root_path_seqs,
            "leaf_type":leaf_type[1:],
        }
        


