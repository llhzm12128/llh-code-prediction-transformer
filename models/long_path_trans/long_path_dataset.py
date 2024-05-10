#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from dataset import PathBaseDataset, BaseSetup, BaseVocab


class Setup(BaseSetup):
    def _create_vocab(self):
        return Vocab(self.filepaths["vocab"])

    def _create_dataset(self, fp, ids_fp,level_fp = None):
        return Dataset(fp, ids_fp)


class Vocab(BaseVocab):
    def convert(self, line):
        leaf_tokens, ext, paths, leaf_type = line
        leaf_tokens_conv = [
            self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
            for token in leaf_tokens
        ]
        paths_conv = [
            [
                self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
                for token in path
            ]
            for path in paths
        ]
        return [leaf_tokens_conv, ext, paths_conv, leaf_type]


class Dataset(PathBaseDataset):
    #原batch打包函数
    """ @staticmethod
    def collate(seqs, pad_idx, bos_idx=None):
        def combine_root_paths(root_paths, max_len, max_path_len):
            paths = []
            for path in root_paths:
                paths.append(path + [pad_idx] * (max_path_len - len(path)))
            len_pad = torch.ones((max_len - len(paths), max_path_len)).long()
            return torch.cat((torch.tensor(paths), len_pad))

        max_len = max(len(seq[0][0]) for seq in seqs)
        max_len = max(2, max_len)
        max_path_len = max(max(len(path) for path in seq[0][2]) for seq in seqs)
        max_path_len - max(2, max_path_len)
        input_seqs = []
        target_seqs = []
        extended = []
        root_path_seqs = []
        ids = {name: [] for name in seqs[0][1].keys()}
 
        for i, ((seq, ext, root_paths), ids_lst) in enumerate(seqs):
            padding = [pad_idx] * (max_len - len(seq))
            input_seqs.append(seq[:-1] + padding)
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)
            root_path_seqs.append(combine_root_paths(root_paths, max_len, max_path_len))
            for name, lst in ids_lst.items():
                ids[name] += [j - 1 + (max_len - 1) * i for j in lst]

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended),
            "root_paths": torch.stack(root_path_seqs)[:, 1:, :],
            "ids": ids,
        } """
        
    #删除ids的batch打包函数
    @staticmethod
    def collate(seqs, pad_idx, bos_idx=None):
        def combine_paths(root_paths, max_len, max_path_len):
            paths = []
            for path in root_paths:
                paths.append(path + [pad_idx] * (max_path_len - len(path)))
            #下面代码的目的是将 paths 序列进行填充，使其达到指定的最大长度 max_len
            # 如果 paths 的长度小于 max_len，则需要进行填充操作。    
            len_pad = torch.ones((max_len - len(paths), max_path_len)).long() 
            return torch.cat((torch.tensor(paths), len_pad))

        max_len = max(len(seq[0]) for seq in seqs)
        max_len = max(2, max_len)
        max_path_len = max(max(len(path) for path in seq[2]) for seq in seqs)
        max_path_len = max(2, max_path_len)
        input_seqs = []
        target_seqs = []
        extended = []
        path_seqs = []
        #ids = {name: [] for name in seqs[0][1].keys()}

        for i, (seq, ext, paths,leaf_type) in enumerate(seqs):
            padding = [pad_idx] * (max_len - len(seq))
            #input删除最后一个元素，并使用padding进行尾部填充，使其长度达到max_len（叶子节点最多的序列长度）
            input_seqs.append(seq[:-1] + padding) 
            #target删除第一个元素，并使用padding进行尾部填充，使其长度达到max_len（叶子节点最多的序列长度）
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)
            #将每个 path 进行填充，使其达到指定的最大长度 max_path_len
            # 如果 path 的长度小于 max_len-1，则需要进行填充操作
            #然后对将 paths 序列进行填充，使其达到指定的最大长度 max_len-1
            #减一是因为paths的长度本来就比seq的长度小一
            path_seqs.append(combine_paths(paths, max_len-1, max_path_len))
            
            new_path_seqs = torch.stack(path_seqs)[:, :, :]
            #input删除最后一个元素，target删除第一个元素，paths删除第一个path

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended),
            #将root_path_seqs去除第一个path
            "paths": new_path_seqs,
            "leaf_type":leaf_type[1:],
        }
        


