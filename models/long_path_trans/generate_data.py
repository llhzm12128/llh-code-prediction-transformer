#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import argparse
import json
import logging
import os
import sys
from itertools import chain, combinations, product
sys.path.append('/root/llh-code-prediction-transformer')

from utils import file_tqdm
from utils import file_tqdm, get_dfs, separate_dps, get_terminal_nodes


logging.basicConfig(level=logging.INFO)
#todo
#针对每个AST生成多个dp，通过增加默认的路径长度的方式，尽可能多的获取端到端路径
#1.获取AST所有端到端路径（把路径长度延长，尽可能多的获取路径）（代码重用）
#2.针对每个叶子节点尽量都获取一条端到端路径对（注意满足前一个路径对的右端是后一个路径对的左端，如果某个叶子节点没有路径对，则用一条根到叶子的路径代替，同时左叶子和右叶子和使用路径对相同）
#3.路径对的获取：优先使用code2seq的路径对，把最大长度设置为26，计算一下包含根节点的路径对百分比（后续可以减少或者增加路径长度进行实验，长度过长可能会由于代码局部性而使准确率降低）
# （路径对之间按序）（获取dp）（注意可能不是所有叶子节点都有对用的路径对）
#4.使用和path_trans类似的批处理方式


""" def get_leaf_type(ast,leaf_ids):
    leaf_type = []
    for id in leaf_ids:
        for i, node in enumerate(ast):
            if(id == i+1):
                if node["type"] == "attr":
                    leaf_type.append("attr")  
                elif node["type"] == "Num":
                    leaf_type.append("num")
                elif node["type"] in {"NameLoad", "NameStore"}:
                    leaf_type.append("name")
                elif node["type"] == "NameParam":
                    leaf_type.append("param")
                elif node["type"] == "Str":
                    leaf_type.append("str")
                else:
                    leaf_type.append(node["type"])
                
    return leaf_type
 """
#返回叶子节点对应的type
def get_leaf_type(ast,leaf_ids):
    leaf_type = []
    for i,node in enumerate(ast):
        if "type" in node and i+1 in leaf_ids:
            if node["type"] == "attr":
                leaf_type.append("attr")  
            elif node["type"] == "Num":
                leaf_type.append("num")
            elif node["type"] in {"NameLoad", "NameStore"}:
                leaf_type.append("name")
            elif node["type"] == "NameParam":
                leaf_type.append("param")
            elif node["type"] == "Str":
                leaf_type.append("str")
            else:
                leaf_type.append(node["type"])
                
    return leaf_type

#获取指定node的type或value
def get_value(node):
    return node["value"] if "value" in node else node["type"]

def get_leaf_info(ast):
    leaf_tokens = []
    leaf_ids = []
    
    for i, node in enumerate(ast):
        if "value" in node:
            leaf_ids.append(i)
            leaf_tokens.append(node["value"])
    return leaf_tokens, leaf_ids

def extract_paths(ast, max_length):
    def dfs(i):
        node = ast[i]
        if "children" not in node:
            full_paths = []
            half_paths = [[i]]
        else:
            children = node["children"]
            child_to_full_paths, child_to_half_paths = zip(
                *(dfs(child_id) for child_id in children)
            )
            full_paths = list(chain.from_iterable(child_to_full_paths))
            for i_child in range(len(children) - 1):
                for j_child in range(i_child + 1, len(children)):
                    i_child_half_paths = child_to_half_paths[i_child]
                    j_child_half_paths = child_to_half_paths[j_child]
                    for i_half_path, j_half_path in product(
                        i_child_half_paths, j_child_half_paths
                    ):
                        path_len = len(i_half_path) + len(j_half_path) + 1
                        if path_len > max_length:
                            continue
                        path = list(chain(i_half_path, [i], reversed(j_half_path))) 
                        full_paths.append(path)
            half_paths = [
                half_path + [i]
                for half_path in chain.from_iterable(child_to_half_paths)
                if len(half_path) + 1 < max_length
            ]
        return full_paths, half_paths

    return dfs(0)[0]


def get_all_paths(ast, all_paths):
    #ast_values = [get_value(i) for i in ast]#ast的所有节点
    #terminal_words = [get_value(ast[i]) for i in get_terminal_nodes(ast)] #ast的所有叶子节点
    #tokenized_words = {word: tokenize(word) for word in terminal_words} #ast叶子节点token化
    node_to_path_idx = {i: [] for i in range(len(ast))} #每个叶子节点对应的所有路径对的索引（中间节点没有路径对）
    for i, path in enumerate(all_paths):
        node_to_path_idx[path[0]].append(i) #获取start叶子节点对应的路径对索引
    return node_to_path_idx

""" def get_ancestors(ast):
    ancestors = {0: []}
    node2parent = {0: 0}
    for i, node in enumerate(ast):
        if "children" in node:
            for child in node["children"]:
                node2parent[child] = i
        token = node["value"] if "value" in node else node["type"]
        ancestors[i] = [token] + ancestors[node2parent[i]]
    return ancestors
 """
#根据左叶子节点，获取路径中叶子节点相邻且最长的路径,并将id转为token
def get_paths(node_to_path_idx, leaf_ids, all_paths, ast_values):
    id_paths = []
    for i ,ids in enumerate(leaf_ids):
        if (i<=len(leaf_ids)-2):#最后一个叶子节点没有对应的端到端路径
            temp = []
            for path_index in node_to_path_idx[ids]:
                if(all_paths[path_index][-1] == leaf_ids[i+1]):#路径的两个叶子节点在ast中是相邻的
                    temp.append(all_paths[path_index])
            assert(len(temp)>0)
            path = max(temp, key=len)
            id_paths.append(path[1:-1])#获取满足上面条件的最长路径,并且在路径中删除双端的叶子节点
    
    #将path对的id转为token
    token_paths = [
            [ast_values[i] for i in p] for p in id_paths
        ]
    return token_paths       
            


def get_dps(ast, max_len, max_path_len):
    ast_values = [get_value(node) for node in ast]#ast的所有节点
    leaf_tokens, leaf_ids = get_leaf_info(ast)
    leaf_types = get_leaf_type(ast,leaf_ids)
    #ancestors = get_ancestors(ast) 
    #print(len(leaf_tokens))
    #print(len(leaf_types))
    all_paths = extract_paths(ast, max_path_len) #提取ast所有端到端路径，中间节点不一定是根节点
    node_to_path_idx = get_all_paths(ast, all_paths) #每个叶子节点对应的路径对索引
    
    #return dp：[leaf_start,paths,leaf_end,ext,leaf_type](注：在对dp进行batch打包时，不需要删除start的最后一个，end的第一个，和paths的第一个)
    assert(len(leaf_tokens) == len(leaf_types))
    if len(leaf_tokens) <= max_len:
        return [[leaf_tokens, 0, get_paths(node_to_path_idx, leaf_ids, all_paths,ast_values), leaf_types]]

    half_len = int(max_len / 2)
    aug_dps = [
        [
            leaf_tokens[:max_len],
            0,
            get_paths(node_to_path_idx, leaf_ids[:max_len], all_paths,ast_values),
            leaf_types[:max_len],
        ]
    ]
    i = half_len
    while i < len(leaf_tokens) - max_len:
        aug_dps.append(
            [
                leaf_tokens[i : i + max_len],
                half_len,
                get_paths(node_to_path_idx, leaf_ids[i : i + max_len], all_paths, ast_values),
                leaf_types[i : i + max_len],
            ]
        )
        i += half_len
    idx = max_len - (len(leaf_tokens) - (i + half_len))
    aug_dps.append(
        [
            leaf_tokens[-max_len:],
            idx,
            get_paths(node_to_path_idx, leaf_ids[-max_len:], all_paths, ast_values),
            leaf_types[-max_len:],
        ]
    )
    return aug_dps



def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath with the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    parser.add_argument(
        "--max_path_len",
        "-p",
        type=int,
        default=13,
        help="Max length of rootpath route",
    )

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Writing dps to: {}".format(args.out_fp))

    num_dps = 0
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            for dp in get_dps(dp, args.n_ctx, args.max_path_len):
                if len(dp[0]) > 1:
                    #json中保存从AST中提取的路径
                    #每条json数据的结构为：[leaf_tokens列表(一维列表)，已经计算过loss的token的数量(int)，提取的路径列表(二维列表)]
                    json.dump(dp, fout)    
                    fout.write("\n")
                    num_dps += 1

    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
