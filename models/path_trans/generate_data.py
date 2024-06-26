#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
* Function: PathTrans模型的数据预处理；AST包含的叶子node数量最大为1000，每条路径的最大长度为13
* detail: 对于叶子node数量大于1000的AST，使用滑动窗口将其拆分为多个包含1000个node的AST。对于大于13的路径，保留叶子节点截断根节点
'''

import argparse
import json
import logging
import os
import sys
import platform
if(platform.system() == "Windows"):
    sys.path.append("C:/Users/llh/Desktop/ISCAS/llh-code-prediction-transformer")
else:
    sys.path.append("/root/llh-code-prediction-transformer")

from utils import file_tqdm, get_dfs, separate_dps



logging.basicConfig(level=logging.INFO)
'''
在原generate_data.py返回值的基础上添加叶子节点对应的type list
'''


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
    
#获取所有叶子节点
def get_leaf_info(ast):
    leaf_tokens = []
    leaf_ids = []
    
    for i, node in enumerate(ast):
        if "value" in node:
            leaf_ids.append(i)
            leaf_tokens.append(node["value"])
    return leaf_tokens, leaf_ids


def get_ancestors(ast):#ancestors中包括叶子节点
    ancestors = {0: []}
    node2parent = {0: 0}
    for i, node in enumerate(ast):
        if "children" in node:
            for child in node["children"]:
                node2parent[child] = i
        token = node["value"] if "value" in node else node["type"]
        ancestors[i] = [token] + ancestors[node2parent[i]]
    return ancestors


def get_root_paths(ancestors, leaf_ids, max_path_len):
    return [ancestors[i][1 :max_path_len + 1] for i in leaf_ids]


def get_dps(ast, max_len, max_path_len):
     
    leaf_tokens, leaf_ids = get_leaf_info(ast)
    leaf_types = get_leaf_type(ast,leaf_ids)
    ancestors = get_ancestors(ast) #ancestors包含叶子节点,path的方向是从孩子节点到根节点
    #print(len(leaf_tokens))
    #print(len(leaf_types))
    assert(len(leaf_tokens) == len(leaf_types))
    if len(leaf_tokens) <= max_len:
        #root_path的长度为max_path_len，且不包括叶子节点
        return [[leaf_tokens, 0, get_root_paths(ancestors, leaf_ids, max_path_len), leaf_types]] #路径中不包含叶子节点

    half_len = int(max_len / 2)
    aug_dps = [
        [
            leaf_tokens[:max_len],
            0,
            get_root_paths(ancestors, leaf_ids[:max_len], max_path_len),
            leaf_types[:max_len],
        ]
    ]
    i = half_len
    while i < len(leaf_tokens) - max_len:
        aug_dps.append(
            [
                leaf_tokens[i : i + max_len],
                half_len,
                get_root_paths(ancestors, leaf_ids[i : i + max_len], max_path_len),
                leaf_types[i : i + max_len],
            ]
        )
        i += half_len
    idx = max_len - (len(leaf_tokens) - (i + half_len))
    aug_dps.append(
        [
            leaf_tokens[-max_len:],
            idx,
            get_root_paths(ancestors, leaf_ids[-max_len:], max_path_len),
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
        help="Max length of rootpath route, Excluding leaf nodes",
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
