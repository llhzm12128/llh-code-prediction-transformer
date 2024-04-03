import argparse
import json
import os
import pickle
import random
import re
from collections import defaultdict
from itertools import chain, combinations, product
import sys
sys.path.append('/root/llh-code-prediction-transformer')

from utils import get_ancestors, get_terminal_nodes, parallelize, tokenize
from tqdm import tqdm


PLACEHOLDER = "<placeholder_token>"
UNK = "<unk_token>"


def get_leaf_nodes(ast, id_type):
    # get ids for special leaf types: attr, num, name, param
    if id_type == "attr":
        types_ = {"attr"}
    elif id_type == "num":
        types_ = {"Num"}
    elif id_type == "name":
        types_ = {"NameLoad", "NameStore"}
    elif id_type == "param":
        types_ = {"NameParam"}

    nodes = []
    for i, node in enumerate(ast):
        if "type" in node and node["type"] in types_:
            nodes.append(i + 1)
    return nodes


def get_value(d):
    return d["value"] if "value" in d else d["type"]


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


def get_all_paths(ast, id_type, max_path_len, max_num_paths):
    if id_type == "leaves":
        nodes = get_terminal_nodes(ast) #获取所有叶子节点id
    else:
        nodes = get_leaf_nodes(ast, id_type) #获取指定类型的叶子节点id
    if not nodes:
        return []
    
    all_paths = extract_paths(ast, max_path_len) #提取ast所有端到端路径，中间节点不一定是根节点
    ast_values = [get_value(i) for i in ast]#ast的所有节点
    terminal_words = [get_value(ast[i]) for i in get_terminal_nodes(ast)] #ast的所有叶子节点
    tokenized_words = {word: tokenize(word) for word in terminal_words} #ast叶子节点token化
    node_to_path_idx = {i: [] for i in range(len(ast))} #每个叶子节点对应的所有路径对的索引（中间节点没有路径对）
    for i, path in enumerate(all_paths):
        node_to_path_idx[path[-1]].append(i)
    
    dps = []
    paths_to_choose_from = [] #某个叶子节点之前的所有路径对
    prev_node = 0
    for node in nodes:
        for j in range(prev_node, node):
            paths_to_choose_from += [
                all_paths[path_i] for path_i in node_to_path_idx[j]
            ]
        prev_node = node

        paths_to_here = [all_paths[path_i] for path_i in node_to_path_idx[node]] #右终端节点刚好在node的路径对
        if len(paths_to_choose_from) + len(paths_to_here) <= max_num_paths:  #两个列表长度之和小于200，则连接两个列表 
            paths = paths_to_choose_from.copy() + paths_to_here
        else:
            if len(paths_to_here) > max_num_paths: #到目标右孩子节点的路径对个数大于200，则从中随机选择200个路径对
                paths = random.sample(paths_to_here, max_num_paths)
            else:
                paths = paths_to_here + random.sample(#从之前的路径对中随机选择
                    paths_to_choose_from, max_num_paths - len(paths_to_here)
                )

        # convert to vocab
        target = ast_values[node]
        #将path对的id转为token，目标叶子token用PLACEHOLDER代替
        paths = [
            [ast_values[i] if i != node else PLACEHOLDER for i in p] for p in paths
        ]
        lefts = [tokenized_words[p[0]] for p in paths]  #将路径对的左叶子节点token化
        #将路径对的右叶子节点token化，若右节点为PLACEHOLDER则返回PLACEHOLDER
        rights = [
            tokenized_words[p[-1]] if p[-1] != PLACEHOLDER else [PLACEHOLDER]
            for p in paths
        ]
       
        #target：目标token
        #lefts：二维数组，两个目标node之间的所有路径对的左孩子的token化结果
        #paths：二维数组，两个目标node之间的所有路径对（包含左右叶子节点）
        #rights：二维数组，两个目标node之间的所有路径对的右孩子的token化结果
        dps.append([target, lefts, paths, rights])
    return dps


def get_word2idx(out_fp):
    with open(out_fp, "rb") as fin: 
        vocab = pickle.load(fin)
    word2idx = {word: i for i, word in enumerate(vocab)}
    word2idx = defaultdict(lambda: word2idx[UNK], word2idx)
    print("Read vocab from: {}".format(out_fp))
    return word2idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate terminal to terminal paths from AST"
    )
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath for the output dps"
    )
    parser.add_argument("--max_path_len", type=int, default=9, help="Max path len.")
    parser.add_argument("--max_num_paths", type=int, default=200)
    parser.add_argument("--base_dir", "-b", type=str)
    parser.add_argument(
        "id_type",
        choices=["attr", "num", "name", "param", "leaves"],
        default="attr",
        help="Which ids to generate. Default = attr",
    )
    args = parser.parse_args()
    print("Max path len: {}".format(args.max_path_len))
    print("Max num paths: {}".format(args.max_num_paths))
    print("Writing to {}".format(args.out_fp))

    # read the vocabs
    base_dir = args.base_dir
    token_vocab = get_word2idx(os.path.join(base_dir, "token_vocab.pkl"))
    subtoken_vocab = get_word2idx(os.path.join(base_dir, "subtoken_vocab.pkl"))
    output_vocab = get_word2idx(os.path.join(base_dir, "output_vocab.pkl"))

    data = []
    i = 0
    c = 0
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for _ in range(20):
            i += 1
            print("Starting {} / 50".format(i))
            for _ in range(5000):
                dp = json.loads(f.readline().strip())
                if len(dp) <= 1:
                    continue
                data.append(dp)
            print(" > Finished reading: {}".format(len(data)))
            for ast in tqdm(data):
                dp = get_all_paths(ast, args.id_type, args.max_path_len, args.max_num_paths)
                for target, lefts, paths, rights in dp:
                    target = output_vocab[target]
                    lefts = [[subtoken_vocab[t] for t in lst] for lst in lefts]
                    paths = [[token_vocab[t] for t in lst] for lst in paths]
                    rights = [[subtoken_vocab[t] for t in lst] for lst in rights]

                    json.dump([target, lefts, paths, rights], fout)
                    fout.write("\n")
                    c += 1
            data = []
            print(" > Finished writing to file")
    print("Wrote {} datapoints to {}".format(c, args.out_fp))


if __name__ == "__main__":
    main()
