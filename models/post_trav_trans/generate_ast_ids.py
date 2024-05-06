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
import platform
import sys
if(platform.system() == "Windows"):
    sys.path.append("C:/Users/llh/Desktop/ISCAS/llh-code-prediction-transformer")
else:
    sys.path.append("/root/llh-code-prediction-transformer")

from utils import file_tqdm,split_type_and_index_sequence


logging.basicConfig(level=logging.INFO)



#后序遍历AST，获取value节点的type，同时记录后序序列中value节点和内部节点的index

def postorder_type_and_index(ast):
    dp = [] #记录后序遍历序列，value节点记录父节点的type
    post_index = []  #记录后序序列中value节点和内部节点的index
    def traverse(node, parent_type=None):
        if 'children' in node:
            for child_index in node['children']:
                traverse(ast[child_index],node['type'])
        if 'value' in node:
            dp.append(parent_type)
            post_index.append(0) #0表示value节点
        if 'type' in node:
            dp.append(node['type'])
            post_index.append(1) #1表示type节点
    traverse(ast[0])
    return dp, post_index



def get_leaf_ids(split_ids):
    ids = {"leaf_ids": [], "internal_ids": []}
    for i, flag in enumerate(split_ids):
        if flag == 0:
            ids["leaf_ids"].append(i)
        else:
            ids["internal_ids"].append(i)
    return ids


def get_value_ids(split_type, split_id):
    ids = {"attr_ids": [], "num_ids": [], "name_ids": [], "param_ids": [], "string_ids": []}
    for i, flag in enumerate(split_id):
        if flag == 0:
            if split_type[i] == "attr":
                ids["attr_ids"].append(
                    i 
                )  
            elif split_type[i] == "Num":
                ids["num_ids"].append(i)
            elif split_type[i] in {"NameLoad", "NameStore"}:
                ids["name_ids"].append(i)
            elif split_type[i] == "NameParam":
                ids["param_ids"].append(i)
            # RQ3/RQ4 additional metrics
            elif split_type[i] == "Str":
                ids["string_ids"].append(i)        
    return ids

def get_valueType_ids(split_type, split_id):
    ids = {"type_attr_ids": [], "type_num_ids": [], "type_name_ids": [], "type_param_ids": [], "type_string_ids": []}
    for i, flag in enumerate(split_id):
        stack = []
        if flag == 0 and i+1<len(split_type):
            for index in range(i+1,len(split_type)):
                if(split_id[index] == 0):
                    stack.append(0)
                elif(split_id[index] == 1 and len(stack)!=0):
                    stack.pop()
                elif split_type[i] =="attr" and split_type[index] == "attr" and split_id[index] == 1 and len(stack) == 0:
                    ids["type_attr_ids"].append(
                        index 
                    )  
                elif split_type[i] == "Num" and split_type[index] == "Num" and split_id[index] == 1:
                    ids["type_num_ids"].append(index)
                elif split_type[i] in {"NameLoad", "NameStore"}  and split_type[index] in {"NameLoad", "NameStore"} and split_id[index] == 1:
                    ids["type_name_ids"].append(index)
                elif split_type[i] == "NameParam" and split_type[index] == "NameParam" and split_id[index] == 1:
                    ids["type_param_ids"].append(index)
                # RQ3/RQ4 additional metrics
                elif split_type[i] == "Str"  and split_type[index] == "Str" and split_id[index] == 1:
                    ids["type_string_ids"].append(index)  
                break      
    return ids


""" def get_value_ids(split_type, split_id):
    ids = {"attr_type_ids": [], "num_type_ids": [], "name_type_ids": [], "param_type_ids": [], "string_type_ids": []}
    stack = []
    for i, flag in enumerate(split_id):
        if flag == 0:
            stack.append(i)
        elif len(stack)!=0 and split_type[i] == split_type[stack.pop()]:
            value_type = split_type[stack.pop()]

                  
    return ids """


def get_type_ids(split_type, split_id):
    ids = {
        "call_ids": [],
        "assign_ids": [],
        "return_ids": [],
        "list_ids": [],
        "dict_ids": [],
        "raise_ids": [],
        ## New IDs from RQ3
        "attribute_ids": [],
        "cond_ids": [],
        "comp_ids": [],
        "tuple_ids": []
    }
    for i, flag in enumerate(split_id):
        if flag == 1:
            type_ = split_type[i]
            if type_ == "Call":
                ids["call_ids"].append(i)
            elif type_ == "Assign":
                ids["assign_ids"].append(i)
            elif type_ == "Return":
                ids["return_ids"].append(i)
            elif type_ in {"ListComp", "ListLoad", "ListStore"}:
                ids["list_ids"].append(i)
            elif type_ in {"DictComp", "DictLoad", "DictStore"}:
                ids["dict_ids"].append(i)
            elif type_ == "Raise":
                ids["raise_ids"].append(i)
            # RQ3 additional metrics
            elif type_ in {"AttributeLoad", "AttributeStore"}:
                ids["attribute_ids"].append(i)
            elif type_ in {"If", "orelse"}:
                ids["cond_ids"].append(i)
            elif type_ in {"CompareEq", "CompareIn", "CompareIs"}:
                ids["comp_ids"].append(i)
            elif type_ in {"TupleDel", "TupleLoad", "TupleStore"}:
                ids["tuple_ids"].append(i)
            
    return ids



""" def external(file_path, suffix, n_ctx):
    outfile = "output/{}_ids.txt".format(suffix)

    if os.path.exists(outfile):
        os.remove(outfile)
    logging.info("Type of id to get: {}".format("all"))

    logging.info("Loading dps from: {}".format(file_path))
    with open(file_path, "r") as f, open(outfile, "w") as fout:
        for line in file_tqdm(f):
            ast = json.loads(line.strip())
            type, index = postorder_type_and_index(ast)
            #1.按照generate_data相同的方式对type数组和index数组进行分割
            ids = split_type_and_index_sequence(type,index,)
            #2.对分割的每一部分，参考gennerate_ids的结果生成ids

                #2.1获取每一部分的leaf_ids和internal_id；0表示leaf，1表示internal

                #2.2获取要计算的value_ids；使用type和index数组，通过index数组找leaf，通过type数组将index写入ids对应type

                #2.3获取要计算的type_ids；和2.2方法类似

                #保存ids
            for ast, _ in asts:
                ids = {}
                if len(ast) > 1:
                    if "all" in {"leaf", "all"}:
                        ids.update(get_leaf_ids(ast))
                    if "all" in {"value", "all"}:
                        ids.update(get_value_ids(ast))
                    if "all" in {"type", "all"}:
                        ids.update(get_type_ids(ast))

                    json.dump(ids, fp=fout) 
                    fout.write("\n")
    logging.info("Wrote to: {}".format(outfile)) """

def main():
    parser = argparse.ArgumentParser(
        description="Generate ids (leaf, values, types) from AST"
    )
    parser.add_argument(
        "--ast_fp", "-a", help="Filepath with the new ASTs to be parsed"
    )
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/ids.txt", help="Filepath for the output ids"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    parser.add_argument(
        "id_type",
        choices=["leaf", "value", "type", "all"],
        default="all",
        help="Which ids to generate. Default = leaf",
    )

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Type of id to get: {}".format(args.id_type))

    logging.info("Loading dps from: {}".format(args.ast_fp))
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            ast = json.loads(line.strip())
            type, index = postorder_type_and_index(ast)
            #1.按照generate_data相同的方式对type数组和index数组进行分割
            ids = split_type_and_index_sequence(type,index,args.n_ctx)
            #2.对分割的每一部分，参考gennerate_ids的结果生成ids 
            for split_type, split_id in ids:
                ids = {}
                if(len(split_type)>1):
                    #2.1获取每一部分的leaf_ids和internal_id；0表示leaf，1表示internal
                    if args.id_type in {"leaf", "all"}:
                        ids.update(get_leaf_ids(split_id))
                    #2.2获取要计算的value_ids；使用type和index数组，通过index数组找leaf，通过type数组将index写入ids对应type
                    if args.id_type in {"value", "all"}:
                        ids.update(get_value_ids(split_type,split_id))
                    #2.3获取要计算的type_ids；和2.2方法类似
                    if args.id_type in {"type", "all"}:
                        ids.update(get_type_ids(split_type,split_id))
                    #2.4获取要计算的valueType_ids(即value节点对应的type节点的id) 
                    ids.update(get_valueType_ids(split_type,split_id))
                    #保存ids
                    json.dump(ids, fp=fout) 
                    fout.write("\n")
    logging.info("Wrote to: {}".format(args.out_fp))


if __name__ == "__main__":
    main()  
