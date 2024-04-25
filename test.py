""" ast = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
max_len = 9

# 1000
# 4320

def separate_dps(ast, max_len):

    asts = []
    i = max_len

    if len(ast) <= max_len:
        return [[ast]]
    
    asts.append([ast[:max_len]])
    
    while i < len(ast) - max_len:
        asts.append([ast[i : i + max_len]])
        i += max_len

    asts.append(ast[i:])

    return asts

print(separate_dps(ast, max_len)) """

""" with open("./tmp/path_dps_50k_eval.txt" ,'r') as f:
    lines = f.readlines()
    num_lines = len(lines) 
print(num_lines) """



'''
import argparse
import os
import pickle
import json

value_scores = {
        "attr":[],
        "num":[],
        "name":[],
        "param":[],
        "str":[],
    }

type_scores = {
        "call_ids": [],
        "assign_ids": [],
        "return_ids": [],
        "list_ids": [],
        "dict_ids": [],
        "raise_ids": [],
        "attribute_ids": [],
        "cond_ids": [],
        "comp_ids": [],
        "tuple_ids": []
    }

scores = {"value_scores": value_scores, "type_scores": type_scores}


parser = argparse.ArgumentParser(description="Evaluate GPT2 Model")
parser.add_argument("--save", default="output/path_trans/value_scores.json", help="Record evaluate results")
args = parser.parse_args()
save_fp = args.save
if(os.path.exists(save_fp)):
    os.remove(save_fp)

with open(save_fp, "w") as file:
    json.dump(scores, file)
    


with open(save_fp, "rb") as file:
    loaded_data = json.load(file)
print(loaded_data)
'''
""" save_fp = "test1.txt"
import json
value_scores = {
        "attr":[],
        "num":[],
        "name":[],
        "param":[],
        "str":[],
    }

with open(save_fp, "w") as file:
    json.dump(value_scores, file)
    file.write("\n") """


""" #读取losses.pickle文件
import pickle

# 指定 pickle 文件路径
pickle_file_path = "output\\path_trans\\losses.pickle"

# 读取 pickle 文件
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

# 打印读取的数据
print(data) """

""" value_scores = {
        "attr":[1,2],
        "num":[2,3],
        "name":[2,2],
        "param":[1,1],
        "str":[2,3]
    }
for k,v in value_scores.items():
    print("{}".format(k))
    if len(value_scores[k]) > 0:
        print("\tValue Prediction: {}".format(sum(value_scores[k])/len(value_scores[k])))
    else:
        print("\tValue Prediction: None") """
    
""" list = [1,2,3]
print(list[1:5]) """

""" #后序遍历
def postorder_traversal(ast):
    dp = []
    def traverse(node):
        if 'children' in node:
            for child_index in node['children']:
                traverse(ast[child_index])
        if 'value' in node:
            dp.append(node['value'])
        if 'type' in node:
            dp.append(node['type'])

    traverse(ast[0])
    return dp """

""" #打印数据集中的一条数据
import json
date_file = "tmp\\new_100k_train.json"
with open(date_file, "r") as f:
    for line in f:
        dp = json.loads(line.strip())
        postorder_traversal(dp) 
       
        break """




""" ast = [{'type': 'Module', 'children': [1, 4, 8, 12, 16]}, 
       {'type': 'Expr', 'children': [2]}, 
       {'type': 'Str', 'children': [3]}, 
       {'value': ' Provides ``mapping`` of url paths to request handlers.\n'}, 
       {'type': 'ImportFrom', 'children': [5, 6]}, 
       {'value': 'bootstrap'}, 
       {'type': 'alias', 'children': [7]},
       {'value': 'Bootstrap'}, 
       {'type': 'ImportFrom', 'children': [9, 10]}, 
       {'value': 'fund'}, 
       {'type': 'alias', 'children': [11]}, 
       {'value': 'InstantPaymentNotificationHandler'}, 
       {'type': 'ImportFrom', 'children': [13, 14]}, 
       {'value': 'fund'}, 
       {'type': 'alias', 'children': [15]}, 
       {'value': 'ThankYouHandler'}, 
       {'type': 'ImportFrom', 'children': [17, 18]}, 
       {'value': 'view'}, 
       {'type': 'alias', 'children': [19]}, 
       {'value': '*'}, 
       ]

dp = postorder_traversal(ast)
print(dp) """

""" #test词汇表
import pickle

# 读取.pkl文件
with open('tmp\\vocab.pkl', 'rb') as file:
    data = pickle.load(file)
    print(len(data))
    print(data[100000])
    print(data[100001])
    print(data[0])
    print(data[1])
    """




""" import sys
import platform
if(platform.system() == "Windows"):
    sys.path.append("C:/Users/llh/Desktop/ISCAS/llh-code-prediction-transformer")
else:
    sys.path.append("/root/llh-code-prediction-transformer")
from utils import file_tqdm

#先序遍历整棵树，然后再分解，并和trav_trans的数据集进行一致性对比
#1.先序遍历AST
def get_dfs(ast, only_leaf=False):
    dp = []
    for node in ast:
        if "value" in node:
            dp.append(node["value"])
        else:
            if not only_leaf:
                dp.append(node["type"])
    return dp

def split_sequence(sequence, max_len):
    half_len = int(max_len / 2)
    if len(sequence) <= max_len:
        return [[sequence, 0]]
    aug_asts = [[sequence[:max_len], 0]]
    i = half_len
    while i < len(sequence) - max_len:
        aug_asts.append([sequence[i : i + max_len], half_len])
        i += half_len
    idx = max_len - (len(sequence) - (i + half_len))
    aug_asts.append([sequence[-max_len:], idx])
    return aug_asts

import json
import logging
#3.和数据集进行一致性对比
ast_file = "tmp\\new_100k_train.json"
output_fp = "tmp\\dps_split_after_traversal.txt"
num_dps = 0
with open(ast_file,"r") as f, open(output_fp,"w") as fout:
    #2.1遍历每个AST
    for line in file_tqdm(f):
    
        ast = json.loads(line.strip())
        #2.2获取AST的先序序列
        sequence = get_dfs(ast)
        #分解先序序列
        dps = split_sequence(sequence, 1000)
        for dp, extended in dps:
            if(len(dp)>1):
                json.dump([dp,extended],fp=fout)
                fout.write("\n")
                num_dps += 1
logging.info("Wrote {} datapoints to {}".format(num_dps, output_fp))


#和trav_trans的数据集进行一致性对比
trav_trans_dps = "tmp\\trav_trans\\dps_train.txt"  #trav_trans的数据集
output_fp = "tmp\\dps_split_after_traversal.txt" #先遍历再分割的数据集
count1 = 0#trav_trans_dps的行数
count2 = 0#先遍历在分割的数据集的行数
with open(trav_trans_dps,"r") as f1:
    for line in f1:
        count1 += 1
with open(output_fp,"r") as f2:
    for line in f2:
        count2 += 1
print(count1)
print(count2)
assert(count1 == count2) """


""" #输出第一个原AST
with open("data\\python100k_train.json","r") as f1:
    for line in f1:
        print(line)
        break;
#判断原AST是否有单独的value节点 """



""" #后序遍历AST，获取value节点的type，同时记录后序序列中value节点和内部节点的index
post_index = [] #记录后序序列中value节点和内部节点的index
def postorder_traversal(ast):
    dp = []
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
    traverse(ast[0],0)
    return dp

ast = [{'type': 'Module', 'children': [1, 4, 8, 12, 16]}, 
       {'type': 'Expr', 'children': [2]}, 
       {'type': 'Str', 'children': [3]}, 
       {'value': ' Provides ``mapping`` of url paths to request handlers.\n'}, 
       {'type': 'ImportFrom', 'children': [5, 6]}, 
       {'value': 'bootstrap'}, 
       {'type': 'alias', 'children': [7]},
       {'value': 'Bootstrap'}, 
       {'type': 'ImportFrom', 'children': [9, 10]}, 
       {'value': 'fund'}, 
       {'type': 'alias', 'children': [11]}, 
       {'value': 'InstantPaymentNotificationHandler'}, 
       {'type': 'ImportFrom', 'children': [13, 14]}, 
       {'value': 'fund'}, 
       {'type': 'alias', 'children': [15]}, 
       {'value': 'ThankYouHandler'}, 
       {'type': 'ImportFrom', 'children': [17, 18]}, 
       {'value': 'view'}, 
       {'type': 'alias', 'children': [19]}, 
       {'value': '*'}, 
       ]

dp = postorder_traversal(ast)
print(dp)
print(post_index) """


#测试post_trav_trans的dps和ids文件是否相同
with open("tmp\\post_trav_trans\dps_train.txt", "r") as fdp, open("tmp\\post_trav_trans\ids_train.txt", "r") as fid:
    lines_dp = fdp.readlines()
    lines_id = fid.readlines()
    print(len(lines_dp))
    print(len(lines_id))

