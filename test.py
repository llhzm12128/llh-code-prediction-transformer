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

parser = argparse.ArgumentParser(description="Evaluate GPT2 Model")
parser.add_argument("--save", default="output/path_trans/value_scores.json", help="Record evaluate results")
args = parser.parse_args()
save_fp = args.save
if(os.path.exists(save_fp)):
    os.remove(save_fp)
with open(save_fp, "w") as file:
    json.dump(value_scores, file)

with open(save_fp, "rb") as file:
    loaded_data = json.load(file)

# 使用加载的数据进行操作
print(loaded_data)


