import argparse
import json
import logging
import os
import numpy as np
import sys
import platform
if(platform.system() == "Windows"):
    sys.path.append("D:\\projects\\llh-code-prediction-transformer")
else:
    sys.path.append("/root/llh-code-prediction-transformer")
from utils import file_tqdm, separate_dps, split_sequence


logging.basicConfig(level=logging.INFO)

#后序遍历的node层级
def get_postTrans_node_level(ast):
    levels = []
    
    def traverse(node, level=1):
        if 'children' in node:
            for child_index in node['children']:
                traverse(ast[child_index], level + 1)  # 递归时层级加1
        if 'value' in node:
            levels.append(level)  # 将节点值和层级一起添加到结果中
        if 'type' in node:
            levels.append(level)   # 将节点类型和层级一起添加到结果中

    traverse(ast[0])  # 从根节点开始遍历
    return levels


def main():
    parser = argparse.ArgumentParser(
        description="Generate ids (leaf, values, types) from AST"
    )
    parser.add_argument(
        "--ast_fp", "-a", help="Filepath with the new ASTs to be parsed"
    )
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/levels.txt", help="Filepath for the output ids"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)

    logging.info("Loading dps from: {}".format(args.ast_fp))
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            #获取整个AST的node层数
            levels = get_postTrans_node_level(dp)
            #根据参数（args.n_ctx）对levels进行分割
            levels = split_sequence(levels, args.n_ctx)
            #处理分割后的levels数组，使每个数组中的level从1开始计算,并保存
            for level, _ in levels:
                if len(level) > 1:
                    min_value = min(level)
                    result_level = [x - (min_value - 1) for x in level]
                    min_value = min(result_level)
                    assert(min_value == 1)
                    json.dump(result_level, fp=fout) 
                    fout.write("\n")
        f.close
        fout.close
    logging.info("Wrote to: {}".format(args.out_fp))


if __name__ == "__main__":
    main()