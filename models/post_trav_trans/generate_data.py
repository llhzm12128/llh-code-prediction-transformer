 
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
import platform
if(platform.system() == "Windows"):
    sys.path.append("D:\\projects\\llh-code-prediction-transformer")
else:
    sys.path.append("/root/llh-code-prediction-transformer")

from utils import file_tqdm, postorder_traversal, split_sequence


logging.basicConfig(level=logging.INFO)

""" def external(file_path, suffix, context_size):
    outfile = "output/{}_dps.txt".format(suffix)
    if os.path.exists(outfile):
        os.remove(outfile)
    logging.info("Number of context: {}".format(context_size))

    num_dps = 0
    logging.info("Loading asts from: {}".format(file_path))
    with open(file_path, "r") as f, open(outfile, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            asts = separate_dps(dp, context_size)
            for ast, extended in asts:
                if len(ast) > 1:
                    dp = postorder_traversal(ast)
                    json.dump([dp, extended], fp=fout)
                    fout.write("\n")
                    num_dps += 1

    logging.info("Wrote {} datapoints to {}".format(num_dps, outfile)) """

def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath for the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Number of context: {}".format(args.n_ctx))

    num_dps = 0
    logging.info("Loading asts from: {}".format(args.ast_fp))
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            ast = json.loads(line.strip())
            #2.2获取AST的后序序列
            post_order_sequence = postorder_traversal(ast)
            #分解后序序列
            dps = split_sequence(post_order_sequence, args.n_ctx)
            for dp, extended in dps:
                if(len(dp)>1):
                    json.dump([dp,extended],fp=fout)
                    fout.write("\n")
                    num_dps += 1

    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
