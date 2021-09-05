import argparse
from utils import file_tqdm
import json
import logging

import generate_new_trees

from tokenizers import Tokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for rq4")
    parser.add_argument("--file_path")
    parser.add_argument("--tokenizer")
    parser.add_argument("--suffix")

    args = parser.parse_args()

    # Generate new trees
    print("Generating new trees")
    generate_new_trees.external(args.file_path, args.suffix)
    # Remove comma character from trees
    print("Removing \",\" character from new trees")
    clean_trees("output/{}_new_trees.json".format(args.suffix), args.suffix)
    # Split trees and traverse DFS
    print("Splitting and encoding trees")
    preprocess("output/{}_new_trees_cleaned.json".format(args.suffix), args.suffix, args.tokenizer)

def split(ast, max_len, tokenizer):
    # list contains leaf ID and encoded tokens    
    d = []

    # Iterate through tree nodes, fill list
    for i, a in enumerate(ast):
        if "type" in a:
            ids = tokenizer.encode(a["type"]).ids
            d.extend(ids)
        elif "value" in a:
            ids = tokenizer.encode(a["value"]).ids
            d.extend(ids)
    
    half_len = int(max_len / 2)
    if len(d) <= max_len:
        return [[d, 0]]

    aug_d = [[d[:max_len], 0]]
    i = half_len
    while i < len(d) - max_len:
        aug_d.append([d[i : i + max_len], half_len])
        i += half_len
    idx = max_len - (len(d) - (i + half_len))
    aug_d.append([d[-max_len:], idx])
    return aug_d

def preprocess(fp, suffix, tokenizer):
    tokenizer = Tokenizer.from_file(tokenizer)
    outfile = "output/{}_dps.txt".format(suffix)
    num_dps = 0
    with open(fp) as fin, open(outfile, "w") as fout:
        for line in file_tqdm(fin):
            dp = json.loads(line.strip())
            asts = split(dp, 1000, tokenizer)
            for ast, extended in asts:
                if len(ast) > 1:
                    json.dump([ast, extended], fp=fout)
                    fout.write("\n")
                    num_dps += 1
    logging.info("Wrote {} datapoints to {}".format(num_dps, outfile))

def clean_trees(fp, suffix):
    with open(fp) as fin, open("output/{}_new_trees_cleaned.json".format(suffix), "w") as fout:
        for i, line in enumerate(tqdm(fin)):
            dp = json.loads(line.strip())
            for j, d in enumerate(dp):
                if "value" in d:
                    if "," in d["value"]:
                        d["value"].replace(",", " ")
            print(json.dumps(dp), file=fout)


if __name__ == "__main__":
    main()