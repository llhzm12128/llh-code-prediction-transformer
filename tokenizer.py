from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit

from utils import file_tqdm, separate_dps

import json
import logging
from tqdm import tqdm

# Clean up raw py150 dataset by removing "," character

if False:
    with open("data/python150k.json") as fin, open("data/python150k_rq4.json", "w") as fout:
        for i, line in enumerate(tqdm(fin)):
            dp = json.loads(line.strip())
            for j, d in enumerate(dp):
                if "value" in d:
                    if "," in d["value"]:
                        d["value"].replace(",", " ")
            print(json.dumps(dp), file=fout)
    
    with open("data/python150k_rq4.json") as fin:
        for line in tqdm(fin):
            dp = json.loads(line.strip())
            for d in enumerate(dp):
                if "value" in d:
                    if "," in d["value"]:
                        print('Not cleaned up')

# Extract value/types from trees and store in comma separated raw file (all_raw.json)

if False:
    with open("output/all_new_trees.json") as fin, open("output/all_raw.json", "w") as fout:
        for i, line in enumerate(tqdm(fin)):
            dp = json.loads(line)
            token_list = []
            for d in dp:
                if "value" in d:
                    token_list.append(d["value"])
                elif "type" in d:
                    token_list.append(d["type"])
            raw = ",".join(token_list)
            print(json.dumps(raw), file=fout)

# Train tokenizer on raw file

#tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = CharDelimiterSplit(delimiter=",")
# trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]"])

# tokenizer.train(["output/all_raw.json"], trainer)

# tokenizer.save("output/tokenizer.json")

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

    # print("d len: {}".format(len(d)))
    
    half_len = int(max_len / 2)
    if len(d) <= max_len:
        return [[d, 0]]

    aug_d = [[d[:max_len], 0]]
    # print("first: {}\n\n".format(aug_d[-1]))
    i = half_len
    while i < len(d) - max_len:
        aug_d.append([d[i : i + max_len], half_len])
        # print("loop: {}\n\n".format(aug_d[-1]))
        i += half_len
    idx = max_len - (len(d) - (i + half_len))
    aug_d.append([d[-max_len:], idx])
    # print("last: {}\n\n".format(aug_d[-1]))
    return aug_d

tokenizer = Tokenizer.from_file("output/tokenizer.json")
outfile = "output/train_rq4_dps.txt"
num_dps = 0
with open("output/train_new_trees.json") as fin, open(outfile, "w") as fout:
    for line in file_tqdm(fin):
        dp = json.loads(line.strip())
        asts = split(dp, 1000, tokenizer)
        for ast, extended in asts:
            if len(ast) > 1:
                json.dump([ast, extended], fp=fout)
                fout.write("\n")
                num_dps += 1
logging.info("Wrote {} datapoints to {}".format(num_dps, outfile))