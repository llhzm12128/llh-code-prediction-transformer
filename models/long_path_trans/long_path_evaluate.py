import argparse
import model
import torch
import pickle
import os
from tqdm import tqdm
import numpy as np
from models.long_path_trans import dataset
import pickle
import json

def generate_test(model, context, device, depth=2, top_k=10):
    model.eval()
    with torch.no_grad():
        context = torch.tensor(context).to(device)
        output = model(context, None)[-1]
        top_k_values, top_k_indices = torch.topk(output, top_k)
        return top_k_values, top_k_indices

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT2 Model")
    parser.add_argument("--model", default="output/pathTransDemo-model-final.pt", help="Specify the model file")
    parser.add_argument("--dps", default="tmp/long_path_trans/50k_eval.txt", help="Specify the data file (dps) on which the model should be tested on")
    parser.add_argument("--ids", default="tmp/ids_100k_train.txt", help="Specify the data file (ids) on which the model should be tested on")
    parser.add_argument("--output", default="output/long_path_trans")
    args = parser.parse_args()

    eval(args.model, args.dps, args.ids, args.output)

#返回某个类型叶子节点的平均MRR
def mean_reciprocal_rank(labels, predictions, unk_idx):
    scores = []
    for i, l in enumerate(labels):
        if l == unk_idx:
            scores.append(0)
            continue
        score = 0
        for j, p in enumerate(predictions[i]):
            if l == p:
                score = 1 / (j + 1)
                break
        scores.append(score)
    if len(scores) > 0:
        return sum(scores) / len(scores)
    else:
        return 0

def eval(model_fp, dps, ids, output_fp, embedding_size = 300, n_layers = 6):
    
    setup = dataset.Setup(output_fp, dps, ids, mode="eval")
    ds = setup.dataset
    vocab = setup.vocab
    unk_idx = vocab.unk_idx
    m = model.from_file(model_fp, len(vocab), vocab.pad_idx, embedding_size, n_layers)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size = 1,
        collate_fn = lambda b: ds.collate(b, setup.vocab.pad_idx)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    m = m.to(device)
    m.eval()

    print("Evaluating {} batches".format(len(dataloader)))

    # Values (Predict type + value)
    # Contains one value per batch 
    """ value_scores = {
        "attr_ids": {"v_scores": [], "t_scores": []},
        "num_ids": {"v_scores": [], "t_scores": []},
        "name_ids": {"v_scores": [], "t_scores": []},
        "param_ids": {"v_scores": [], "t_scores": []},
        "string_ids": {"v_scores": [], "t_scores": []}
    } """
    
    value_scores = {
        "attr":[],
        "num":[],
        "name":[],
        "param":[],
        "str":[],
    }

    # Types (Predict type only)
    """ type_scores = {
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
    } """

    for i, batch in tqdm(enumerate(dataloader)):
        if True:
            with torch.no_grad():
                x = batch["input_seq"][0]
                y = batch["target_seq"][0]
                paths = batch["root_paths"]
                leaf_type = batch["leaf_type"]
                
                paths = paths.to(device)
                x = x.to(device)
                output = m(x=x,y=None, paths=paths)
                
                ### Evaluate value scores,  path_trans only value ###
                
                for key in value_scores:
                    value_ids = [i for i, type in enumerate(leaf_type) if type == key]
                    if len(value_ids) > 0:
                        value_predictions = [torch.topk(o, 10)[1].tolist() for o in output[value_ids]]
                        value_scores[key].append(mean_reciprocal_rank(y[value_ids], value_predictions, unk_idx))
                if i % 100 == 0:
                    print("Batch {}, It. {}/{}".format(i, i, ds.__len__() / 1))            
                    

    for k in value_scores():
        print("{}".format(k))
        if len(value_scores[k]) > 0:
            print("\tType Prediction: {}".format(sum(value_scores[k])/len(value_scores[k])))
        else:
            print("\tType Prediction: None")
    save_file = os.path.join(output_fp, "value_scores.json")
    if(os.path.exists(save_file)):
        os.remove(save_file)
    with open(save_file, "w") as file:
        json.dump(value_scores, file)
    return value_scores

                    
"""                 for key in value_scores:
                    # print("{}".format(key))
                    value_ids = [a for a in batch["ids"][key] if a < len(output)]    #叶子节点在AST中的位置（每个AST都是从零开始计算）
                    type_ids = [a - 1 for a in batch["ids"][key] if a > 0 and a < len(output)]#叶子的父类型节点在AST中的位置（每个AST都是从零开始计算） 
                    
                    # print("{}: {}".format(i, value_ids))

                    # print(" ".join([vocab.idx2vocab[v] for v in y[value_ids]]))
                    # print(" ".join([vocab.idx2vocab[v] for v in y[type_ids]]))

                    # value scoring
                    if len(value_ids) > 0:
                        value_predictions = [torch.topk(o, 10)[1].tolist() for o in output[value_ids]]
                        value_scores[key]["v_scores"].append(mean_reciprocal_rank(y[value_ids], value_predictions, unk_idx))
                        

                    # type scoring
                    if len(type_ids) > 0:
                        type_predictions = [torch.topk(o, 10)[1].tolist() for o in output[type_ids]]
                        value_scores[key]["t_scores"].append(mean_reciprocal_rank(y[type_ids], type_predictions, unk_idx))

                for key in type_scores:
                    type_ids = [a - 1 for a in batch["ids"][key] if a > 0]

                    if len(type_ids) > 0:
                        type_predictions = [torch.topk(o, 10)[1].tolist() for o in output[type_ids]]
                        type_scores[key].append(mean_reciprocal_rank(y[type_ids], type_predictions, unk_idx))
        
    for k, s in value_scores.items():
        print("{}".format(k))
        if len(value_scores[k]["t_scores"]) > 0:
            print("\tType Prediction: {}".format(sum(value_scores[k]["t_scores"])/len(value_scores[k]["t_scores"])))
        else:
            print("\tType Prediction: None")
        if len(value_scores[k]["v_scores"]) > 0:
            print("\tValue Prediction: {}".format(sum(value_scores[k]["v_scores"])/len(value_scores[k]["v_scores"])))
        else:
            print("\tValuePrediction: None")

    for k, s in type_scores.items():
        print("{}".format(k))
        if len(type_scores[k]) > 0:
            print("\tType Prediction: {}".format(sum(type_scores[k])/len(type_scores[k])))
        else:
            print("\tType Prediction: None")

    return {"value_scores": value_scores, "type_scores": type_scores}
 """
if __name__ == "__main__":
    main()

# Evaluation: 

# - Create DataLoader for test files
# - Iterate through all batches and calculate predictions
# - Collect predictions in list
# - depending on leaf ids, request MRR