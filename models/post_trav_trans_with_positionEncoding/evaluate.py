import json
import argparse
import sys
sys.path.append("D:\\projects\\llh-code-prediction-transformer")
import model_with_position
import torch
import pickle
import os
from tqdm import tqdm
import numpy as np
from models.trav_trans_with_positionEncoding.trav_with_position_dataset import Setup

def generate_test(model, context, device, depth=2, top_k=10):
    model.eval()
    with torch.no_grad():
        context = torch.tensor(context).to(device)
        output = model(context, None)[-1]
        top_k_values, top_k_indices = torch.topk(output, top_k)
        return top_k_values, top_k_indices

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT2 Model")
    parser.add_argument("--model", default="output\\trav_trans_with_positionEncoding\\trav_trans_with_positionEncoding-model-10.pt", help="Specify the model file")
    parser.add_argument("--dps", default="tmp\\trav_trans\\dps_eval.txt", help="Specify the data file (dps) on which the model should be tested on")
    parser.add_argument("--ids", default="tmp\\trav_trans\\ids_eval.txt", help="Specify the data file (ids) on which the model should be tested on")
    parser.add_argument("--levels", default="tmp\\trav_trans_with_positionEncoding\\levels_eval.txt")
    parser.add_argument("--output", default="output\\trav_trans_with_positionEncoding") #中间文件保存目录
    parser.add_argument("--save", default="output\\trav_trans_with_positionEncoding\\value_scores.json", help="Record evaluate results")
    args = parser.parse_args()

    eval(args.model, args.dps, args.ids, args.levels, args.output, args.save)

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

def eval(model_fp, dps, ids, levels, output_fp, save_fp, embedding_size = 300, n_layers = 6):
    
    setup = Setup( output_fp, dps, ids, levels, mode="eval") 
    ds = setup.dataset
    vocab = setup.vocab
    unk_idx = vocab.unk_idx
    m = model_with_position.from_file(model_fp, len(vocab), vocab.pad_idx, embedding_size, n_layers)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size = 1,
        collate_fn = lambda b: ds.collate(b, setup.vocab.pad_idx)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('cuda')
    else:
        print('cpu')
    m = m.to(device)
    m.eval()

    print("Evaluating {} batches".format(len(dataloader)))

    # Values (Predict type + value)
    # Contains one value per batch
    value_scores = {
        "attr_ids": {"v_scores": [], "t_scores": []},
        "num_ids": {"v_scores": [], "t_scores": []},
        "name_ids": {"v_scores": [], "t_scores": []},
        "param_ids": {"v_scores": [], "t_scores": []},
        "string_ids": {"v_scores": [], "t_scores": []}
    }

    # Types (Predict type only)
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
    
    all_scores={
        "leaf_ids":[],
        "internal_ids":[]
    }
    avg = {
        "avg":[]
    }

    for i, batch in tqdm(enumerate(dataloader)):
        if True:
            with torch.no_grad():
                x = batch["input_seq"][0]
                y = batch["target_seq"][0]
                levels = batch["level"][0]

                x = x.to(device)
                levels = levels.to(device)
                output = m(x = x, y = None, position_ids= levels)[0]
                
                ### Evaluate value scores, type + value ###

                for key in value_scores:
                    # print("{}".format(key))
                    value_ids = [a for a in batch["ids"][key] if a < len(output)]
                    type_ids = [a - 1 for a in batch["ids"][key] if a > 0 and a < len(output)]
                    
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
                    type_ids = [a for a in batch["ids"][key] if a >= 0]

                    if len(type_ids) > 0:
                        type_predictions = [torch.topk(o, 10)[1].tolist() for o in output[type_ids]]
                        type_scores[key].append(mean_reciprocal_rank(y[type_ids], type_predictions, unk_idx))
                
                for key in all_scores:
                    if key == "leaf_ids":
                        node_ids = [a for a in batch["ids"][key] if a >= 0]
                        if len(node_ids) > 0:       
                            type_ids = [a - 1 for a in batch["ids"][key] if a > 0 and a < len(output)]
                            node_predictions = [torch.topk(o, 10)[1].tolist() for o in output[node_ids]]
                            type_predictions = [torch.topk(o, 10)[1].tolist() for o in output[type_ids]]
                            all_scores[key].append((mean_reciprocal_rank(y[node_ids], node_predictions, unk_idx)+
                                                    mean_reciprocal_rank(y[type_ids], type_predictions, unk_idx))/2)
                    else:
                        node_ids = [a for a in batch["ids"][key] if a >= 0]
                        if len(node_ids) > 0:
                            node_predictions = [torch.topk(o, 10)[1].tolist() for o in output[node_ids]]
                            all_scores[key].append(mean_reciprocal_rank(y[node_ids], node_predictions, unk_idx))
                
                if i % 100 == 0:
                    print("Batch {}, It. {}/{}".format(i, i, ds.__len__() / 1))

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
        print("\tAverage Prediction: {}".format(
            ((sum(value_scores[k]["v_scores"])/len(value_scores[k]["v_scores"]))+
            (sum(value_scores[k]["t_scores"])/len(value_scores[k]["t_scores"])))/2
            ))
    for k, s in type_scores.items():
        print("{}".format(k))
        if len(type_scores[k]) > 0:
            print("\tType Prediction: {}".format(sum(type_scores[k])/len(type_scores[k])))
        else:
            print("\tType Prediction: None")
            
    for k, s in all_scores.items():
        print("{}".format(k))
        if len(all_scores[k]) > 0:
            print("\tType Prediction: {}".format(sum(all_scores[k])/len(all_scores[k])))
        else:
            print("\tType Prediction: None")
            
    acc_sum=0
    ids_count  =0
    for k, s in all_scores.items():
        print("avg:")
        if len(all_scores[k]) > 0:
            acc_sum+=sum(all_scores[k])
            ids_count+=len(all_scores[k])  
        else:
            print("\tType Prediction: None")
    if ids_count > 0:
        avg["avg"] = acc_sum / ids_count
        print("\tAvg Prediction: {}".format(avg["avg"]))
    else:
        avg["avg"] = None
        print("\tAvg Prediction: None")
        
    scores = {"value_scores": value_scores, "type_scores": type_scores,"all_scores":all_scores,"avg":avg}
    if(os.path.exists(save_fp)):
        os.remove(save_fp)
    with open(save_fp, "w") as file:
        
        json.dump(scores, file)

    return scores

if __name__ == "__main__":
    main()

# Evaluation: 

# - Create DataLoader for test files
# - Iterate through all batches and calculate predictions
# - Collect predictions in list
# - depending on leaf ids, request MRR