import argparse
import model as md
import torch
import pickle
import os
from tqdm import tqdm
import numpy as np
from models.trav_trans import dataset

def generate_test(model, context, device, depth=2, top_k=10):
    model.eval()
    with torch.no_grad():
        context = torch.tensor(context).to(device)
        output = model(context, None)[-1]
        top_k_values, top_k_indices = torch.topk(output, top_k)
        return top_k_values, top_k_indices

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT2 Model")
    parser.add_argument("--model", default="output/model-8.pkl", help="Specify the model file")
    parser.add_argument("--dps", default="output/test_dps.txt", help="Specify the data file (dps) on which the model should be tested on")
    parser.add_argument("--ids", default="output/test_ids.txt", help="Specify the data file (ids) on which the model should be tested on")
    parser.add_argument("--batch_size", default=1, type=int, help="Specify the batch size")

    args = parser.parse_args()

    eval(args.model, args.dps, args.ids)

def mean_reciprocal_rank(y_labels, y_pred):
    scores = []
    for i, e in enumerate(y_labels):
        score = 0
        for j, f in enumerate(y_pred[i]):
            if e == f:
                score = 1 / (j + 1)
                break
        scores.append(score)

    return scores

def eval(model_fp, dps, ids, batch_size = 1, epoch = 0):
    
    setup = dataset.Setup("output", dps, ids, mode="eval")

    m = md.from_file(model_fp, setup.vocab)

    dataloader = torch.utils.data.DataLoader(
        setup.dataset,
        batch_size = batch_size,
        collate_fn = lambda b: dataset.Dataset.collate(b, setup.vocab.pad_idx)
    )
    vocab = setup.vocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    m.eval()

    mrrs = {
        "total": [],
        "attribute_access": [],
        "numeric_constant": [],
        "variable_name": [],
        "function_prameter_name": []
    }
    c = {}

    print("Evaluating {} batches".format(len(dataloader)))
    for i, batch in tqdm(enumerate(dataloader)):
        
        with torch.no_grad():
            x = batch["input_seq"][0]
            y = batch["target_seq"][0]
            leaf_ids = [b for b in batch["ids"]["leaf_ids"] if b >= 0]
            prev_leaf_ids = [l - 1 for l in leaf_ids if l > 0]
            x = x.to(device)
            output = m(x, None)

            y_type_pred = torch.topk(output[prev_leaf_ids], 10)[1].cpu().numpy() # Top 10 predictions for type
            y_type_labels = y[prev_leaf_ids].cpu().numpy()

            y_value_pred = torch.topk(output[leaf_ids], 10)[1].cpu().numpy() # Top 10 predictions for value
            y_value_labels = y[leaf_ids].cpu().numpy()

            type_scores = []
            value_scores = []

            attribute_access_scores = []
            numeric_constant_scores = []
            variable_name_scores = []
            # function_parameter_name_scores = []

            type_scores = mean_reciprocal_rank(y_type_labels, y_type_pred)
            value_scores = mean_reciprocal_rank(y_value_labels, y_value_pred)

            for j, t in enumerate(y_type_labels):
                if vocab.idx2vocab[t] == "attr":
                    attribute_access_scores.append([type_scores[j], value_scores[j]])
                elif vocab.idx2vocab[t] == "Num":
                    numeric_constant_scores.append([type_scores[j], value_scores[j]])
                elif vocab.idx2vocab[t] == "NameLoad":
                    variable_name_scores.append([type_scores[j], value_scores[j]])
            
            # Add entries to mrrs

            if len(value_scores) > 0 and len(type_scores) > 0:
                mrrs["total"].append({
                    "type": sum(type_scores) / len(type_scores),
                    "value": sum(value_scores) / len(value_scores)
                })
            if len(attribute_access_scores) == 2:
                mrrs["attribute_access"].append({
                    "type": sum(attribute_access_scores[0]) / len(attribute_access_scores[0]),
                    "value": sum(attribute_access_scores[1]) / len(attribute_access_scores[1])
                })
            if len(numeric_constant_scores) == 2:
                mrrs["numeric_constant"].append({
                    "type": sum(numeric_constant_scores[0]) / len(numeric_constant_scores[0]),
                    "value": sum(numeric_constant_scores[1]) / len(numeric_constant_scores[1])
                })
            if len(variable_name_scores) == 2:
                mrrs["variable_name"].append({
                    "type": sum(variable_name_scores[0]) / len(variable_name_scores[0]),
                    "value": sum(variable_name_scores[1]) / len(variable_name_scores[1])
                })

    # with open("output/c.pkl", "wb") as fout:
    #     pickle.dump(c, fout)

    total_mrr = {
        "type": sum([a["type"] for a in mrrs["total"]]) / len([a["type"] for a in mrrs["total"]]),
        "value": sum([a["value"] for a in mrrs["total"]]) / len([a["value"] for a in mrrs["total"]])
    }

    attribute_access_mrr = {
        "type": sum([a["type"] for a in mrrs["attribute_access"]]) / len([a["type"] for a in mrrs["attribute_access"]]),
        "value": sum([a["value"] for a in mrrs["attribute_access"]]) / len([a["value"] for a in mrrs["attribute_access"]])
    }

    numeric_constant_mrr = {
        "type": sum([a["type"] for a in mrrs["numeric_constant"]]) / len([a["type"] for a in mrrs["numeric_constant"]]),
        "value": sum([a["value"] for a in mrrs["numeric_constant"]]) / len([a["value"] for a in mrrs["numeric_constant"]])
    }

    variable_name_mrr = {
        "type": sum([a["type"] for a in mrrs["variable_name"]]) / len([a["type"] for a in mrrs["variable_name"]]),
        "value": sum([a["value"] for a in mrrs["variable_name"]]) / len([a["value"] for a in mrrs["variable_name"]])
    }

    print("Eval epoch {}".format(epoch))
    print("Total: {}/{}".format(total_mrr["type"], total_mrr["value"]))
    print("Attribute Access: {}/{}".format(attribute_access_mrr["type"], attribute_access_mrr["value"]))
    print("Numeric Constant: {}/{}".format(numeric_constant_mrr["type"], numeric_constant_mrr["value"]))
    print("Variable Name: {}/{}".format(variable_name_mrr["type"], variable_name_mrr["value"]))

    mrr_dict = {
        "epoch": epoch,
        "total": total_mrr,
        "attribute_access": attribute_access_mrr,
        "numeric_constant": numeric_constant_mrr,
        "variable_name": variable_name_mrr
    }

    return mrr_dict

if __name__ == "__main__":
    main()

# Evaluation: 

# - Create DataLoader for test files
# - Iterate through all batches and calculate predictions
# - Collect predictions in list
# - depending on leaf ids, request MRR