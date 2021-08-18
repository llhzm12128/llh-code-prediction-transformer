import argparse
import model
import torch
import pickle
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
    parser.add_argument("--vocab", help="Specify the vocab file")
    parser.add_argument("--batch_size", default=1, type=int, help="Specify the batch size")

    args = parser.parse_args()

    setup = dataset.Setup("output", args.dps, args.ids, mode="test")

    m = model.from_file("output/model-8.pt", setup.vocab)

    dataloader = torch.utils.data.DataLoader(
        setup.dataset,
        batch_size = args.batch_size,
        collate_fn = lambda b: dataset.Dataset.collate(b, setup.vocab.pad_idx)
    )
    vocab = setup.vocab

    eval(m, dataloader, vocab)

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

def eval(model, dataloader, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

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
        if True:
            with torch.no_grad():
                x = batch["input_seq"][0]
                y = batch["target_seq"][0]
                leaf_ids = [b for b in batch["ids"]["leaf_ids"] if b >= 0]
                prev_leaf_ids = [l - 1 for l in leaf_ids if l > 0]
                x = x.to(device)
                output = model(x, None)

                y_type_pred = torch.topk(output[prev_leaf_ids], 10)[1].cpu().numpy() # Top 10 predictions for each leaf
                y_type_labels = y[prev_leaf_ids].cpu().numpy()

                y_value_pred = torch.topk(output[leaf_ids], 10)[1].cpu().numpy()
                y_value_labels = y[leaf_ids].cpu().numpy()

                
                type_scores = []
                value_scores = []

                attribute_access_scores = []
                numeric_constant_scores = []
                variable_name_scores = []
                function_parameter_name_scores = []

                type_scores = mean_reciprocal_rank(y_type_labels, y_type_pred)
                value_scores = mean_reciprocal_rank(y_value_labels, y_value_pred)


                for j, t in enumerate(y_type_labels):
                    if vocab.idx2vocab[t] not in c:
                        c[vocab.idx2vocab[t]] = 1
                    else:
                        c[vocab.idx2vocab[t]] += 1
                        
                    if vocab.idx2vocab[t] == "attr":
                        attribute_access_scores.append((value_scores[j] + type_scores[j]) / 2)
                    elif vocab.idx2vocab[t] == "Num":
                        numeric_constant_scores.append((value_scores[j] + type_scores[j]) / 2)
                    elif vocab.idx2vocab[t] == "NameLoad":
                        variable_name_scores.append((value_scores[j] + type_scores[j]) / 2)
                
                if len(value_scores) > 0 and len(type_scores) > 0:
                    mrrs["total"].append(((sum(type_scores)/len(type_scores)) + (sum(value_scores)/len(value_scores))) / 2)
                if len(attribute_access_scores) > 0:
                    mrrs["attribute_access"].append(sum(attribute_access_scores) / len(attribute_access_scores))
                if len(numeric_constant_scores) > 0:
                    mrrs["numeric_constant"].append(sum(numeric_constant_scores) / len(numeric_constant_scores))
                if len(variable_name_scores) > 0:
                    mrrs["variable_name"].append(sum(variable_name_scores) / len(variable_name_scores))
        else:
            break
    with open("output/c.pkl", "wb") as fout:
        pickle.dump(c, fout)

    with open("output/mrrs.pkl", "wb") as fout:
        pickle.dump(mrrs, fout)

    # with open("output/value_scores.pkl", "wb") as fout:
    #     pickle.dump(value_scores, fout)


    # for i, batch in tqdm(enumerate(dataloader)):
    #     if i % 100 == 0:
    #         print("Batch {}".format(i))
    #     x = batch["input_seq"][0]
    #     y = batch["target_seq"][0]
    #     ids = batch["ids"]["leaf_ids"]

    #     for id in ids:
    #         if id > 0:
    #             y_type = x[id].item()
    #             y_value = y[id].item()

    #             with torch.no_grad():
    #                 y_type_pred = generate_test(model, [i.item() for i in x[range(id)]], device)
    #                 y_value_pred = generate_test(model, [i.item() for i in x[range(id + 1)]], device)

    #                 type_rank = 0
    #                 value_rank = 0

    #                 if y_type in y_type_pred[1]:
    #                     type_rank = 1 / ((y_type_pred[1] == y_type).nonzero(as_tuple=True)[0].item() + 1)
    #                 if y_value in y_value_pred[1]:
    #                     value_rank = 1 / ((y_value_pred[1] == y_value).nonzero(as_tuple=True)[0].item() + 1)
    #                 reciprocal_rank["all_leaf_tokens"].append((type_rank + value_rank) / 2)

if __name__ == "__main__":
    main()

# Evaluation: 

# - Create DataLoader for test files
# - Iterate through all batches and calculate predictions
# - Collect predictions in list
# - depending on leaf ids, request MRR