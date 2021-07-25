import argparse
import model
import torch
from tqdm import tqdm
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
    parser.add_argument("--model", help="Specify the model file")
    parser.add_argument("--dps", help="Specify the data file (dps) on which the model should be tested on")
    parser.add_argument("--ids", help="Specify the data file (ids) on which the model should be tested on")
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

    eval(m, dataloader)

def eval(model, dataloader):
    print("Evaluating {} batches".format(len(dataloader)))
    reciprocal_rank = {
        "all_leaf_tokens": [],
        "attribute_access": [],
        "numeric_constant": [],
        "variable_name": [],
        "function_parameter_name": []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for i, batch in tqdm(enumerate(dataloader)):
        if i % 100 == 0:
            print("Batch {}".format(i))
        x = batch["input_seq"][0]
        y = batch["target_seq"][0]
        ids = batch["ids"]["leaf_ids"]

        for id in ids:
            if id > 0:
                y_type = x[id].item()
                y_value = y[id].item()

                with torch.no_grad():
                    y_type_pred = generate_test(model, [i.item() for i in x[range(id)]], device)
                    y_value_pred = generate_test(model, [i.item() for i in x[range(id + 1)]], device)

                    type_rank = 0
                    value_rank = 0

                    if y_type in y_type_pred[1]:
                        type_rank = 1 / ((y_type_pred[1] == y_type).nonzero(as_tuple=True)[0].item() + 1)
                    if y_value in y_value_pred[1]:
                        value_rank = 1 / ((y_value_pred[1] == y_value).nonzero(as_tuple=True)[0].item() + 1)
                    reciprocal_rank["all_leaf_tokens"].append((type_rank + value_rank) / 2)

if __name__ == "__main__":
    main()