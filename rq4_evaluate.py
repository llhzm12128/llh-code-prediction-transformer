import torch
import argparse
import model

from rq4_dataset import Dataset
from tokenizers import Tokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Evaluate new tokenizer GPT-2 model")
    parser.add_argument("--model", default="rq4/rq4_model.pt", help="Specify model file")
    parser.add_argument("--dps", default="output/rq4_test_dps.txt", help="Specify data file (dps) on which to evaluate")
    parser.add_argument("--ids", default="output/rq4_test_ids.txt", help="Specify data file (ids) on which to evaluate")
    parser.add_argument("--tokenizer", default="output/tokenizer.json", help="Specify tokenizer file")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    eval(args.model, args.dps, args.ids, tokenizer)

def mean_reciprocal_rank(labels, predictions):
    scores = []
    for i, l in enumerate(labels):
        score = 0
        for j, p in enumerate(predictions[i]):
            if l == p:
                score = 1 / (j + 1)
                break
        scores.append(score)
    return sum(scores) / len(scores)

def eval(model_fp, dps, ids, tokenizer):
    dataset = Dataset(dps, ids)
    pad_idx = tokenizer.encode("[PAD]").ids[0]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda b: dataset.collate(b, pad_idx)
    )
    m = model.from_file(model_fp, tokenizer.get_vocab_size(), pad_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    m.eval()
    print("Evaluating {} batches".format(len(dataloader)))
    string_accuracy = []
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = batch["input_seq"][0]
            y = batch["target_seq"][0]
            # Subtract 1 to get prediction for string ids
            string_ids = [a - 1 for a in batch["ids"]["string_ids"] if a > 0]
            # print(string_ids)

            x = x.to(device)
            output = m(x, None)
            
            iter_string_mrrs = []

            for id in string_ids:
                # Counter for max string length to prevent inf loop, until 20
                breaker = 20
                counter = 0
                string_mrrs = []
                # print("ID: {}".format(id))
                while counter < breaker:
                    # print("\tCounter: {}".format(counter))
                    if counter == 0:
                        label = y[id + counter]
                        top_10 = torch.topk(output[id + counter], 10)[1]
                        string_mrrs.append(mean_reciprocal_rank([label.item()], [top_10.tolist()]))
                        # print("\tLabel: {}\n\tTop10: {}".format(label, ", ".join([str(t.item()) for t in top_10])))
                    else:
                        if id + counter < len(y):
                            label = y[id + counter]
                            top_10 = torch.topk(output[id + counter], 10)[1]
                            if not tokenizer.decode([top_10[0]]).startswith("#"):
                                # print("\tNext token no subword continuation, breaking")
                                break
                            string_mrrs.append(mean_reciprocal_rank([label.item()], [top_10.tolist()]))
                            # print("\tLabel: {}\n\tTop10: {}".format(label, ", ".join([str(t.item()) for t in top_10])))
                        else:
                            break
                    counter += 1 
                iter_string_mrrs.append(sum(string_mrrs) / len(string_mrrs))
            if len(iter_string_mrrs) > 0:
                string_accuracy.append(sum(iter_string_mrrs) / len(iter_string_mrrs))
    print("String accuracy: {}".format(sum(string_accuracy) / len(string_accuracy)))

if __name__ == "__main__":
    main()