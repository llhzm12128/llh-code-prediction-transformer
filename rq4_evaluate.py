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

def eval(model_fp, dps, ids, tokenizer):
    dataset = Dataset(dps)
    pad_idx = tokenizer.encode("[PAD]").ids[0]
    m = model.from_file(model_fp, tokenizer.get_vocab_size(), pad_idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda b: dataset.collate(b, pad_idx)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    m.eval()

    print("Evaluating {} batches".format(len(dataloader)))
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            if i == 0:
                x = batch["input_seq"][0]
                y = batch["target_seq"][0]
                x = x.to(device)
                output = m(x, None)
                for j in range(len(y)):
                    print("y @{}: {}".format(j, tokenizer.decode([y[j].item()])))
                    print("y_predict top 10 @{}: {}\n".format(j, [tokenizer.decode([c.item()]) for c in torch.topk(output[j], 10).indices]))
            else:
                return

if __name__ == "__main__":
    main()