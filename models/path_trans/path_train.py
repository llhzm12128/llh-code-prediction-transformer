import argparse
from path_trainer import Trainer, TrainingArgs
import sys
sys.path.append("C:/Users/llh/Desktop/ISCAS/llh-code-prediction-transformer")

from model import TransformerModel
import models.path_trans.path_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


#python models\path_trans\path_train.py --batch_size 4 --num_epoch 16 --learning_rate 5e-5 --dps tmp\path_trans\dps_train.txt --output output\path_trans --suffix path_trans
def main():
    parser = argparse.ArgumentParser(description="Train GPT2 Model")
    parser.add_argument("--batch_size", type=int, default=4, help="Specify batch size")
    parser.add_argument("--num_epoch", type=int, default=16, help="Specify number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Specify AdamW learning rate")
    parser.add_argument("--dps", default="tmp/path_trans/")
    parser.add_argument("--ids", default="tmp/trav_trans/ids_train.txt")
    parser.add_argument("--output", default="output/path_trans")
    parser.add_argument("--suffix", default="path_trans")
    parser.add_argument("--save_on_epoch", type=bool, default = False)

    args = parser.parse_args()

    setup = models.path_trans.path_dataset.Setup(args.output, args.dps, args.ids)

    model = TransformerModel(
        len(setup.vocab.idx2vocab),
        CrossEntropyLoss(ignore_index=setup.vocab.pad_idx),
        6,
        300,
        1000,
        6,
        1e-05,
        root_paths=True
    )

    training_args = TrainingArgs(
        batch_size = args.batch_size,
        num_epoch = args.num_epoch,
        output_dir = args.output,
        optimizer = AdamW(model.parameters(), lr=args.learning_rate),
        save_model_on_epoch = args.save_on_epoch,
        suffix = args.suffix
    )

    trainer = Trainer(
        model,
        setup,
        training_args
    )

    trainer.train()


if __name__ == "__main__":
    main()
