import argparse
from trainer import Trainer, TrainingArgs
from model import TransformerModel
import models.trav_trans.dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW,Adam

#python train.py --batch_size 4 --num_epoch 12 --learning_rate 5e-5 --dps tmp/trav_trans/dps_train.txt --ids tmp/trav_trans/ids_train.txt --suffix trav_trans

def main():
    parser = argparse.ArgumentParser(description="Train GPT2 Model")
    parser.add_argument("--batch_size", type=int, default=4, help="Specify batch size")
    parser.add_argument("--num_epoch", type=int, default=3, help="Specify number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Specify AdamW learning rate")
    parser.add_argument("--dps", default="tmp/trav_trans/dps_train.txt")
    parser.add_argument("--ids", default="tmp/ids_100k_train.txt")
    parser.add_argument("--output", default="output/travtrans")
    parser.add_argument("--suffix", default="unnamed")
    parser.add_argument("--save_on_epoch", type=bool, default = False)

    args = parser.parse_args()

    setup = models.trav_trans.dataset.Setup(args.output, args.dps, args.ids)

    model = TransformerModel(
        len(setup.vocab.idx2vocab),
        CrossEntropyLoss(ignore_index=setup.vocab.pad_idx),
        6,
        300,
        1000,
        6,
        1e-06
    )

    training_args = TrainingArgs(
        batch_size = args.batch_size,
        num_epoch = args.num_epoch,
        output_dir = args.output,
        optimizer = Adam(model.parameters(), lr=args.learning_rate),
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
