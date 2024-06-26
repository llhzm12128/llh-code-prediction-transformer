import argparse
from long_path_trainer import Trainer, TrainingArgs
import sys
sys.path.append("C:/Users/llh/Desktop/ISCAS/llh-code-prediction-transformer")

from model import TransformerModel
import models.long_path_trans.long_path_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim import Adam

#python models\long_path_trans\long_path_train.py --batch_size 4 --num_epoch 16 --learning_rate  5e-5 --dps tmp\long_path_trans\dps_train.txt --ids tmp\trav_trans\ids_eval.txt 
#--output output\long_path_trans\ --suffix long_path_trans
def main():
    parser = argparse.ArgumentParser(description="Train GPT2 Model")
    parser.add_argument("--batch_size", type=int, default=2, help="Specify batch size")
    parser.add_argument("--num_epoch", type=int, default=16, help="Specify number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Specify AdamW learning rate")
    parser.add_argument("--dps", default="tmp/long_path_trans/dps_train.txt")
    parser.add_argument("--ids", default="tmp/trav_trans/ids_eval.txt") #实际没有使用这个参数
    parser.add_argument("--output", default="output/long_path_trans/")#dp convert后的保存目录，模型保存目录,以及其他训练过程中生成的文件
    parser.add_argument("--suffix", default="long_path_trans")
    parser.add_argument("--save_on_epoch", type=bool, default = False)

    args = parser.parse_args()

    setup = models.long_path_trans.long_path_dataset.Setup(args.output, args.dps, args.ids)

    model = TransformerModel(
        len(setup.vocab.idx2vocab),
        CrossEntropyLoss(ignore_index=setup.vocab.pad_idx),
        6,
        300,
        1000,
        6,
        1e-6,
        root_paths=True
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
