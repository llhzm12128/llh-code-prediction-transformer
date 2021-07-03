from trainer import Trainer, TrainingArgs
import model
import models.trav_trans.dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

setup = models.trav_trans.dataset.Setup("tmp", "tmp/dps.txt", "tmp/ids.txt")

model = model.TransformerModel(
    len(setup.vocab.idx2vocab),
    CrossEntropyLoss(ignore_index=setup.vocab.pad_idx),
    6,
    300,
    1000,
    6,
    1e-05
)

training_args = TrainingArgs(
    batch_size = 2,
    num_epoch = 1,
    output_dir = "output",
    optimizer = AdamW(model.parameters(), lr=5e-5),
    save_model_on_epoch = True
)

trainer = Trainer(
    model,
    setup,
    training_args
)

trainer.train()