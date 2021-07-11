import models.trav_trans.dataset as dataset
import torch, torch.nn, torch.optim
from tqdm import tqdm
import os
import pickle

class Trainer(object):
    def __init__(
        self,
        model,
        setup,
        args
    ):
        super().__init__()
        
        self.model = model
        self.dataset = setup.dataset
        self.load_args(args)

        self.dataloader = torch.utils.data.DataLoader(
            setup.dataset,
            batch_size = self.batch_size,
            collate_fn = lambda b: self.dataset.collate(b, setup.vocab.pad_idx)
        )

    def load_args(self, args):
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.output_dir = args.output_dir
        self.optimizer = args.optimizer
        self.save_model_on_epoch = args.save_model_on_epoch
        self.model_name = args.model_name

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model = self.model.to(device)
        losses = []
        for epoch in range(self.num_epoch):
            batch_counter = 0
            for i, batch in tqdm(enumerate(self.dataloader)):
                x = batch["input_seq"]
                y = batch["target_seq"]
                ext = batch["extended"]
                x = x.to(device)
                y = y.to(device)
                ext = ext.to(device)
                loss = self.model(x, y, ext, return_loss = True)
                loss.backward()
                if batch_counter % 8 == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                if batch_counter % 100 == 0:
                    losses.append([epoch, i, loss.item()])
                if batch_counter % 1000 == 0:
                    print("Epoch {}, It. {}/{}, Loss {}".format(epoch, i, self.dataset.__len__() / self.batch_size, loss))
                    with open(os.path.join(self.output_dir, "losses.pickle"), "wb") as fout:
                        pickle.dump(losses, fout)
                batch_counter += 1
            if self.save_model_on_epoch:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, f"{self.model_name}-{epoch}.pt")
                )
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.output_dir, f"{self.model_name}-final.pt")
        )

class TrainingArgs(object):
    def __init__(
        self,
        batch_size,
        num_epoch,
        optimizer,
        model_name = "transformer4code",
        output_dir = "output",
        save_model_on_epoch = False
    ):
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.model_name = model_name
        self.output_dir = output_dir
        self.save_model_on_epoch = save_model_on_epoch