import torch
from torch.nn import CrossEntropyLoss
from model import TransformerModel

class Generator():
    def __init__(
        self,
        model_path,
        vocab
    ):
        super(Generator, self).__init__()
        self.model = TransformerModel(
            len(vocab.idx2vocab),
            CrossEntropyLoss(ignore_index=vocab.pad_idx),
            6,
            300,
            1000,
            6,
            1e-05
        )
        self.model.load_state_dict(torch.load(model_path))
    
    def generate(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.input = input.to(device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input, None)
            return out