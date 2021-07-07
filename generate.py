import torch
from generator import Generator
from models.trav_trans.dataset import Setup

setup = Setup("output", "output/dps.txt", "output/ids.txt")
vocab = setup.vocab
gen = Generator("output/transformer4code-final.pt", vocab)

idx = vocab.convert([["Module", "Expr", "Str", "Provides mapping for xyz", "ImportFrom", "bootstrap", "alias", "Bootstrap", "ImportFrom", "fund", "alias", "ImportFrom", "fund"], 0])[0]

idx_tensor = torch.tensor(idx)

print(gen.generate(idx_tensor))