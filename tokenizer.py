from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = CharDelimiterSplit(delimiter=",")
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"])

tokenizer.train(["output/train_raw.json"], trainer)

tokenizer.save("output/tokenizer.json")
