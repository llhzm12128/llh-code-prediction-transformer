from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = CharDelimiterSplit(delimiter=",")
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"])

tokenizer.train(["output/train_raw.json"], trainer)

tokenizer.save("output/tokenizer.json")
