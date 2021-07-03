from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Replace
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer


# Tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

# Normalizer
# tokenizer.normalizer = Replace(" ", "")

# Pre-Tokenizer
tokenizer.pre_tokenizer = WhitespaceSplit()

# Post-Processing

# Training

trainer = WordLevelTrainer(
    vocab_size=100000,
    special_tokens=[
        "<unk>",
        "<mask>",
        "<pad>",
        "<ast>"
    ]
)
tokenizer.train(["output/new_ast_raw.txt"], trainer)
tokenizer.save("tokenizer/code-tokenizer.json")