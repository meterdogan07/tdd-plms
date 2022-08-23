import os
from tokenizers import trainers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex
from tokenizers.processors import TemplateProcessing

if not os.path.isdir("Tokenizers"):
    os.mkdir("Tokenizers")

reg = r"[^\'\n\ \!\"\#\$\%\&\'\(\)\*\+\,\-\.\/0123456789\:\;\<\=\>\?\@ABCDEFGHIJKLMNOPQRSTUVWXYZ\[\\\]\^\_\`abcdefghijklmnopqrstuvwxyz\{\|\}\~\±\´\·ÂÃÄÅÇÖÜâçéêîöûüĞğ\İ\ıŞş\‘\’\…\′\']"

def build_spiece_tokenizer(vocab_size):
    """ Change based on tokenizer model."""

    # Normalizer is set as NFKC and regex is defined. 
    # Normalizer splits: digits, whitespaces (regex: \w+|[^\w\s]+), and punctuations.
    # Pre-tokenizer, post-proccessing and decoder are ByteLevel.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(reg), "")])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation(), pre_tokenizers.ByteLevel()])
    tokenizer.decoder = decoders.ByteLevel()

    # Set training settings
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    vocab_size = vocab_size - len(special_tokens)
    
    
    # Change trainer to match tokenizer model
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens) 

    # Postprocessing is changed, as the default is for GPT-2.
    # TODO: Let's double check this is same as BART
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 1)]
    )

    # Train tokenizer
    tokenizer.train([f"tdd0{i}.txt" for i in range(10)], trainer=trainer)

    return tokenizer


for m, v in (("8k", 2**13), ("16k", 2**14), ("32k", 2**15), ("64k", 2**16)):
    toke = build_spiece_tokenizer(v)
    toke.save(f"Tokenizers/BPE.byte.tokenizer.{m}.json", pretty=True)