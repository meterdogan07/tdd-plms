import sys
import os
import json
import random

from glob import glob
from tokenizers import ByteLevelBPETokenizer, SentencePieceUnigramTokenizer
from tokenizers import trainers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex
from tokenizers import processors

if not os.path.isdir("Tokenizers"):
    os.mkdir("Tokenizers")

reg = r"[^\'\n\ \!\"\#\$\%\&\'\(\)\*\+\,\-\.\/0123456789\:\;\<\=\>\?\@ABCDEFGHIJKLMNOPQRSTUVWXYZ\[\\\]\^\_\`abcdefghijklmnopqrstuvwxyz\{\|\}\~\±\´\·ÂÃÄÅÇÖÜâçéêîöûüĞğ\İ\ıŞş\‘\’\…\′\']"

def build_spiece_tokenizer(vocab_size):
    # Change based on tokenizer model
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(reg), "")])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation(), pre_tokenizers.Metaspace()])
    tokenizer.decoder = decoders.Metaspace()
    special_tokens = ["<pad>", "<unk>", "<eos>"]
    
    # Change trainer to match tokenizer model
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(special_tokens),
        special_tokens=special_tokens
    )

    tokenizer.unk_id = 1
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$0 <eos>",
        special_tokens=[("<eos>", 2), ],
    )

    tokenizer.train([f"tdd0{i}.txt" for i in range(10)], trainer=trainer)

    return tokenizer


for m, v in (("8k", 2**13), ("16k", 2**14), ("32k", 2**15), ("64k", 2**16)):
    toke = build_spiece_tokenizer(v)
    toke.save(f"Tokenizers/BPE.spiece.tokenizer.trwiki-67.{m}.json", pretty=True)
