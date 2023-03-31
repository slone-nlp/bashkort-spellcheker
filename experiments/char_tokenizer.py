"""
Copypasted from 
https://huggingface.co/IlyaGusev/ru-word-stress-transformer/blob/main/char_tokenizer.py
with Apache 2.0 license
"""

import os
from typing import Optional, Tuple, List
from collections import OrderedDict

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer


def load_vocab(vocab_file):
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class CharTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}

    def __init__(
        self,
        vocab_file=None,
        pad_token="[pad]",
        unk_token="[unk]",
        bos_token="[bos]",
        eos_token="[eos]",
        cls_token="[cls]",
        sep_token="[sep]",
        mask_token="[mask]",
        space_token="▁",
        do_lower_case=False,
        *args,
        **kwargs
    ):
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs
        )
        self.do_lower_case = do_lower_case
        self.space_token = space_token

        if not vocab_file or not os.path.isfile(vocab_file):
            self.vocab = OrderedDict()
            self.ids_to_tokens = OrderedDict()
        else:
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    def train(self, file_path):
        vocab = set()
        with open(file_path) as r:
            for line in r:
                word = line.strip()
                if self.do_lower_case:
                    word = word.lower()
                vocab |= set(word)
        vocab = list(vocab)
        vocab.sort()
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        vocab = special_tokens + vocab

        for i, ch in enumerate(vocab):
            self.vocab[ch] = i
        self.ids_to_tokens = vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _convert_token_to_id(self, token):
        if self.do_lower_case:
            token = token.lower()
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens[index]
    
    def prepare_for_tokenization(
        self, text, is_split_into_words: bool = False, spaces=0, **kwargs
    ):
        if spaces:
            pad = self.space_token * spaces
            text = pad + pad.join(text) + pad
        return (text, kwargs)

    def _tokenize(self, text, spaces=0):
        if self.do_lower_case:
            text = text.lower()
        return list(text)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        return bos + token_ids_0 + eos

    def get_special_tokens_mask(
         self,
         token_ids_0: List[int],
         token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return (len(token_ids_0) + 2) * [0]

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        assert os.path.isdir(save_directory)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") +
            self.vocab_files_names["vocab_file"]
        )
        index = 0
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                assert index == token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
    
    def clean_up_tokenization(self, text, space='▁'):
        res = []
        prev = space
        for c in text:
            if c != prev and c != space:
                res.append(c)
            prev = c
        return ''.join(res)

AutoTokenizer.register("char_tokenizer", CharTokenizer)