import os
import numpy as np
import torch



def get_word2idx_idx2word(vocab):
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    word2idx['<PAD>'] = len(word2idx)
    # word2idx['<START>'] = len(word2idx)
    # word2idx['<END>'] = len(word2idx)
    # word2idx['<UNK>'] = len(word2idx)
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def char_to_num(texts, word2idx):
    return [word2idx[char] for char in texts if char in word2idx]

def num_to_char(nums, idx2word):
    return [idx2word[num] for num in nums]


# def text_to_tensor(text: str, max_len: int = 50) -> torch.Tensor:
#     text = text.lower()
#     text = text.ljust(max_len)[:max_len]
#     tensor = torch.zeros(max_len, dtype=torch.long)
#     for i, c in enumerate(text):
#         tensor[i] = word2idx[c]
#     return tensor