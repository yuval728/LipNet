import os
import numpy as np
import torch



def get_word2idx_idx2word(vocab):
    word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
    # word2idx['<PAD>'] = len(word2idx)
    # word2idx['<START>'] = len(word2idx)
    # word2idx['<END>'] = len(word2idx)
    word2idx['<UNK>'] = len(word2idx)
    idx2word = {idx+1: word for idx, word in enumerate(vocab)}
    # idx2word[len(idx2word)] = '<PAD>'
    # idx2word[len(idx2word)] = '<UNK>'
    return word2idx, idx2word

def char_to_num(texts, word2idx):
    return [word2idx[char] for char in texts if char in word2idx]

def num_to_char(nums, idx2word):
    return [idx2word[num] for num in nums]


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, is_best=False):
    
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')
    
    if is_best:
        best_path = checkpoint_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f'Best model saved at {best_path}')
    
    
# def text_to_tensor(text: str, max_len: int = 50) -> torch.Tensor:
#     text = text.lower()
#     text = text.ljust(max_len)[:max_len]
#     tensor = torch.zeros(max_len, dtype=torch.long)
#     for i, c in enumerate(text):
#         tensor[i] = word2idx[c]
#     return tensor