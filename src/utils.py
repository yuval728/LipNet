import os
import numpy as np
import torch
import torch.nn.functional as F


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


def split_dataset(dataset, val_split=0.2):
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    return torch.utils.data.random_split(dataset, [n_train, n_val])


def ctc_loss_fn(y_true, y_pred, ctc_loss):
    batch_len = y_true.size(0)  # Number of sequences in the batch
    input_length = y_pred.size(1)  # Time steps per batch sequence
    
    # Correctly create input_lengths with shape (batch_len,)
    input_lengths = torch.full((batch_len,), input_length, dtype=torch.int32)

    # Calculate target lengths based on actual lengths of sequences in y_true
    target_lengths = torch.tensor([len(seq[seq != 0]) for seq in y_true], dtype=torch.int32)

    # print(input_lengths, target_lengths, y_true.size(), y_pred.shape)
    
    y_true_flattened = y_true[y_true != 0].view(-1)  # Flattening while ignoring padding
    
    y_preds_logits = y_pred.permute(1,0,2).log_softmax(dim=-1)


    loss = ctc_loss(y_preds_logits, y_true_flattened, input_lengths, target_lengths)
    
    return loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, min_loss, is_best=False):
    
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'min_loss': min_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')
    
    if is_best:
        best_path = checkpoint_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f'Best model saved at {best_path}')
    
    
def ctc_greedy_decode(y_pred, idx2word, blank_index=40):
    # y_pred: tensor of shape (max_time_steps, batch_size, num_classes)
    
    # Get the predicted class index for each time step
    y_pred_softmax = F.softmax(y_pred, dim=2)
    max_indices = torch.argmax(y_pred_softmax, dim=2)  # Shape: (max_time_steps, batch_size)

    # Decode sequences
    decoded_sequences = []
    for seq in max_indices.permute(1, 0):  # Shape: (batch_size, max_time_steps)
        decoded_seq = []
        prev_index = -1
        for index in seq:
            # Skip duplicates and blank
            if index != blank_index and index != prev_index:
                decoded_seq.append(index.item())
                prev_index = index.item()
#         print(num_to_char(decoded_seq, idx2word))
        decoded_sequences.append(decoded_seq)
    print(decoded_sequences)
    return decoded_sequences