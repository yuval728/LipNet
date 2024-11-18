import os
import numpy as np
import torch
import torch.nn.functional as F
import editdistance
import cv2
import mediapipe as mp



def extract_lip_region_from_image(frame,face_mesh, padding=20):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the lip landmarks (use indices for lips from MediaPipe's documentation)
            lips_indices = [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
                402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415
            ]
            h, w, _ = frame.shape
            lip_coords = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in lips_indices]
            
            # Get bounding box around lips
            x_min = min([x for x, y in lip_coords])
            y_min = min([y for x, y in lip_coords])
            x_max = max([x for x, y in lip_coords])
            y_max = max([y for x, y in lip_coords])
            
            # Add some padding
            x_min, y_min = max(x_min - padding, 0), max(y_min - padding//2, 0)
            x_max, y_max = min(x_max + padding, w), min(y_max + padding//2, h)
            
            # Crop the lip region
            lip_region = frame[y_min:y_max, x_min:x_max]
            
            lip_region = cv2.resize(lip_region, (100, 50))
            
            return lip_region
    return None


def extract_lip_region_from_video(video_path, face_mesh, padding=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lip_region = extract_lip_region_from_image(frame, face_mesh, padding)
        if lip_region is not None:
            frames.append(lip_region)
    cap.release()
    return frames


def get_word2idx_idx2word(vocab):
    word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
    word2idx[''] = 0

    idx2word = {idx+1: word for idx, word in enumerate(vocab)}
    idx2word[0] = ''
    return word2idx, idx2word

def char_to_num(texts, word2idx):
    return [word2idx[char] for char in texts if char in word2idx]

def num_to_char(nums, idx2word):
    return [idx2word[num] for num in nums]


def split_dataset(dataset, val_split=0.2):
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    return torch.utils.data.random_split(dataset, [n_train, n_val])


def ctc_loss_fn(y_true, y_pred, ctc_loss, device):
    batch_len = y_true.size(0)  # Number of sequences in the batch
    input_length = y_pred.size(1)  # Time steps per batch sequence

    input_lengths = torch.full((y_pred.size(0),), y_pred.size(1), dtype=torch.long).to(device)
    target_lengths = torch.full((y_true.size(0),), y_true.size(1), dtype=torch.long).to(device)

    # print(input_lengths, target_lengths, y_true.size(), y_pred.shape)
    
    y_preds_logits = y_pred.permute(1,0,2).log_softmax(dim=2)

    loss = ctc_loss(y_preds_logits, y_true, input_lengths, target_lengths)
    
    return loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, min_loss, is_best=False):
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'min_loss': min_loss,
    }
    
    checkpoint_name = os.path.join(checkpoint_path, f'checkpoint_{epoch}.pt')
    torch.save(checkpoint, checkpoint_name)
    print(f'Checkpoint saved at {checkpoint_name}')
    
    if is_best:
        best_path = checkpoint_name.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f'Best model saved at {best_path}')
    
    
def ctc_greedy_decode(y_pred, idx2word, blank_index=0, print_output=True):
    # Apply softmax to the model outputs to get probabilities
    probs = F.softmax(y_pred, dim=-1)
    
    # Get the predicted classes by taking the argmax
    predicted_indices = torch.argmax(probs, dim=-1)  # Shape: (batch_size, max_time)

    # Now we will decode the indices into strings
    decoded_outputs = []
    for batch_idx in range(predicted_indices.size(0)):
        current_output = []
        previous_index = -1  # Initialize to -1 to not include blank at the start
        
        for time_step in range(predicted_indices.size(1)):
            index = predicted_indices[batch_idx, time_step].item()
            if index != blank_index and index != previous_index:
                current_output.append(index)
            previous_index = index
        
        decoded_outputs.append(current_output)  # Store decoded output for each batch
    if print_output:
        print(decoded_outputs)

    return decoded_outputs


def word_error_rate(actual, prediction):
    
    word_pairs = []
    for act, pred in zip(actual, prediction):
        act = ''.join(act).split(" ")
        pred = ''.join(pred).split(" ")
        word_pairs.append((act, pred))
    wer = [editdistance.eval(act, pred)/len(act) for act, pred in word_pairs]
    return np.mean(wer)

def character_error_rate(actual, prediction):
    char_pairs = [(list(act), list(pred)) for act, pred in zip(actual, prediction)]
    
    cer = [editdistance.eval(act, pred)/len(act) for act, pred in char_pairs]
    return np.mean(cer)
    
    
    # error_count = 0
    # total_count = 0
    # for act, pred in word_pairs:
    #     total_count += len(act)
    #     error_count += sum([1 for a, b in zip(act, pred) if a != b])
    # error_count = 0
    # total_count = 0
    # for act, pred in char_pairs:
    #     total_count += len(act)
    #     error_count += sum([1 for a, b in zip(act, pred) if a != b])
    # return error_count / total_count

    