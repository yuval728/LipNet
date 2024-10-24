import torch
import cv2
import os
from torch.utils.data import Dataset
from src.utils import char_to_num, num_to_char



class LipDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, vocab: list, word2idx: dict, idx2word: dict, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data = os.listdir(data_dir)
        self.label = os.listdir(label_dir)
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data[idx])
        label_path = os.path.join(self.label_dir, self.label[idx])

        assert os.path.exists(data_path), f"Data path {data_path} does not exist"
        assert os.path.exists(label_path), f"Label path {label_path} does not exist"

        assert (
            data_path.split("/")[-1].split(".")[0]
            == label_path.split("/")[-1].split(".")[0]
        ), "Data and label file names do not match"

        frames = self.load_video(data_path)

        label = self.load_alignment(label_path)

        return frames, label

    def load_video(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = frame[
                190:236, 80:220
            ]  # TODO: Make it dynamic using dlib  # Take only the lip part of the frame
            
            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        cap.release()
        
        frames = torch.stack(frames)
        
        std = torch.std(frames)
        mean = torch.mean(frames)
        
        frames = (frames - mean) / std # Normalize the frames (z-score normalization

        return frames # (T, H, W, C)
    
    
    def load_alignment(self, path: str) -> torch.Tensor:
        with open(path, "r") as f:
            lines = f.readlines() 
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != "sil":
                # tokens = [*tokens, ' ',line[2]]
                tokens.append(' ')
                tokens.extend(list(line[2]))  

        token_nums = char_to_num(tokens, self.word2idx)

        
        return torch.tensor(token_nums[1:], dtype=torch.long)