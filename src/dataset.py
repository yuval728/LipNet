import torch
from torchvision import transforms
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from .utils import  get_word2idx_idx2word, char_to_num, num_to_char



class LipDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, vocab: list, word2idx: dict, idx2word: dict, transform=transforms.ToTensor()) -> None:
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
        try:
            data_path = os.path.join(self.data_dir, self.data[idx])
            label_file = self.data[idx].split(".")[0] + ".align"
            label_path = os.path.join(self.label_dir, label_file)
        

            assert os.path.exists(data_path), f"Data path {data_path} does not exist"
            assert os.path.exists(label_path), f"Label path {label_path} does not exist"

            assert (
                data_path.split("/")[-1].split(".")[0]
                == label_path.split("/")[-1].split(".")[0]
            ), "Data and label file names do not match"

            frames = self.load_video(data_path)
            if frames is None:
                print(idx)

            label = self.load_alignment(label_path)

            return frames, label
        except Exception as e:
            print(idx, e)

    def get_data_name(self, idx):
        return self.data[idx].split(".")[0]

    def get_data_idx(self, name):
        return self.data.index(name + ".mpg")


    def load_video(self, path: str) -> torch.Tensor:
        np_frames = np.load(path)
        frames = []
        for i in np_frames:
            frames.append(self.transform(i))
            
        frames = torch.stack(frames)
        
        # Normalize frames (Z-score normalization)
        std = torch.std(frames)
        mean = torch.mean(frames)
        frames = (frames - mean) / std

        return frames  # (T, C, H, W) format

    def load_alignment(self, path: str) -> torch.Tensor:
        with open(path, "r") as f:
            lines = f.readlines() 
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != "sil":
                tokens.append(' ')
                tokens.extend(list(line[2]))  

        token_nums = char_to_num(tokens, self.word2idx)
        return torch.tensor(token_nums[1:], dtype=torch.long)
 
      
def collate_fn(batch, pad_value=0):
    frames, labels = zip(*batch)

    # Pad the frames to the same length
    max_len = max([f.shape[0] for f in frames])
    frames = [torch.nn.functional.pad(input=f, pad=(0, 0, 0, 0, 0, 0, 0, max_len - f.shape[0]), mode='constant', value=0) for f in frames] 
    
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return torch.stack(frames), labels



if __name__ == '__main__':
    import constants
    
    vocab = constants.vocab
    word2idx, idx2word = get_word2idx_idx2word(vocab)
    
    data_dir = "data/s1"
    label_dir = "data/alignments/s1"
    
    
    data_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),                      
            # transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
        ]
    )

    dataset = LipDataset(data_dir=data_dir, label_dir=label_dir, vocab=vocab, word2idx=word2idx, idx2word=idx2word, transform=data_transform)
    
    print(len(dataset))
    frames, label = dataset[50]
    print(frames.shape, label.shape)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    frames, labels = next(iter(loader))
    print(frames.shape, labels.shape)