import torch
from torchvision import transforms
import cv2
import os
from torch.utils.data import Dataset
from utils import  get_word2idx_idx2word, char_to_num, num_to_char



class LipDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, vocab: list, word2idx: dict, idx2word: dict, transform=transforms.ToTensor()) -> None:
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data = os.listdir(data_dir)
        self.data.remove('sgib8n.mpg')
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
            
#             print(idx, label_file)

            return frames, label
        except Exception as e:
            print(idx, e)

    def load_video(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = frame[
                190:236, 80:220, :
            ]  # TODO: Make it dynamic using dlib  # Take only the lip part of the frame
            
            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        cap.release()
        
        frames = torch.stack(frames)
        
        std = torch.std(frames)
        mean = torch.mean(frames)
#         print(std, mean)
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
    
def collate_fn(batch, pad_value=0):
    frames, labels = zip(*batch)

    # Pad the frames to the same length
    max_len = max([f.shape[0] for f in frames])
    frames = [torch.nn.functional.pad(input=f, pad=(0, 0, 0, 0, 0, 0, 0, max_len - f.shape[0]), mode='constant', value=0) for f in frames] 
    
    # Pad the labels to the same length
    max_len = max([l.shape[0] for l in labels])  # noqa: E741
    labels = [torch.nn.functional.pad(input=l, pad=(0, max_len - l.shape[0]), mode='constant', value=pad_value) for l in labels]  # noqa: E741
    
    return torch.stack(frames), torch.stack(labels)



if __name__ == '__main__':
    import constants
    
    vocab = constants.vocab
    word2idx, idx2word = get_word2idx_idx2word(vocab)
    
    data_dir = "data/s1"
    label_dir = "data/alignments/s1"
    
    
    data_transform = transforms.Compose(
        [
            transforms.ToPILImage(),                      # Convert the OpenCV image (NumPy array) to a PIL image
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
            # transforms.ToTensor(),                        # Convert the PIL image to a PyTorch tensor (values between 0 and 1)
            # transforms.Normalize(mean=[0.5], std=[0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    dataset = LipDataset(data_dir=data_dir, label_dir=label_dir, vocab=vocab, word2idx=word2idx, idx2word=idx2word, transform=data_transform)
    
    print(len(dataset))
    frames, label = dataset[50]
    print(frames.shape, label.shape)