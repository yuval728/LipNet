import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import os
from tqdm.auto import tqdm
import argparse
import math

from dataset import LipDataset, collate_fn
import utils
import constants
from models import LipNet


word2idx, idx2word = utils.get_word2idx_idx2word(constants.vocab)


def train(model, dataloader, criterion, optimizer, device, lr_scheduler, print_every=40):
    
    model.train()
    
    total_loss = 0.0
    
    for i, (frames, labels) in enumerate(dataloader):
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(frames.permute(0,2,1,3,4))
        

        loss = utils.ctc_loss_fn(labels, output, criterion)
         
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        frames, labels = frames.cpu(), labels.cpu()
        
        if (i+1) % print_every == 0:
            utils.ctc_greedy_decode(output.clone(), idx2word)
            print(f'Batch {i+1}/{len(dataloader)} - Loss: {loss.item()}')
    
    lr_scheduler.step()
    print(f'Learning rate: {lr_scheduler.get_last_lr()}')
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, print_every=10):
    model.eval()
    
    total_loss = 0.0
    
    with torch.inference_mode():
        for i, (frames, labels) in enumerate(dataloader):
            frames, labels = frames.to(device), labels.to(device)
            
            output = model(frames.permute(0,2,1,3,4))
             
            loss = utils.ctc_loss_fn(labels, output, criterion)
            
            total_loss += loss.item()
            
            frames, labels = frames.cpu(), labels.cpu()
            
            if (i+1) % print_every == 0:
                utils.ctc_greedy_decode(output.clone(), idx2word)
                print(f'Batch {i+1}/{len(dataloader)} - Loss: {loss.item()}')
                
            
    return total_loss / len(dataloader)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler, num_epochs, device, checkpoint_path, prev_checkpoint=None, print_every=10):

     
    min_val_loss = float('inf')
    start_epoch = 0

    if prev_checkpoint is not None:
        checkpoint = torch.load(prev_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        # min_val_loss = checkpoint['min_loss']
        print(f'Model loaded from checkpoint {prev_checkpoint}', checkpoint.keys())
   
        
    
    loss_history = {'train': [], 'val': []}    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'Epoch {epoch}/{num_epochs}')
        
        # train_loss = train(model, train_dataloader, criterion, optimizer, device, lr_scheduler,print_every )
        # loss_history['train'].append(train_loss)
        
        val_loss = evaluate(model, val_dataloader, criterion, device, print_every)
        loss_history['val'].append(val_loss)
        
        # print(f'Train Loss: {train_loss} - Val Loss: {val_loss}')
        
        min_val_loss = min(min_val_loss, val_loss)
        
        utils.save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, min_val_loss, is_best=(val_loss == min_val_loss))
        
    return loss_history


def parse_args():
    parser = argparse.ArgumentParser(description='Train the LipNet model')
    
    parser.add_argument('--data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--label_dir', type=str, help='Path to the directory containing the alignments')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save the model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--print_every', type=int, default=40, help='Print every n iterations')
    parser.add_argument('--seed', type=int, default=71, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
    parser.add_argument('--prev_checkpoint', type=str, default=None, help='Path to a previous checkpoint to resume training')
    parser.add_argument('--lr_schedule_step', type=int, default=50, help='Step size for the learning rate scheduler')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for the LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),       
        transforms.Grayscale(num_output_channels=1),
    ])
    
    dataset = LipDataset(args.data_dir, args.label_dir, constants.vocab, word2idx, idx2word, transform)
    
    train_dataset, val_dataset = utils.split_dataset(dataset, args.split_ratio)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    model = LipNet(vocab_size=len(word2idx),input_size=75, input_channels=1, hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    
    criterion = nn.CTCLoss(blank=40, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lambda_lr = lambda epoch: 1.0 if epoch < 40 else math.exp(-0.1 * (epoch - args.lr_schedule_step))   # noqa: E731
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    
    loss_history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler, args.num_epochs, device, checkpoint_path, prev_checkpoint=args.prev_checkpoint, print_every=args.print_every)
    
    return loss_history

if __name__ == '__main__':
    loss_history = main()
    print(loss_history)