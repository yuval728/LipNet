import torch
import numpy as np
import os
from tqdm.auto import tqdm
import argparse
import dataset
import utils
import constants
import models
from torchvision import transforms

word2idx, idx2word = utils.get_word2idx_idx2word(constants.vocab)

def test(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    total_wer = 0
    total_cer = 0
    with torch.inference_mode():
        for frames, labels in tqdm(test_loader):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames.permute(0, 2, 1, 3, 4))
            loss = utils.ctc_loss_fn(labels, outputs, criterion, device)
            test_loss += loss.item()
            decoded_preds = utils.ctc_greedy_decode(outputs.clone(), idx2word, print_output=False)

            truth  = [utils.num_to_char(label.tolist(), idx2word) for label in labels]
            decoded_preds = [utils.num_to_char(pred, idx2word) for pred in decoded_preds]
            word_error_rate = utils.word_error_rate(truth, decoded_preds)
            char_error_rate = utils.character_error_rate(truth, decoded_preds)
            total_wer += word_error_rate
            total_cer += char_error_rate
            print(f'Word Error Rate: {word_error_rate} | Character Error Rate: {char_error_rate}')
            
    test_loss /= len(test_loader)
    total_wer /= len(test_loader)
    total_cer /= len(test_loader)
    
    return test_loss, total_wer, total_cer

def parse_args():
    parser = argparse.ArgumentParser( description='Test the LipNet model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = dataset.LipDataset(args.data_dir, args.label_dir, transform=test_transforms, vocab=constants.vocab, word2idx=word2idx,idx2word=idx2word)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collate_fn)
    
    model = models.LipNet(len(word2idx), input_channels=3, hidden_size=128)
    model.to(args.device)
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from {checkpoint['epoch']} epoch with loss {checkpoint['loss']}")
    
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    
    test_loss, total_wer, total_cer = test(model, test_loader, args.device, criterion)
    
    print(f'Test Loss: {test_loss} | Total WER: {total_wer} | Total CER: {total_cer}')
    
if __name__ == '__main__':
    main()