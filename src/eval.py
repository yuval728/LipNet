import torch
import numpy as np
import os
from tqdm.auto import tqdm
import argparse
from . import dataset, utils, constants, models
from torchvision import transforms
import mlflow

word2idx, idx2word = utils.get_word2idx_idx2word(constants.vocab)

def test(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    total_wer = 0
    total_cer = 0
    
    count = 0
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
            
            count += 1
            if count%10==0:
                print([("".join(ref),"".join(pred)) for ref, pred in zip(truth, decoded_preds)])
            # print(f'Word Error Rate: {word_error_rate} | Character Error Rate: {char_error_rate}')
            
    test_loss /= len(test_loader)
    total_wer /= len(test_loader)
    total_cer /= len(test_loader)
    
    return test_loss, total_wer, total_cer

def parse_args():
    parser = argparse.ArgumentParser( description='Test the LipNet model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint to be loaded')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--label_dir', type=str, required=True, help='Path to the label directory')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5000', help='URI of the MLFlow server')
    parser.add_argument('--experiment_name', type=str, default='LipNet', help='Name of the experiment')
    parser.add_argument('--run_id', type=str, help='ID of the run to log the metrics')
    return parser.parse_args()

def main():
    args = parse_args()
    
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(run_id=args.run_id, nested=True)
    
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = dataset.LipDataset(args.data_dir, args.label_dir, transform=test_transforms, vocab=constants.vocab, word2idx=word2idx,idx2word=idx2word)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collate_fn)
    
    model = models.LipNet(len(constants.vocab), input_channels=3, hidden_size=256)
    model.to(args.device)
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from {checkpoint['epoch']} epoch with loss {checkpoint['loss']}")
    
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    
    test_loss, total_wer, total_cer = test(model, test_loader, args.device, criterion)
    
    print(f'Test Loss: {test_loss} | Total WER: {total_wer} | Total CER: {total_cer}')
    
    mlflow.log_metrics({'test_loss': test_loss, 'total_wer': total_wer, 'total_cer': total_cer})
    
    mlflow.end_run()
    
if __name__ == '__main__':
    main()