import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

from tqdm.auto import tqdm
import argparse
import mlflow

from .dataset import LipDataset, collate_fn
from . import constants, utils
from .models import LipNet

word2idx, idx2word = utils.get_word2idx_idx2word(constants.vocab)


def train(model, dataloader, criterion, optimizer, device, print_every=40):
    model.train()

    total_loss = 0.0

    for i, (frames, labels) in enumerate(dataloader):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()

        frames = frames.permute(0, 2, 1, 3, 4)
        output = model(frames)

        loss = utils.ctc_loss_fn(labels, output, criterion, device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        frames, labels = frames.cpu(), labels.cpu()

        if (i + 1) % print_every == 0:
            utils.ctc_greedy_decode(output.clone(), idx2word)
            print(f"Batch {i+1}/{len(dataloader)} - Loss: {loss.item()}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, print_every=10):
    model.eval()

    total_loss = 0.0

    with torch.inference_mode():
        for i, (frames, labels) in enumerate(dataloader):
            frames, labels = frames.to(device), labels.to(device)

            output = model(frames.permute(0, 2, 1, 3, 4))

            loss = utils.ctc_loss_fn(labels, output, criterion, device)

            total_loss += loss.item()

            frames, labels = frames.cpu(), labels.cpu()

            if (i + 1) % print_every == 0:
                utils.ctc_greedy_decode(output.clone(), idx2word)
                print(f"Batch {i+1}/{len(dataloader)} - Loss: {loss.item()}")

    return total_loss / len(dataloader)


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    num_epochs,
    device,
    checkpoint_path="checkpoints",
    prev_checkpoint=None,
    new_lr=None,
    print_every=10,
):
    min_val_loss = float("inf")
    start_epoch = 0

    if prev_checkpoint is not None:
        checkpoint = torch.load(prev_checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        min_val_loss = checkpoint["min_loss"]
        print(
            f"Model loaded from checkpoint {prev_checkpoint}",
            start_epoch,
            "Old Loss",
            min_val_loss,
        )
        if new_lr:
            for param_group in optimizer.param_groups:
                print("Old lr: ", param_group["lr"])
                param_group["lr"] = new_lr
                print("New lr: ", param_group["lr"])

    loss_history = {"train": [], "val": []}
    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
        print(f"Epoch {epoch}/{start_epoch+num_epochs}")

        train_loss = train(
            model,
            train_dataloader,
            criterion,
            optimizer,
            device,
            print_every,
        )
        loss_history["train"].append(train_loss)

        val_loss = evaluate(model, val_dataloader, criterion, device, print_every)
        loss_history["val"].append(val_loss)

        print(f"Train Loss: {train_loss} - Val Loss: {val_loss}")

        min_val_loss = min(min_val_loss, val_loss)
        
        is_best = val_loss == min_val_loss

        utils.save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            checkpoint_path,
            min_val_loss,
            is_best=is_best,
        )
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        
    
    return loss_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LipNet model")

    parser.add_argument("--data_dir", type=str, help="Path to the dataset directory")
    parser.add_argument(
        "--label_dir", type=str, help="Path to the directory containing the alignments"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to save the checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the dataloader",
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--print_every", type=int, default=40, help="Print every n iterations"
    )
    parser.add_argument("--seed", type=int, default=71, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--prev_checkpoint",
        type=str,
        default=None,
        help="Path to a previous checkpoint to resume training",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden size for the LSTM layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability"
    )
    parser.add_argument(
        "--input_channels", type=int, default=3, help="Number of input channels"
    )
    parser.add_argument(
        "--new_lr",
        type=float,
        default=None,
        help="New learning rate when resuming training",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="LipNet",
        help="Name of the experiment for MLflow",
    )
    parser.add_argument(
        "--mlflow_url",
        type=str,
        default="http://localhost:5000",
        help="URL of the MLflow server",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    mlflow.set_tracking_uri(args.mlflow_url)
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(log_system_metrics=True, nested=True)
    mlflow.log_params(vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = LipDataset(
        args.data_dir, args.label_dir, constants.vocab, word2idx, idx2word, transform
    )

    train_dataset, val_dataset = utils.split_dataset(dataset, args.split_ratio)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = LipNet(
        vocab_size=len(constants.vocab),
        input_channels=args.input_channels,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CTCLoss(zero_infinity=True, blank=0)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    loss_history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        checkpoint_path=args.checkpoint_path,
        prev_checkpoint=args.prev_checkpoint,
        print_every=args.print_every,
        new_lr=args.new_lr,
    )

    mlflow.end_run()
    
    return loss_history


if __name__ == "__main__":
    loss_history = main()
    print(loss_history)
