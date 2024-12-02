import os
import mediapipe as mp
import mlflow
import logging
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.download_data import download_data
from src.preprocess import save_lip_region_from_video_to_npy
from src.dataset import LipDataset, collate_fn
from src.models import LipNet
from src.train import train_model
from src.eval import test
from src import utils
from src.constants import vocab
from src.trace_model import trace_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def main():
    # Load configuration
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(log_system_metrics=True) as run:
        mlflow.log_params(config)

        # Download data
        if config["data"]["download"]:
            logging.info("Downloading data...")
            download_data(config["data"]["url"])
            logging.info("Data downloaded successfully!")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")

        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])

        word2idx, idx2word = utils.get_word2idx_idx2word(vocab)

        # Preprocess data
        logging.info("Preprocessing data...")
        # face_mesh = mp.solutions.face_mesh.FaceMesh(
        #     static_image_mode=True, max_num_faces=1, refine_landmarks=True
        # )
        # save_lip_region_from_video_to_npy(
        #     config["data"]["videos_path"],
        #     face_mesh,
        #     padding=config["data"]["padding"],
        #     output_path=config["data"]["lip_region_path"],
        # )
        # face_mesh.close()
        logging.info("Data preprocessed successfully!")

        # Transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Load datasets
        logging.info("Loading datasets...")
        data = LipDataset(
            config["data"]["lip_region_path"],
            config["data"]["label_path"],
            vocab,
            word2idx,
            idx2word,
            transform=transform,
        )

        test_data = None
        if config["test_split_ratio"] is None or config["test_split_ratio"] == 0:
            train_data, val_data = utils.split_dataset(
                data, val_split=config["val_split_ratio"], seed=config["seed"]
            )
        else:
            train_data, val_data, test_data = utils.split_dataset(
                data,
                val_split=config["val_split_ratio"],
                seed=config["seed"],
                test_split=config["test_split_ratio"],
            )
        logging.info("Datasets loaded successfully!")

        # Create data loaders
        logging.info("Creating data loaders...")
        train_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            collate_fn=collate_fn,
        )
        logging.info("Data loaders created successfully!")

        test_loader = DataLoader(
            test_data if test_data is not None else val_data,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            collate_fn=collate_fn,
        )

        # Load model
        logging.info("Loading model...")
        model = LipNet(
            vocab_size=len(vocab),
            input_channels=config["input_channels"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
        ).to(device)
        logging.info("Model loaded successfully!")

        # Create criterion and optimizer
        logging.info("Creating criterion and optimizer...")
        criterion = torch.nn.CTCLoss(zero_infinity=True, blank=0)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        logging.info("Criterion and optimizer created successfully!")

        # Train model
        logging.info("Training model...")
        loss_history = train_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config["num_epochs"],
            device=device,
            checkpoint_path=config["checkpoint_path"],
            prev_checkpoint=config["prev_checkpoint"],
            print_every=config["print_every"],
            new_lr=config["new_lr"],
        )
        logging.info("Model trained successfully!")
        logging.info(f"Loss history: {loss_history}")

        # Test model
        logging.info("Testing model...")
        test_loss, total_wer, total_cer = test(
            model, test_loader, device, criterion
        )
        
        mlflow.log_metrics({"test_loss": test_loss, "total_wer": total_wer, "total_cer": total_cer})
        
        logging.info(
            f"The model has a test loss of {test_loss}, a total WER of {total_wer}, and a total CER of {total_cer}"
        )
        
        # Tracing  model
        logging.info("Tracing model...")
        sampled_input = torch.rand(1, config["input_channels"], 75, 50, 100)
        traced_model = trace_model(config["checkpoint_path"]+"checkpoint_best.pt", sampled_input)
        traced_model_path = f"{config['model_name']}.pt"
        traced_model.save(traced_model_path)
        logging.info("Model traced successfully!")
        
        
        # Register model
        logging.info("Registering model...") 
        mlflow.pytorch.log_model(traced_model, artifact_path=config["model_name"])
        mlflow.register_model(f"runs:/{run.info.run_id}/{config['model_name']}", config["model_name"])
        logging.info("Model registered successfully!")
        
        logging.info("End of training!")
        


if __name__ == "__main__":
    main()
