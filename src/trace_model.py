import os
import argparse
import torch
from .models import LipNet
from . import constants

def load_model(checkpoint_path):
    model = LipNet(vocab_size=len(constants.vocab), hidden_size=256, input_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def trace_model(checkpoint_path, example_input=torch.rand(1, 3, 75, 50, 100)):
    model = load_model(checkpoint_path)
    traced_model = torch.jit.trace(model, example_input)
    return traced_model

def save_traced_model(traced_model, output_path):
    traced_model.save(output_path)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Trace the LipNet model")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--output_path", type=str, help="Path to save the traced model")
    return parser.parse_args()

def main():
    args = parse_args()
    traced_model = trace_model(args.checkpoint_path)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_traced_model(traced_model, args.output_path)
    
if __name__ == "__main__":
    main()