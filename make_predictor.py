import torch
import torch.nn as tnn
import argparse

import os
from pathlib import Path

from transformers import AutoConfig

class ReluMLP(tnn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ReluMLP, self).__init__()
        self.fc1 = tnn.Linear(input_dim, hidden_dim, bias=False)
        self.relu = tnn.ReLU()
        self.fc2 = tnn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def from_file(model_file: Path):
        model = torch.load(model_file, map_location="cpu")
        hidden_size, input_size = model.get("fc1.weight").shape
        output_size, _ = model.get("fc2.weight").shape
        mlp = ReluMLP(input_size, hidden_size, output_size)
        mlp.load_state_dict(model)
        return mlp


def parse_args():
    parser = argparse.ArgumentParser(
        description='generate fake MLP sparsity predictor'
    )
    parser.add_argument('--name', type=str, required=True, help='huggingface model name')
    parser.add_argument('--outdir', type=Path, required=True, help='predictor output path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = AutoConfig.from_pretrained(args.name)
    args.outdir.mkdir(exist_ok=True)

    hidden_dim = config.hidden_size
    n_layers = config.num_hidden_layers
    try:
        ffn_dim = config.intermediate_size
    except AttributeError:
        ffn_dim = config.ffn_dim
    
    pred = ReluMLP(hidden_dim, hidden_dim // 4, ffn_dim)
    for layer in range(n_layers):
        path = args.outdir / f'model_{layer}.pt'
        with open(path, 'wb') as f:
            torch.save(pred.state_dict(), f)
