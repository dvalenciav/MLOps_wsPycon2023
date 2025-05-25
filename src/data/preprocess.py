
import torch
import torchvision
from torch.utils.data import TensorDataset

# testing
import os
import argparse
import wandb

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

def preprocess(dataset, normalize=True, expand_dims=True):
    """
    Preprocess the dataset by normalizing and reshaping.
    """
    x, y = dataset.tensors

    if normalize:
        x = x.type(torch.float32) / 255.0  # Normalize to [0, 1]

    if expand_dims:
        x = torch.unsqueeze(x, 1)  # Shape: (N, 1, 28, 28)

    return TensorDataset(x, y)

def preprocess_and_log(steps):
    with wandb.init(project="MLOps-Pycon2023",
                    name=f"Preprocess Data ExecId-{args.IdExecution}",
                    job_type="preprocess-data") as run:
        
        processed_data = wandb.Artifact(
            "fashion-mnist-preprocessed", type="dataset",
            description="Preprocessed FashionMNIST dataset",
            metadata=steps)
        
        # Cargar el artefacto crudo
        raw_data_artifact = run.use_artifact("fashion-mnist-raw:latest")
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")

        # Procesar cada partición
        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_split = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_split.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)

# Pasos a ejecutar
steps = {"normalize": True, "expand_dims": True}  # <- usamos expand_dims porque es imagen
preprocess_and_log(steps)
