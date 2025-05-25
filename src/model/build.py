import torch
from src.ConvNet import ConvNet  # ahora importamos el nuevo modelo

import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Crea el directorio del modelo si no existe
if not os.path.exists("./model"):
    os.makedirs("./model")

# Parámetros del modelo
model_config = {
    "input_shape": (1, 28, 28),  # Para info, no se usa directamente
    "num_classes": 10,
    "architecture": "ConvNet"
}

# Construir el modelo
model = ConvNet(num_classes=model_config["num_classes"])

def build_model_and_log(config, model, model_name="ConvNet", model_description="Simple CNN"):
    with wandb.init(project="MLOps-Pycon2023", 
                    name=f"Initialize Model ExecId-{args.IdExecution}", 
                    job_type="initialize-model", config=config) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))

        name_artifact_model = f"initialized_model_{model_name}.pth"
        torch.save(model.state_dict(), f"./model/{name_artifact_model}")
        model_artifact.add_file(f"./model/{name_artifact_model}")

        wandb.save(name_artifact_model)
        run.log_artifact(model_artifact)

build_model_and_log(model_config, model)
