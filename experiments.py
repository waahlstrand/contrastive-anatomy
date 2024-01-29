import wandb
from train_contrastive import train
import yaml
from rich import print

def sweep():

    wandb.init(project="contrastive")

    train(wandb.config)



if __name__ == "__main__":
    
    with open("configs/base.yml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters = {k : {"values": [v]} for k, v in parameters.items()}

    parameters.update(**{
        "model": {"values": ["anatomic_simsiam", "simsiam"]},
        "pretrained": {"values": [True, False]},
        "n_patches_per_side": {"values": [1, 2, 3, 4, 5]}
        })
  
    sweep_configuration = {
        "method": "grid",
        "name": "contrastive",
        "metric": {"goal": "minimize", "name": "val_stage_loss"},
        "parameters": parameters,
    }

  
    sweep_id = wandb.sweep(sweep_configuration, project="contrastive")
 
    wandb.agent(sweep_id, function=sweep)

