from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from torch import nn
from models.simsiam import AnatomicSimSiam, SimSiam
from pathlib import Path
from typing import *
import numpy as np
from models.base import Index
import matplotlib.pyplot as plt
from rich import print
import wandb

class SaveLatentsCallback(L.Callback):

    def __init__(self,  batch_size: int, dim: int, log_dir: Path, latents_path: Union[Path, None]) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.dim = dim
        self.log_dir = log_dir
        self.latents_path = latents_path

    def on_test_start(self, trainer: L.Trainer, module: AnatomicSimSiam | SimSiam) -> None:

        if self.latents_path is not None:
            latents_path = self.latents_path
        
        elif self.latents_path is None and (self.log_dir / "latents.pt").exists():
            latents_path = self.log_dir / "latents.pt"

        else:
            raise FileNotFoundError("Index not found. Please run training first.")

        print(f"Loading latents from {latents_path}...")
        latents = torch.load(latents_path)
        module.index = Index(latents, module.prediction_dim)

    def on_fit_end(self, trainer: L.Trainer, module: AnatomicSimSiam | SimSiam) -> None:
        
        # Get model
        model = module.eval()
        module = module.to(trainer.device_ids[0])

        # Log dir
        self.log_dir = Path(trainer.logger.experiment.dir) if isinstance(trainer.logger.experiment.dir, str) else Path(self.log_dir)
        
        # Get training data
        train_loader = trainer.train_dataloader

        print("Saving latents[green]...[/green]", ":floppy_disk:")
        latents = []
        for i, batch in enumerate(train_loader):
                
            x, _ = batch
            x = x.to(module.device)

            with torch.no_grad():

                ps = module.test_features(x)
                
            latents.append(ps)

        latents = torch.cat(latents, dim=0)

        torch.save(latents, self.log_dir / "latents.pt")
        print(f"Saved latents to {self.log_dir / 'latents.pt'}.")

        # Create index
        print("Building index[green]...[/green]", ":hammer_and_wrench:")
        module.index = Index(latents, module.prediction_dim)



class PlotPairsCallback(L.Callback):

    def __init__(self, n_samples: int) -> None:
        super().__init__()

        assert n_samples % 2 == 0, "n_samples must be even."
        self.n_samples = n_samples

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:

        loss, x1, x2 = outputs["loss"], outputs["x1"], outputs["x2"]

        # Get a number of samples
        idx = np.random.choice(len(x1), self.n_samples, replace=False)

        x1_samples = x1[idx]
        x2_samples = x2[idx]

        # Plot in a grid
        if batch_idx % 500 == 0:
            fig, ax = plt.subplots(2, self.n_samples // 2, figsize=(20, 10))
            for i in range(self.n_samples // 2):
                ax[0, i].imshow(x1_samples[i][0].cpu().numpy(), cmap="gray")
                ax[1, i].imshow(x2_samples[i][0].cpu().numpy(), cmap="gray")

            trainer.logger.experiment.log({"train_stage/pairs": wandb.Image(fig)})


        


