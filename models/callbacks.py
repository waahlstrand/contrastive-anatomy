from typing import Any
import lightning as L
import torch
from torch import Tensor
from torch import nn
from models.simsiam import AnatomicSimSiam, SimSiam
from pathlib import Path
from typing import *
import numpy as np
import faiss
from models.base import Index


class SaveLatentsCallback(L.Callback):

    def __init__(self, n_patches_per_side: int, batch_size: int, dim: int, log_dir: Path | str) -> None:
        super().__init__()

        self.n_patches_per_side = n_patches_per_side
        self.batch_size = batch_size
        self.dim = dim
        self.log_dir = Path(log_dir)

    def on_test_start(self, trainer: L.Trainer, module: AnatomicSimSiam | SimSiam) -> None:

        # Check if index exists
        if not (self.log_dir / "latents.pt").exists():
            raise FileNotFoundError("Index not found. Please run training first.")
        
        # Load index
        latents = torch.load(self.log_dir / "latents.pt")
        module.index = Index(latents, self.n_patches_per_side)

    def on_fit_end(self, trainer: L.Trainer, module: AnatomicSimSiam | SimSiam) -> None:
        
        # Get model
        model = module.model.eval()

        # Move to device
        model = model.cuda(trainer.device_ids[0])
        
        # Get training data
        train_loader = trainer.train_dataloader
        
        if not (self.log_dir / "latents.pt").exists():
            
            latents = []
            for i, batch in enumerate(train_loader):
                
                x, _ = batch

                x = x.cuda(trainer.device_ids[0])

                with torch.no_grad():
                    p, z = model(x)

                    p = p.view(-1, self.dim) # (batch_size*n_patches, dim)
                
                latents.append(p)

            latents = torch.cat(latents, dim=0)

            torch.save(latents, self.log_dir / "latents.pt")

        else:
            latents = torch.load(self.log_dir / "latents.pt")

        # Create index
        module.index = Index(latents, self.n_patches_per_side)




