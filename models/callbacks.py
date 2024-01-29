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

class Index:

    def __init__(self, latents: Tensor, batch_size: int, eps: float = 1e-10) -> "Index":
        
        self.latents = latents
        self.batch_size = batch_size
        self.eps = eps

        self.index = self.build()

    def normalize(self, latents: Tensor | np.array) -> np.array:

        latents = latents.cpu().numpy()

        latents /= (np.linalg.norm(latents, axis=-1, keepdims=True) + self.eps)

        latents = np.ascontiguousarray(latents)

        return latents

    def build(self) -> faiss.IndexFlatL2:

        # Reshape to (batch_size, n_patches*dim)
        # self.latents    = self.latents.view(self.batch_size, -1)
        dimensionality  = self.latents.shape[-1]

        # Normalize latents
        train_latents = self.normalize(self.latents)

        # Create index
        index = faiss.IndexFlatL2(dimensionality)
        index.add(train_latents)

        return index
    
    def __call__(self, z: np.ndarray, n_neighbours: int) -> Tuple[np.ndarray, np.ndarray]:

        # z = z.view(-1, self.latents.shape[-1])
        z = self.normalize(z)

        return self.index.search(z, n_neighbours)


class SaveLatentsCallback(L.Callback):

    def __init__(self, log_dir: Path | str) -> None:
        super().__init__()

        self.log_dir = Path(log_dir)

    def on_fit_end(self, trainer: L.Trainer, module: AnatomicSimSiam | SimSiam) -> None:
        
        # Get model
        model = module.model.eval()

        # Move to device
        model = model.cuda(trainer.device_ids[0])
        
        # Get training data
        train_loader = trainer.train_dataloader
        
        latents = []
        for batch in train_loader:
            
            x, _ = batch

            x = x.cuda(trainer.device_ids[0])

            with torch.no_grad():
                p, z = model(x)

                p = p.view(module.batch_size, -1) # (batch_size, n_patches*dim)
            
            latents.append(p)

        latents = torch.cat(latents, dim=0)

        # Create index
        module.index = Index(latents, module.batch_size)

        torch.save(latents, self.log_dir / "latents.pt")



