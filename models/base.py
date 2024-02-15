from typing import *
import lightning as L
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms as T
from typing import *
import torchmetrics
from pathlib import Path
import faiss
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.nn.functional import normalize

class Index:

    def __init__(self, latents: Tensor, dimensionality: int, eps: float = 1e-10) -> "Index":
        
        self.latents = latents
        self.eps = eps
        self.dimensionality = dimensionality
        self.index = self.build()


    def build(self) -> faiss.IndexFlatL2:

        # Normalize latents
        train_latents = normalize(self.latents, dim=1).cpu().numpy()

        # Create index
        index = faiss.IndexFlatL2(self.dimensionality)
        index.add(train_latents)

        return index
    
    def __call__(self, z: np.ndarray, n_neighbours: int) -> Tuple[Tensor, Tensor]:

        z = z.view(-1, self.dimensionality)
        z = normalize(z, dim=1).cpu().numpy()
        
        d, _ = self.index.search(z, n_neighbours)

        return torch.from_numpy(d), torch.from_numpy(_)
    
class Contrastive(L.LightningModule):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 n_neighbours: int = 5,
                 target_size: Tuple[int, int] = (224, 224),
                 **kwargs):

        super().__init__()
        
        self.dim = dim
        self.prediction_dim = prediction_dim
        self.n_neighbours = n_neighbours
        self.test_true = []
        self.test_pred = []
        self.latents   = []

        self.save_hyperparameters()
        self.criterion: Callable[[Tensor, Tensor], Tensor] = None

        self.index: Index = None
        self.resize = T.Resize(target_size)

        self.metrics = torchmetrics.MetricCollection({
            "auroc": torchmetrics.AUROC(task="binary"),
            "ap": torchmetrics.AveragePrecision(task="binary"),
            "roc": torchmetrics.ROC(task="binary"),
        })

    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tuple[Tensor, Tensor, Tensor]:

        raise NotImplementedError
    
    def test_features(self, x: Tensor) -> Tensor:

        raise NotImplementedError
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="train_stage") -> Dict[str, Tensor]:

        loss, x1, x2 = self.step(batch, name)

        self.eval()
        ps = self.test_features(batch[0].detach())
        self.train()
        self.latents.append(ps.cpu())

        self.log(f"{name}/loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "x1": x1.detach(),
            "x2": x2.detach()
        }
    
    def on_train_epoch_start(self) -> None:
        
        self.latents = []
    
    def on_validation_epoch_start(self) -> None:
        
        self.latents = torch.cat(self.latents, dim=0) if len(self.latents) > 0 else torch.zeros((0, self.prediction_dim))
        self.index   = Index(self.latents, self.prediction_dim)
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="val_stage") -> Dict[str, Tensor]:
        
        return self.test_step(batch, batch_idx, 0)
    
    def on_validation_epoch_end(self) -> None:
        return self.on_test_epoch_end()
    
    def on_test_epoch_end(self) -> None:
        
        self.test_true = torch.cat(self.test_true, dim=0)
        self.test_pred = torch.cat(self.test_pred, dim=0)

        self.metrics(self.test_pred, self.test_true)

        for name, metric in self.metrics.items():
            if name != "roc":
                vals = metric.compute()
                self.log(name, vals, on_step=False, on_epoch=True)
            else:
                metric.compute()
                f, ax = metric.plot()
                self.logger.experiment.log({name: wandb.Image(f)})

        self.test_true = []
        self.test_pred = []
    
    def __call__(self, *args: Any, **kwds: Any) -> Tuple[Tensor, Tensor]:
        return super().__call__(*args, **kwds)