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

class Contrastive(L.LightningModule):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 batch_size: int = 32,
                 lr: float = 0.05, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-6, 
                 n_neighbours: int = 5,
                 n_epochs: int = 100, **kwargs):

        super().__init__()
        
        self.dim = dim
        self.init_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_neighbours = n_neighbours

        self.save_hyperparameters()
        self.criterion: Callable[[Tensor, Tensor], Tensor] = None

        self.index: Callable = None

        self.metrics = nn.ModuleDict({
            'test_stage': torchmetrics.MetricCollection({
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "auroc": torchmetrics.AUROC(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
            })
        })

    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tensor:

        raise NotImplementedError
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="train_stage") -> Tensor:

        loss = self.step(batch, name)

        self.log(f"{name}_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="val_stage") -> Tensor:

        loss = self.step(batch, name)

        self.log(f"{name}_loss", loss.detach(), on_step=True, on_epoch=True)

        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="test_stage") -> Tensor:

        assert self.index is not None, "Index is not initialized"

        patches, labels = batch # (batch_size*n_patches, 1, patch_size, patch_size), (batch_size*n_patches,)
        p, z = self(patches)

        d, _ = self.index(p, self.n_neighbours)

        d = torch.from_numpy(d).view(self.batch_size, -1, self.n_neighbours)

        y_score = d.prod(dim=1).pow(1 / d.shape[1])

        y_true = labels.view(self.batch_size, -1)

        for metric in self.metrics[name]:
            ms = self.metrics[name][metric](y_score, y_true)
            self.log(f"{name}_{metric}", ms, on_step=False, on_epoch=True)



    def configure_optimizers(self) -> Any:
        
        optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler
        }
    
    def __call__(self, *args: Any, **kwds: Any) -> Tuple[Tensor, Tensor]:
        return super().__call__(*args, **kwds)