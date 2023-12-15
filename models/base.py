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


class Contrastive(L.LightningModule):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 lr: float = 0.05, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-6, 
                 n_epochs: int = 100, **kwargs):

        super().__init__()
        
        self.dim = dim
        self.init_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

        self.save_hyperparameters()
        self.criterion: Callable[[Tensor, Tensor], Tensor] = None

    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tensor:

        x1, x2 = batch

        x1 = self(x1)
        x2 = self(x2)

        loss = self.criterion(x1, x2).mean()

        return loss
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="train_stage") -> Tensor:

        loss = self.step(batch, name)

        self.log(f"{name}_loss", loss)

        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="val_stage") -> Tensor:

        loss = self.step(batch, name)

        self.log(f"{name}_loss", loss)

        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, name="test_stage") -> Tensor:

        loss = self.step(batch, name)

        self.log(f"{name}_loss", loss)

        return loss


    def configure_optimizers(self) -> Any:
        
        optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def __call__(self, *args: Any, **kwds: Any) -> Tuple[Tensor, Tensor]:
        return super().__call__(*args, **kwds)