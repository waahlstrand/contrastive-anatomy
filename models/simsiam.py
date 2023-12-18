from typing import Any, Tuple
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

from .base import Contrastive

class SimSiamModel(nn.Module):

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
        self.channel_projection = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        self.encoder = models.resnet50(pretrained=False)
        previous_dim = self.encoder.fc.in_features

        # Update the3 encoder's fc layer to output the desired dimension
        self.encoder.fc = nn.Sequential(
            nn.Linear(previous_dim, previous_dim, bias=False),
            nn.BatchNorm1d(previous_dim),
            nn.ReLU(inplace=True),
            nn.Linear(previous_dim, previous_dim, bias=False),
            nn.BatchNorm1d(previous_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )
        self.encoder.fc[6].bias.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Linear(dim, prediction_dim, bias=False), 
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(prediction_dim, dim)
            )
        
        self.augmentation = nn.Sequential(nn.Identity())
        
    def augment(self, x: Tensor) -> Tensor:
            
        return self.augmentation(x)

    def encode(self, x: Tensor) -> Tensor:

        return self.encoder(x)
    
    def predict(self, x: Tensor) -> Tensor:

        return self.predictor(x)

    def forward(self, x: Tensor) -> Tensor:
        
        x = self.channel_projection(x)
        x = self.augment(x)
        z = self.encode(x)
        p = self.predict(z)

        return p, z
    
    def negative_cosine_similarity(self, p: Tensor, z: Tensor) -> Tensor:

        z = z.detach()

        p = p.norm(p=2, dim=1, keepdim=True)
        z = z.norm(p=2, dim=1, keepdim=True)

        loss = -(p * z).sum(dim=1).mean()

        return loss
    
    def criterion(self, x: Tuple[Tensor, Tensor], y: Tuple[Tensor, Tensor]) -> Tensor:
        p1, z1 = x
        p2, z2 = y

        loss = 0.5 * (self.negative_cosine_similarity(p1, z2) + self.negative_cosine_similarity(p2, z1))

        return loss

class SimSiam(Contrastive):

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

        self.model = SimSiamModel(
            dim, 
            prediction_dim, 
            lr, 
            momentum, 
            weight_decay, 
            n_epochs, 
            **kwargs)
        
        self.criterion = self.model.criterion

    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tensor:

        x1, x2 = batch

        p1, z1 = self(x1)
        p2, z2 = self(x2)

        # Compute the standard deviation of the norm of the predictions
        # this should be around 1/sqrt(dim) for varied predictions
        # and around 0 for constant predictions
        std = p1.norm(dim=1).std(dim=0).mean()
        self.log(f"{name}_std", std)

        loss = self.criterion((p1, z1), (p2, z2)).mean()

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        loss = self.step(batch, "train_stage")

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        loss = self.step(batch, "val_stage")

        self.log("val_loss", loss)

        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        loss = self.step(batch, "test_stage")

        self.log("test_loss", loss)

        return loss

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        return self.model(x)
    
    def configure_metrics(self) -> None:

        pass

class AnatomicSimSiam(Contrastive):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 lr: float = 0.05, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-6, 
                 n_epochs: int = 100, **kwargs):

        super().__init__(dim, prediction_dim, lr, momentum, weight_decay, n_epochs, **kwargs)

        self.model = SimSiamModel(
            dim, 
            prediction_dim, 
            lr, 
            momentum, 
            weight_decay, 
            n_epochs, 
            **kwargs)
        
        self.criterion = self.model.criterion

    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tensor:

        x1, x2 = batch

        p1, z1 = self(x1)
        p2, z2 = self(x2)

        # Compute the standard deviation of the norm of the predictions
        # this should be around 1/sqrt(dim) for varied predictions
        # and around 0 for constant predictions
        std = p1.norm(dim=1).std(dim=0).mean()
        self.log(f"{name}_std", std)

        loss = self.criterion((p1, z1), (p2, z2)).mean()

        return loss

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        return self.model(x)
    
    def configure_metrics(self) -> None:

        pass

