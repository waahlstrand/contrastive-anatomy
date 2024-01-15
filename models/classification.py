from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import Tensor
import torch.nn as nn
from typing import *
from .simsiam import SimSiam, AnatomicSimSiam, Contrastive
from torchmetrics import MetricCollection, Accuracy, Precision, AUROC

class ImageClassifier(L.LightningModule):

    def __init__(self, n_classes: int, dim: int) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.dim = dim

        self.save_hyperparameters()

        self.head = nn.Sequential(
            nn.Linear(dim, n_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

        self.stages = ["train_stage", "val_stage", "test_stage"]

        self.metrics = nn.ModuleDict({
            stage: self.build_metrics()
            for stage in self.stages
        }).to(self.device)

    def forward(self, x: Tensor) -> Tensor:

        raise NotImplementedError
    
    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tensor:

        raise NotImplementedError
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        return self.step(batch, "train_stage")
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        return self.step(batch, "val_stage")
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
            
        return self.step(batch, "test_stage")

    
    def build_metrics(self) -> MetricCollection:
        # task = "binary" if self.n_classes == 2 else "multiclass"
        task = "multiclass"
        metrics = MetricCollection({
            "accuracy": Accuracy(task=task, num_classes=self.n_classes),
            "precision": Precision(task=task, num_classes=self.n_classes, average="macro"),
            "auroc": AUROC(task=task, num_classes=self.n_classes),
        })

        return metrics
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.n_epochs)

        return [optimizer], []

class PatchwiseClassifier(ImageClassifier):

    def __init__(self, 
                 n_classes: int, 
                 dim: int,
                 batch_size: int,
                 n_patches_per_side: int = 2,
                 feature_extractor: Contrastive = None,
                 **kwargs) -> None:
        
        super().__init__(n_classes, 2*n_patches_per_side*dim)
        
        self.n_patches_per_side = n_patches_per_side
        self.n_patches = 2*self.n_patches_per_side
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.to(self.feature_extractor.device)

    def forward(self, x: Tensor) -> Tensor:
        
        with torch.no_grad():
            p, z = self.feature_extractor(x) # (B*n_patches, dim)

        p = p.reshape(self.batch_size, -1)
        x = self.head(p)
        
        return x
    
    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tensor:
        
        x, y = batch

        y_hat_logit = self(x)

        # Compute loss and log
        loss = self.criterion(y_hat_logit, y)

        self.log(f"{name}_loss", loss, prog_bar=True)

        # Compute metrics
        ms = self.metrics[name](y_hat_logit, y)

        self.log_dict({f"{name}_{k}": v for k, v in ms.items()})

        return loss
