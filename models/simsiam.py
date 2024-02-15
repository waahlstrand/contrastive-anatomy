from typing import Any, Dict, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms as T
from typing import *
import torchmetrics
import traceback
from data.rsna import RSNAStatistics
from .base import Contrastive
from models.patchwise import Patchify
from sklearn.metrics import roc_auc_score, average_precision_score
from models.resnet import get_resnet, name_to_params
from rich import print
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

class SimSiamModel(nn.Module):

    def __init__(self, 
                 dim: int = 2048, 
                 prediction_dim: int = 1024, 
                 encoder_name: str = "resnet50",
                 frozen: bool = False,
                 resnet_checkpoint: str = None,
                **kwargs):

        super().__init__()
        
        self.dim = dim

        self.encoder_name = encoder_name
        self.channel_projection = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

        if resnet_checkpoint is None:
            self.encoder = models.resnet50()
        else:
            print(f"Loading encoder from {resnet_checkpoint}...")
            self.encoder, _ = get_resnet(*name_to_params(resnet_checkpoint))
            self.encoder.load_state_dict(torch.load(resnet_checkpoint)["resnet"])
            print("Encoder loaded.")
   
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.adapter    = nn.Linear(dim, prediction_dim)
        self.predictor  = nn.Linear(prediction_dim, prediction_dim)

    def forward(self, x: Tensor) -> Tensor:
        
        x = self.channel_projection(x)
        z = self.encoder(x)
        z = self.adapter(z)
        p = self.predictor(z)

        return p, z.detach()
    
    def negative_cosine_similarity(self, p: Tensor, z: Tensor) -> Tensor:
        """
        Compute the negative cosine similarity between unnormalized vectors p and z.
        
        Args:
            p (Tensor): The first vector.
            z (Tensor): The second vector.
        
        Returns:
            Tensor: The negative cosine similarity.
        """
        z = z.detach()
        p = normalize(p, dim=1)
        z = normalize(z, dim=1)

        loss = -(p*z).sum(dim=1).mean()

        return loss
    
    def criterion(self, p1: Tensor, z1: Tensor, p2: Tensor, z2: Tensor) -> Tensor:

        loss = 0.5 * (self.negative_cosine_similarity(p1, z2) + self.negative_cosine_similarity(p2, z1))

        return loss

class SimSiam(Contrastive):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512,  
                 size: Tuple[int, int] = (256, 256),
                 target_size: Tuple[int, int] = (224, 224),
                 n_test_augmentations: int = 5,
                 n_patches_per_side: int = None,
                 frozen: bool = False,
                 resnet_checkpoint: str = None,
                n_neighbours: int = 5,
                 **kwargs):

        super().__init__(
            dim=dim, 
            prediction_dim=prediction_dim, 
            size=size,
            target_size=target_size,
            **kwargs)
        
        self.prediction_dim = prediction_dim
        self.dim = dim

        self.size = size
        self.target_size = target_size
        self.n_neighbours = n_neighbours
        self.n_test_augmentations = n_test_augmentations
        self.n_patches_per_side = n_patches_per_side

        self.augmentation = T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # T.RandomResizedCrop(target_size, scale=(0.05, 1.0)),
            # self.resize,
            T.Resize(256),
            T.RandomCrop(target_size),
            T.RandomHorizontalFlip(),
            T.Normalize((0.5,), (0.5,))
        ])

        self.save_hyperparameters()

        self.model = SimSiamModel(
            dim, 
            prediction_dim, 
            frozen=frozen,
            resnet_checkpoint=resnet_checkpoint,
            )
        
        self.criterion = self.model.criterion

    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tuple[Tensor, Tensor, Tensor]:

        # For SimSiam, return two copies of the same image
        x, y = batch

        # Augment separately
        x1, x2 = self.augmentation(x), self.augmentation(x)

        p1, z1 = self(x1)
        p2, z2 = self(x2)

        # Compute the standard deviation of the norm of the predictions
        # this should be around 1/sqrt(dim) for varied predictions
        # and around 0 for constant predictions
        # std = p1.detach().norm(dim=1).std(dim=0).mean()
        # self.log(f"{name}/std", std, on_step=False, on_epoch=True, prog_bar=True)

        loss = self.criterion(p1, z1, p2, z2)

        return loss, x1, x2

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        return self.model(x)
    
    @torch.no_grad()
    def test_features(self, batch: Tensor) -> Tensor:
        
        # Augment separately
        ps = []
        for i in range(self.n_test_augmentations):
            x = self.augmentation(batch)
            p, _ = self(x)
            ps.append(p)

        ps = torch.stack(ps, dim=0).view(-1, self.prediction_dim) # (n_augmentations*batch_size, dim)

        # xs = [self.augmentation(x) for _ in range(self.n_test_augmentations)]
        # xs = torch.stack(xs, dim=0).view(-1, *(1, *self.target_size)) # (n_augmentations*batch_size, 1, size, size)

        # # Get the predictions
        # ps, zs = self(xs)
        # ps     = ps.view(-1, self.prediction_dim) # (n_augmentations*batch_size, dim)

        return ps

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:

        x, y = batch

        ps = self.test_features(x)

        # Compute the distances through KNN
        d, _ = self.index(ps, self.n_neighbours) # (n_augmentations*batch_size, n_neighbours)
        d    = d.view(-1, self.n_test_augmentations, self.n_neighbours).to(x.device)
        d    = d[:,:,-1].view(self.n_test_augmentations, -1)

        # Geometric mean over augmentations
        y_score = d.prod(dim=0).pow(1/self.n_test_augmentations) # (batch_size)

        # Get the labels
        y_true = y.squeeze() # (batch_size)

        # Compute metrics using sklearn
        self.test_true.append(y_true)
        self.test_pred.append(y_score)

        return y_score, y_true
    
class AnatomicSimSiam(Contrastive):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 size: Tuple[int, int] = (224, 224),
                 target_size: Tuple[int, int] = (224, 224),
                 n_patches_per_side: int = 4,
                 frozen: bool = False,
                 resnet_checkpoint: str = None,
                 n_neighbours: int = 5,
                 n_test_augmentations: int = 5,
                 **kwargs):

        super().__init__(
            dim=dim, 
            prediction_dim=prediction_dim, 
            size=size,
            target_size=target_size,
            **kwargs)

        model = SimSiamModel(
            dim, 
            prediction_dim, 
            frozen=frozen,
            resnet_checkpoint=resnet_checkpoint,
            )
        
        # self.model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.criterion = self.model.criterion
        self.prediction_dim = prediction_dim
        self.dim = dim
        self.n_neighbours = n_neighbours
        self.size = size
        self.target_size = target_size
        self.n_patches_per_side = n_patches_per_side
        self.n_test_augmentations = n_test_augmentations
        # self.index_dimensionality = prediction_dim * (n_patches_per_side ** 2)
        self.index_dimensionality = prediction_dim * n_test_augmentations


        self.patchify = Patchify.from_n_patches(n_patches_per_side, image_size=(RSNAStatistics.WIDTH, RSNAStatistics.HEIGHT))

        self.augmentation = T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.Resize(256),
            T.RandomCrop(target_size),
            T.RandomHorizontalFlip(),
            T.Normalize((0.5,), (0.5,)),
        ])


    def step(self, batch: Tuple[Tensor, Tensor], name: str) -> Tuple[Tensor, Tensor, Tensor]:
        
        x, y = batch

        # Create patches
        # x1 = self.augmentation(self.patchify(x)) # (batch_size*n_patches, 1, patch_size, patch_size)
        x1 = self.augmentation(self.patchify(x)) # (batch_size*n_patches, 1, patch_size, patch_size)
        
        # Randomly shuffle the batch indices of x2
        indices = torch.randperm(x.shape[0])
        x2 = x.clone()[indices]

        x2 = self.augmentation(self.patchify(x2)) # (batch_size*n_patches, 1, patch_size, patch_size)
        print(x1.shape, x2.shape)
        p1, z1 = self(x1)
        p2, z2 = self(x2)

        # Compute the standard deviation of the norm of the predictions
        # this should be around 1/sqrt(dim) for varied predictions
        # and around 0 for constant predictions
        # std = p1.detach().norm(dim=1).std(dim=0).mean()
        # self.log(f"{name}/std", std, on_step=False, on_epoch=True, prog_bar=True)

        loss = self.criterion((p1, z1), (p2, z2))

        return loss, x1, x2
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        return self.model(x)
    

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tuple[Tensor, Tensor]:

        x, y = batch
        n_augmentations = self.n_test_augmentations if self.n_test_augmentations > 1 else 1

        ps = self.test_features(x)

        # Compute the distances through KNN
        # d, _ = self.index(ps, self.n_neighbours) # (n_patches*batch_size, n_neighbours)
        # d    = d.view(-1, self.n_patches_per_side ** 2, self.n_neighbours).to(x.device)
        # d    = d[:,:,-1].view(self.n_patches_per_side ** 2, -1) # (n_patches, batch_size)

        # # Geometric mean over augmentations
        # y_score = d.prod(dim=0).pow(1/self.n_patches_per_side ** 2) # (batch_size)

        # Compute the distances through KNN (SIMSIAM)
        d, _ = self.index(ps, self.n_neighbours) # (n_augmentations*batch_size*n_patches, n_neighbours)
        d    = d.view(-1, n_augmentations, self.n_patches_per_side ** 2, self.n_neighbours).to(x.device)
        d    = d[:,:,:,-1] # Select the last neighbour

        # Geometric mean over augmentations
        y_score = d.prod(dim=1).pow(1/n_augmentations).max(dim=1).values # (batch_size*n_patches)
        # y_score = y_score.view(-1, self.n_patches_per_side ** 2).mean(dim=1) # (batch_size) (added)
        # y_score = y_score.view(-1, self.n_patches_per_side ** 2)[:,:2].max(dim=1).values #.prod(dim=1).pow(1/(self.n_patches_per_side ** 2)) # (batch_size) (added)
        # y_score = d.view(-1, self.n_test_augmentations, self.n_patches_per_side ** 2).max(1).values.max(1).values

        # Get the labels
        y_true = y.squeeze() # (batch_size)

        self.test_true.append(y_true)
        self.test_pred.append(y_score)

        return y_score, y_true
    
    @torch.no_grad()
    def test_features(self, x: Tensor) -> Tensor:

        # Create patches
        xs = self.patchify(x)

        # Augment separately (SIMSIAM)
        if self.n_test_augmentations > 1:
            xs = [self.augmentation(xs) for _ in range(self.n_test_augmentations)]
            xs = torch.stack(xs, dim=0).view(-1, *(1, *self.target_size))

        # Get the predictions
        ps, zs = self(xs)
        ps     = ps.view(-1, self.prediction_dim) # (n_augmentations*batch_size*n_patches, dim)
        return ps
