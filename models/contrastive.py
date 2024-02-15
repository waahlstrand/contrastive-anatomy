from typing import *
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
import torchmetrics
from sklearn.metrics import roc_auc_score, average_precision_score
from models.resnet import get_resnet, name_to_params
from rich import print
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
import lightning as L
from faiss import IndexFlatL2
import wandb

class SimSiamModule(nn.Module):

    def __init__(self,
                 dim: int = 1000,
                 prediction_dim: int = 512,
                 resnet_checkpoint: Optional[str] = None,
                 frozen: bool = False,
                 ):
        
        super().__init__()

        # Encoder
        if resnet_checkpoint is None:
            self.encoder = models.resnet50()
        else:
            print(f"Loading encoder from {resnet_checkpoint}...")
            self.encoder, _ = get_resnet(*name_to_params(resnet_checkpoint))
            self.encoder.load_state_dict(torch.load(resnet_checkpoint)["resnet"])
            print("Encoder loaded.")

        # Freeze 
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Adapter
        self.adapter = nn.Linear(self.encoder.fc.in_features, prediction_dim)

        # Predictor
        self.predictor = nn.Linear(prediction_dim, prediction_dim)

    def criterion(self, p: Tensor, z: Tensor) -> Tensor:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        # return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
        return -(p * z.detach()).sum(dim=-1).mean()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        # Encoder
        z = self.encoder(x)

        # Adapter
        z = self.adapter(z)

        # Predictor
        p = self.predictor(z)

        return p, z
    
class SimSiam(L.LightningModule):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 n_neighbours: int = 5,
                 target_size: Tuple[int, int] = (224, 224),
                 resnet_checkpoint: Optional[str] = None,
                 n_test_augmentations: int = 10,
                 frozen: bool = False):

        super().__init__()

        self.dim = dim
        self.prediction_dim = prediction_dim
        self.n_neighbours = n_neighbours
        self.target_size = target_size
        self.n_test_augmentations = n_test_augmentations


        # Model
        self.model = SimSiamModule(dim, prediction_dim, resnet_checkpoint, frozen)

        # Augmentation
        transforms = T.Compose([
            T.Resize(256),
            T.RandomCrop(target_size),
            T.RandomHorizontalFlip(),
            T.Normalize((0.5,), (0.5,))
        ])

        self.augmentation = lambda xs: torch.stack([transforms(x) for x in xs])

        # Index
        self.index = IndexFlatL2(prediction_dim)

        # Metrics
        self.metrics = torchmetrics.MetricCollection({
            "roc": torchmetrics.ROC(task="binary"),
        })

        self.trues = []
        self.preds = []

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(x)
    
    def training_step(self, batch: Tuple[Tensor, Tensor]) -> Mapping[str, Tensor]:
        
        x, y = batch

        # Augment
        x1 = self.augmentation(x)
        x2 = self.augmentation(x)

        # Forward
        p1, z1 = self.model(x1)
        p2, z2 = self.model(x2)

        # Loss
        loss = (self.model.criterion(p1, z2) + self.model.criterion(p2, z1)) / 2

        # Log
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Add to feature bank
        self.add_to_bank(x)

        return {"loss": loss, "x1": x1, "x2": x2}
    
    def augment_n_times(self, x: Tensor, n: int) -> List[Tensor]:
        xs = [self.augmentation(x) for _ in range(n)]
        xs = torch.cat(xs, dim=0)

        return xs
    
    @torch.no_grad()
    def add_to_bank(self, x: Tensor) -> None:
        
        # Augment
        xs = self.augment_n_times(x, self.n_test_augmentations)

        # Forward
        self.model.eval()
        ps, zs = self.model(xs)
        self.model.train()

        self.index.add(normalize(ps, dim=1).cpu().numpy())

    def score(self, p: Tensor) -> Tensor:

        p = normalize(p, dim=1)
        d, _ = self.index.search(p.cpu().numpy(), self.n_neighbours)

        # Get distance to nearest neighbour
        d = torch.from_numpy(d[:, -1])

        # Reshape to (batch_size, n_test_augmentations)
        d = d.reshape(-1, self.n_test_augmentations)

        # Get geometric mean distance
        score = d.prod(dim=1).pow(1 / self.n_test_augmentations)

        return score

    def validation_step(self, batch: Tuple[Tensor, Tensor]) ->  Mapping[str, Tensor] :
        
        x, y = batch

        # Get predictions
        xs = self.augment_n_times(x, self.n_test_augmentations)
        ps, zs = self.model(xs)

        # Get scores
        score = self.score(ps)

        self.trues.append(y.cpu().squeeze())
        self.preds.append(score.cpu().squeeze())

        return {"y": y, "score": score}
    
    def on_validation_epoch_end(self) -> None:
        
        y     = torch.cat(self.trues).cpu().numpy()
        score = torch.cat(self.preds).cpu().numpy()

        auroc   = roc_auc_score(y, score)
        ap      = average_precision_score(y, score)

        # Log
        self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ap", ap, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save ROC curve
        self.metrics["roc"].update(score, y)
        self.metrics["roc"].compute()
        f, ax = self.metrics["roc"].plot()
        self.logger.experiment.log({"roc": wandb.Image(f)})
        self.metrics["roc"].reset()

        # Reset
        self.trues = []
        self.preds = []
        self.index = IndexFlatL2(self.prediction_dim)




