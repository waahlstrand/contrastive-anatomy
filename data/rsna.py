from pathlib import Path
import pydicom
from typing import *
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms as T
import lightning as L
import pandas as pd
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from argparse import Namespace

# To plot rectangles
from matplotlib.patches import Rectangle

@dataclass
class RSNAItem:
    """
    Represents a single image with its label and bounding box, if any.
    """

    id: str
    path: Path | str
    image: Tensor
    label: Tensor
    bbox: Optional[Tensor | None]

    @classmethod
    def from_dict(cls, root: Path, d: Dict[str, Any]) -> "RSNAItem":
        """
        Create an RSNAItem from a dictionary formatted according to the official json file, 
        like e.g.:

        {
            "StudyInstanceUID": "1.2.276.0.7230010.3.1.4.8323329.32643.1517875195.678631",
            "SeriesInstanceUID": "1.2.276.0.7230010.3.1.3.8323329.32643.1517875195.678630",
            "SOPInstanceUID": "1.2.276.0.7230010.3.1.2.8323329.32643.1517875195.678629",
            "x": 0.143,
            "y": 0.143,
            "width": 0.714,
            "height": 0.714,
            "annotationNumber": 1

        }

        Args:
            root (Path): The root path to the dataset
            d (Dict[str, Any]): The dictionary from the labels file

        Returns:
            RSNAItem: The RSNAItem
        """

        try:
            # file = root / Path(d["StudyInstanceUID"]) / Path(d["SeriesInstanceUID"]) / Path(d["SOPInstanceUID"] + ".dcm")
            file = list((root / Path(d["StudyInstanceUID"])).rglob("*.dcm"))[0]

        except Exception as e:

            print(d)

        id     = file.stem
        path   = file
        image  = torch.from_numpy(pydicom.dcmread(file).pixel_array)
        label  = torch.tensor(d["annotationNumber"] or 0)
        annotation = d["data"] if d["data"] else {}
        if "x" in annotation and "y" in annotation and "width" in annotation and "height" in annotation:
            bbox = torch.tensor([annotation["x"], annotation["y"], annotation["width"], annotation["height"]])
        else:
            bbox = None


        return cls(id, path, image, label, bbox)

    def to(self, device: torch.device) -> "RSNAItem":
        
        self.image = self.image.to(device)
        self.label = self.label.to(device)
        self.bbox = self.bbox.to(device) if self.bbox else None

        return self
    
    def plot(self) -> None:

        f, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(self.image.squeeze(), cmap="gray")
        ax.set_title(f"Label: {self.label}")

        if self.bbox is not None:
            # Plot a rectangle
            x, y, w, h = self.bbox

            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)


        return f, ax
    

class RSNA(Dataset):

    """
    Represents the RSNA dataset, found at https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

    Attributes:
        root (Path): The root path to the dataset
        labels_path (Path): The path to the labels file
        labels (List[Dict[str, Any]]): The labels
        size (Tuple[int, int]): The size to resize the images to    
    """

    def __init__(self, root: Path, labels_path: Path, size=(1024, 1024)) -> "RSNA":
        """
        Initialize the dataset
        
        Args:
            root (Path): The root path to the dataset
            labels_path (Path): The path to the labels file
            size (Tuple[int, int]): The size to resize the images to
            
        Returns:
            RSNA: The dataset
        """


        super().__init__()

        self.root = root
        self.labels_path = labels_path

        with open(labels_path) as f:
            data = json.load(f)
            self.labels = data["datasets"][0]["annotations"]
        
        self.size = size
        self.resize = T.Resize(size)

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index: int) -> RSNAItem:
    
        item = RSNAItem.from_dict(self.root, self.labels[index])
        item.image = self.resize(item.image)

        return item
    

class RSNADataModule(L.LightningDataModule):
    """
    Represents the RSNA dataset, found at https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
    
    A Lightning DataModule is a way to organize your data for training, validation and testing used by PyTorch Lightning.

    Attributes:
        root (Path): The root path to the dataset
        labels_path (Path): The path to the labels file
        batch_size (int): The batch size
        size (Tuple[int, int]): The size to resize the images to
        num_workers (int): The number of workers to use for loading the data
    """

    def __init__(self, 
                 root: Path, 
                 labels_path: Path, 
                 batch_size: int = 32, 
                 size=(1024, 1024),
                 num_workers: int = 16,
                 collation: Callable[[List[RSNAItem], Optional[Any]], List[Tensor]] = None
                 ) -> "RSNADataModule":
        """
        Initialize the dataset
        
        Args:
            root (Path): The root path to the dataset
            labels_path (Path): The path to the labels file
            batch_size (int): The batch size
            size (Tuple[int, int]): The size to resize the images to
            num_workers (int): The number of workers to use for loading the data
            collation (Callable[[List[RSNAItem]], List[Tensor]], optional): The collation function to use. Defaults to None.

        Returns:
            RSNADataModule: The dataset
            
        """

        super().__init__()

        self.root = root
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.size = size
        self.num_workers = num_workers
        self.collation = collation or self.base_collation

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the dataset by loading it and splitting it into train, val and test sets.

        Args:
            stage (Optional[str], optional): The stage to setup. Defaults to None.
        """

        dataset = RSNA(self.root, self.labels_path, size=self.size)

        # Split dataset into train, val and test
        n = len(dataset)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val

        # Important, split on patient level
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, 
            [n_train, n_val, n_test]
        )

    def base_collation(self, batch: List[RSNAItem]) -> Tuple[Tensor, Tensor]:
        """
        Collate a batch of RSNAItems into a tuple of images and labels.

        Args:
            batch (List[RSNAItem]): The batch of RSNAItems

        Returns:
            Tuple[Tensor, Tensor]: The images and labels
        """

        images = torch.stack([item.image for item in batch])
        labels = torch.stack([item.label for item in batch])

        return images, labels
    
    def train_dataloader(self) -> DataLoader:
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collation
            )
    
    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collation
        )
    
    def test_dataloader(self) -> DataLoader:

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collation
        )


def simsiam_collation(batch: List[RSNAItem], augmentation: Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Returns a pair of images and their augmentations.
    
    Args:
        batch (List[RSNAItem]): The batch of RSNAItems
        augmentation (Callable[[Tensor], Tensor]): The augmentation function
        
    Returns:
        Tuple[Tensor, Tensor]: The images and their augmentations
    """

    images = torch.stack([item.image for item in batch])
    augmented_images = torch.stack([augmentation(item.image) for item in batch])

    return images, augmented_images

def anatomic_collation(batch: List[RSNAItem], patchify: Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Returns a pair of patches from images and a shuffled version of the patches.
    
    Args:
        batch (List[RSNAItem]): The batch of RSNAItems
        patchify (Callable[[Tensor], Tensor]): The patchify function

    Returns:
        Tuple[Tensor, Tensor]: The patches and their shuffled versions
    """
    batch_size = len(batch)
    images = torch.stack([item.image for item in batch])
    patches = patchify(images) # (batch_size*n_patches, 1, patch_size, patch_size)
    n_patches = patches.shape[0] // batch_size
 
    # Shuffle along the batch dimension
    # by simply shifting the indices by one
    # Combined with random shuffling of the batch this is a simple hack
    shuffled_patches = patches.clone().roll(n_patches, dims=0) # (batch_size, n_patches, patch_size, patch_size

    return patches, shuffled_patches


def build_datamodule(args: Namespace) -> RSNADataModule:
    """
    Build the RSNADataModule from the arguments.

    Args:
        args (Namespace): The arguments

    Returns:
        RSNADataModule: The RSNADataModule
    """

    return RSNADataModule(
        root=args.root,
        labels_path=args.labels_path,
        batch_size=args.batch_size,
        size=(args.size, args.size),
        num_workers=args.num_workers
    )