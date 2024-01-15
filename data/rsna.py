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
from models.patchwise import Patchify


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

        Args:
            root (Path): The root path to the dataset
            d (Dict[str, Any]): The dictionary from the labels file

        Returns:
            RSNAItem: The RSNAItem
        """

        patient_id = d["patientId"]
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        label     = torch.tensor(d["Target"])

        # Create bbox
        if label == 1:
            bbox = torch.tensor([x, y, w, h])
        else:
            bbox = None

        # Load image
        path = root / Path(patient_id + ".dcm")
        image = torch.from_numpy(pydicom.dcmread(path).pixel_array)


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


class RSNAStatistics:
    WIDTH: int = 1024
    HEIGHT: int = 1024
    MEAN: float
    STD: float   

class RSNA(Dataset):

    """
    Represents the RSNA dataset, found at https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

    Attributes:
        root (Path): The root path to the dataset
        labels_path (Path): The path to the labels file
        labels (List[Dict[str, Any]]): The labels
        size (Tuple[int, int]): The size to resize the images to    
    """

    def __init__(self, images_root: Path, labels_path: Path, size=(1024, 1024), remove_disease: bool = False) -> "RSNA":
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

        self.images_root = images_root
        self.labels_path = labels_path # CSV file

        # Load labels
        self.labels = pd.read_csv(labels_path)

        if remove_disease:
            self.labels = self.labels[self.labels.Target == 0]

        self.n_labels = len(self.labels)
        
        self.size = size
        self.resize = T.Resize(size)

    def __len__(self) -> int:
        return self.n_labels
    
    def __getitem__(self, index: int) -> RSNAItem:
    
        item = RSNAItem.from_dict(self.images_root, self.labels.iloc[index].to_dict())

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
                 remove_disease: bool = True,
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
        self.remove_disease = remove_disease
        self.collation = collation or self.base_collation

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the dataset by loading it and splitting it into train, val and test sets.

        Args:
            stage (Optional[str], optional): The stage to setup. Defaults to None.
        """

        dataset = RSNA(self.root, self.labels_path, size=self.size, remove_disease=self.remove_disease)

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

    images = torch.stack([item.image for item in batch]).view(-1, 1, RSNAStatistics.WIDTH, RSNAStatistics.HEIGHT)
    augmented_images = augmentation(images)

    # Normalize images
    images = images / 255.0
    augmented_images = augmented_images / 255.0

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

    # Normalize images
    patches = patches / 255.0
    shuffled_patches = shuffled_patches / 255.0

    return patches, shuffled_patches


def patchwise_with_labels(batch: List[RSNAItem], patchify: Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Returns a pair of patches from images and their labels.
    
    Args:
        batch (List[RSNAItem]): The batch of RSNAItems
    
    """

    batch_size = len(batch)
    images = torch.stack([item.image for item in batch])
    patches = patchify(images) # (batch_size*n_patches, 1, patch_size, patch_size)
    n_patches = patches.shape[0] // batch_size
    # patches = patches.view(batch_size, n_patches, *patches.shape[-2:]) # (batch_size, n_patches, patch_size, patch_size)

    # Labels
    labels = torch.stack([item.label for item in batch])

    # Normalize images
    patches = patches / 255.0

    return patches, labels

def build_datamodule(args: Namespace) -> RSNADataModule:
    """
    Build the RSNADataModule from the arguments.

    Args:
        args (Namespace): The arguments

    Returns:
        RSNADataModule: The RSNADataModule
    """

    data_dir  = Path(args.root) / args.data_dirname
    labels_path = Path(args.root) / args.labels_filename

    if args.model == "anatomic_simsiam":

        patchify = Patchify.from_n_patches(args.n_patches_per_side, image_size=(RSNAStatistics.WIDTH, RSNAStatistics.HEIGHT))
        collation = lambda batch: anatomic_collation(batch, patchify)

    elif args.model == "patchwise_with_labels":

        patchify = Patchify.from_n_patches(args.n_patches_per_side, image_size=(RSNAStatistics.WIDTH, RSNAStatistics.HEIGHT))
        collation = lambda batch: patchwise_with_labels(batch, patchify)

    elif args.model == "simsiam":
        augmentation = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.RandomResizedCrop((int(0.5*args.size), int(0.5*args.size))),
            T.Resize((int(args.target_size), int(args.target_size))),
        ])
        collation = lambda batch: simsiam_collation(batch, augmentation)
    else:
        raise ValueError(f"Collation {args.collation} not recognized.")

    return RSNADataModule(
        root=data_dir,
        labels_path=labels_path,
        batch_size=args.batch_size,
        size=(args.size, args.size),
        num_workers=args.num_workers,
        collation=collation
    )