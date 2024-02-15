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
import numpy as np
from rich import print


# To plot rectangles
from matplotlib.patches import Rectangle

@dataclass
class Item:
    """
    Represents a single image with its label and bounding box, if any.
    """

    id: str
    path: Path | str
    image: Tensor
    label: Tensor
    bbox: Optional[Tensor | None]

    @classmethod
    def from_dict(cls, root: Path, d: Dict[Literal["patientId", "Target", "x", "y", "width", "height"], Union[str, float]]) -> "Item":
        """
        Create an Item from a dictionary formatted according to the official json file, 
        like e.g.:

        Args:
            root (Path): The root path to the dataset
            d (Dict[str, Any]): The dictionary from the labels file

        Returns:
            Item: The Item
        """

        patient_id = d["patientId"]
        x, y, w, h = d.get("x", 0), d.get("y", 0), d.get("width", 0), d.get("height", 0)
        label     = torch.tensor(d["Target"])

        # Create bbox
        if label:
            bbox = torch.tensor([x, y, w, h])
        else:
            bbox = None

        # Load image
        path = root / Path(patient_id + ".dcm")
        if not path.exists():
            path = root / Path(patient_id + ".dicom")
        
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        
        image = pydicom.dcmread(path).pixel_array

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image = torch.from_numpy(image)

        return cls(id, path, image, label, bbox)

    def to(self, device: torch.device) -> "Item":
        
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
    


class ImageDataset(Dataset):

    """
    Represents an image dataset for anomaly detection

    Attributes:
        root (Path): The root path to the dataset
        labels_path (Path): The path to the labels file
        labels (List[Dict[str, Any]]): The labels
        size (Tuple[int, int]): The size to resize the images to    
    """

    def __init__(self, images_root: Path, labels: pd.DataFrame, size=(1024, 1024)) -> "ImageDataset":
        """
        Initialize the dataset
        
        Args:
            root (Path): The root path to the dataset
            labels_path (Path): The path to the labels file
            size (Tuple[int, int]): The size to resize the images to
            
        Returns:
            ImageDataset: The dataset
        """


        super().__init__()

        self.images_root = images_root

        # Load labels and shuffle
        self.labels = labels
        self.labels = self.labels.sample(frac=1)

        self.n_labels = len(self.labels)
        
        self.size = size
        self.resize = T.Resize(224)

    @classmethod
    def from_list_of_images(cls, images_root: Path, images: List[Path], labels: List[Literal[0, 1]], size=(1024, 1024)) -> "ImageDataset":

        labels = pd.DataFrame({
            "patientId": [str(image) for image in images],
            "Target": labels
        })

        return cls(images_root, labels, size)  

    def __len__(self) -> int:
        return self.n_labels
    
    def __getitem__(self, index: int) -> Item:
    
        item = Item.from_dict(self.images_root, self.labels.iloc[index].to_dict())
    
        item.image = item.image.reshape(1, *item.image.shape).repeat(3, 1, 1)

        return item
    
def build_dataset(images_root: Path, splits_file: Path) -> Tuple[ImageDataset, ImageDataset]:
    """
    Build the dataset from the images root and the splits file.
    
    Args:
        images_root (Path): The root path to the images
        splits_file (Path): The path to the splits file formatted as {"train": { "0": [id1, id2, ...], "unlabeled": {"1": [...], "0": [...]}}, "test": { "1": [...], "0": [...]}}
        
    Returns:
        Tuple[RSNA, RSNA]: The train and test datasets
        
    """
    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_ids = splits["train"]["0"] # List[str]
    test_ids_negative = splits["test"]["0"] 
    test_labels_negative = [0] * len(test_ids_negative)
    test_ids_positive = splits["test"]["1"] # List[str]
    test_labels_positive = [1] * len(test_ids_positive)

    # Process to remove .png from each string
    train_ids = [id.replace(".png", "") for id in train_ids]
    test_ids_negative = [id.replace(".png", "") for id in test_ids_negative]
    test_ids_positive = [id.replace(".png", "") for id in test_ids_positive]

    test_ids = test_ids_negative + test_ids_positive
    test_labels = test_labels_negative + test_labels_positive
    
    # Get labels
    train_labels = [0] * len(train_ids)


    train_data = ImageDataset.from_list_of_images(images_root, train_ids, train_labels)
    test_data = ImageDataset.from_list_of_images(images_root, test_ids, test_labels)

    return train_data, test_data


class ImageDataModule(L.LightningDataModule):
    """
    
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
                 splits_file: Path,
                 batch_size: int = 32, 
                 size=(1024, 1024),
                 num_workers: int = 16,
                 remove_disease: bool = True,
                 train_fraction: float = 0.8,
                 ) -> "ImageDataModule":
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
            ImageDataModule: The dataset
            
        """

        super().__init__()

        self.root = root
        self.labels_path = labels_path
        self.splits_file = splits_file
        self.batch_size = batch_size
        self.size = size
        self.num_workers = num_workers
        self.remove_disease = remove_disease
        self.train_fraction = train_fraction

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the dataset by loading it and splitting it into train, val and test sets.

        Args:
            stage (Optional[str], optional): The stage to setup. Defaults to None.
        """

        dataset, self.test_dataset = build_dataset(self.root, self.splits_file)

        # Split dataset into train and val
        n_train = int(self.train_fraction * len(dataset))
        n_val = len(dataset) - n_train

        # Important, split on patient level
        # self.train_dataset, self.val_dataset = random_split(
        #     dataset, 
        #     [n_train, n_val]
        # )
        self.train_dataset = dataset
        self.val_dataset   = self.test_dataset

        # Print statistics
        print(f"Train size:\t{len(self.train_dataset)}")
        print(f"Val size:\t{len(self.val_dataset)}")
        print(f"Test size:\t{len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collation,
                pin_memory=True
            )
    
    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collation,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collation,
            pin_memory=True
        )
    
    def collation(self, batch: List[Item]) -> Tuple[Tensor, Tensor]:
        """
        Collate a batch of RSNAItems into a tuple of images and labels.

        Args:
            batch (List[RSNAItem]): The batch of RSNAItems

        Returns:
            Tuple[Tensor, Tensor]: The images and labels
        """

        images = torch.stack([item.image for item in batch])
        labels = torch.stack([item.label for item in batch])

        images = images.float() 

        return images, labels

def build_datamodule(args: Namespace) -> ImageDataModule:
    """
    Build the ImageDataModule from the arguments.

    Args:
        args (Namespace): The arguments

    Returns:
        ImageDataModule: The ImageDataModule
    """

    data_dir  = Path(args.root) / args.data_dirname
    labels_path = Path(args.root) / args.labels_filename

    return ImageDataModule(
        root=data_dir,
        labels_path=labels_path,
        batch_size=args.batch_size,
        size=(args.size, args.size),
        num_workers=args.num_workers,
        splits_file=Path(args.splits_file),
        train_fraction=args.train_fraction
    )