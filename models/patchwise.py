from typing import *
import torch
from torch import Tensor
import torch.nn as nn

class Patchify(nn.Module):
    """
    A module that divides an image into a grid of patches.
    
    Attributes:
        patch_size (Tuple[int, int]): The size of the patches
        
    """

    def __init__(self, patch_size: Tuple[int, int]):
        """
        Initialize a Patchify module.

        Args:
            patch_size (Tuple[int, int]): The size of the patches
            
        """

        super().__init__()
        self.patch_size = patch_size

    @classmethod
    def from_n_patches(cls, n_patches_per_side: int, image_size: Tuple[int, int] = (256, 256)):
        """
        Create a Patchify module that divides the image into n_patches_per_side^2 patches.
        
        Args:
            n_patches_per_side (int): The number of patches per side
            image_size (Tuple[int, int], optional): The size of the image. Defaults to (256, 256).
            
        Returns:
            Patchify: The Patchify module
            
        """

        height = image_size[0] // n_patches_per_side
        width = image_size[1] // n_patches_per_side

        return cls(patch_size=(height, width))

    def forward(self, x: Tensor) -> Tensor:

        return self.patchify(x, self.patch_size)
    
    def patchify(self, image: Tensor, patch_size: int) -> Tensor:
        """
        Divide an image into patches of size patch_size, using torch.unfold.

        Args:
            image (Tensor): The image to divide into patches
            patch_size (int): The size of the patches

        Returns:
            Tensor: The patches of the image
        """
        
        image = image.squeeze(1)
        batch_size, height, width = image.shape
        image       = image.unfold(1, *patch_size).unfold(2, *patch_size).reshape(-1, 1, *patch_size)
        positions   = torch.arange(image.shape[0] // batch_size).repeat(2, 1).reshape(-1,1)

        return image, positions