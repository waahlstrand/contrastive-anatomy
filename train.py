from lightning.pytorch.cli import LightningCLI
# from models.simsiam import AnatomicSimSiam, SimSiam
from data.dataset import ImageDataModule
import torch.multiprocessing
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('medium')

def main():

    cli = LightningCLI(
        datamodule_class=ImageDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )

    
if __name__ == "__main__":

    main()