
import warnings
import time
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning import Trainer
import lightning as L

from data.rsna import build_datamodule
from models import build_model
from models import PlotCallback
from argparse import ArgumentParser

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

def train(args):

    # Set seed
    seed_everything(args.seed)

    torch.set_float32_matmul_precision('medium')

    # Logging
    if args.log:
        logger = WandbLogger(
            name=args.name + "-" + time.strftime("%Y-%m-%d-%H-%M-%S"),
            project="superb",
            config=args,
            save_dir=args.log_dir,
        )
    else:
        logger = None

    # Callbacks for image logging and checkpointing
    callbacks = [
        ModelCheckpoint(
            monitor="val_stage_total",
            filename="{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
            mode="min",
        ),
        RichProgressBar(),
        RichModelSummary(),

    ]

    # Set up and choose model
    model = build_model(args)

    # try:
    #     model = torch.compile(model, mode="reduce-overhead")
    # except:
    #     raise RuntimeError

    # Set up data module
    dm  = build_datamodule(args)

    # Set up trainer
    trainer = Trainer(
        accelerator="gpu",
        devices = [args.device],
        max_epochs=args.n_epochs,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run = args.debug if args.debug else False,
        # profiler="advanced" if args.debug else None,
    )

    # Train
    trainer.fit(model, dm)

    # Test
    # trainer.test(model, dm)


def main():

    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--name", type=str, default="simsiam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--n_patches_per_side", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--labels_path", type=str, default="labels.csv")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--val_dir", type=str, default="val")
    parser.add_argument("--test_dir", type=str, default="test")
    

    # Get arguments
    args = parser.parse_args()

    # Train
    train(args)


if __name__ == "__main__":

    main()