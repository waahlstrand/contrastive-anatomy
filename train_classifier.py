
import warnings
import time
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning import Trainer
import lightning as L

from data.rsna import build_datamodule
from models import build_classifier
from argparse import ArgumentParser
import yaml

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
            project="anatomy-classification",
            config=args,
            save_dir=args.log_dir,
        )
    else:
        logger = None

    # Callbacks for image logging and checkpointing
    callbacks = [
        ModelCheckpoint(
            monitor="val_stage_loss",
            filename="{epoch:02d}",
            save_top_k=2,
            mode="min",
        ),
        RichProgressBar(),
        RichModelSummary(),

    ]

    # Set up and choose model
    model = build_classifier(args)

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
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log", action=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--n_patches_per_side", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--prediction_dim", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--target_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--labels_filename", type=str, default=None)
    parser.add_argument("--data_dirname", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--encoder_kwargs", type=str, default=None)
    
    # Add config arguments
    parser.add_argument("--config", type=str, default=None)

    # Get arguments
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Keys in args should overwrite keys in config
        for key, value in config.items():
            if not args.__dict__[key]:
                args.__dict__[key] = value
    # Train
    train(args)

if __name__ == "__main__":

    main()