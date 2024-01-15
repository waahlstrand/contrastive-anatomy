from typing import *
import lightning as L

from .simsiam import SimSiam, AnatomicSimSiam
from .classification import PatchwiseClassifier, ImageClassifier

def build_model(args: Any) -> L.LightningModule:

    encoder_kwargs = args.__dict__.pop("encoder_kwargs")

    if args.model == "simsiam":
        model = SimSiam(
            dim=args.dim, 
            prediction_dim=args.prediction_dim, 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay, 
            n_epochs=args.n_epochs,
            encoder_name=args.encoder_name,
            encoder_kwargs=encoder_kwargs
            )
        
    elif args.model == "anatomic_simsiam":

        model = AnatomicSimSiam(
            dim=args.dim, 
            prediction_dim=args.prediction_dim, 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay, 
            n_epochs=args.n_epochs, 
            encoder_name=args.encoder_name,
            encoder_kwargs=encoder_kwargs
            )

    return model

def build_classifier(args: Any) -> ImageClassifier:

    checkpoint_path = args.__dict__.pop("checkpoint_path")
    encoder_kwargs = args.__dict__.pop("encoder_kwargs")

    if args.model == "patchwise_with_labels":
        model = PatchwiseClassifier(
            n_classes=2,
            feature_extractor=AnatomicSimSiam.load_from_checkpoint(
                checkpoint_path,
                **vars(args)
            ),
            encoder_kwargs=encoder_kwargs,
            **vars(args)
        )
    else:
        raise NotImplementedError

    return model