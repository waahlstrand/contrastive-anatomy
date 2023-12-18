from typing import *
import lightning as L

from .simsiam import SimSiam, AnatomicSimSiam

def build_model(args: Any) -> L.LightningModule:

    if args.model == "simsiam":
        model = SimSiam(
            dim=args.dim, 
            prediction_dim=args.prediction_dim, 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay, 
            n_epochs=args.n_epochs, 
            )
        
    elif args.model == "anatomic_simsiam":

        model = AnatomicSimSiam(
            dim=args.dim, 
            prediction_dim=args.prediction_dim, 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay, 
            n_epochs=args.n_epochs, 
            )

    return model