from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .data import StocksDataModule
from .rnn import GRU


if __name__ == "__main__":
    pl.seed_everything(69)
    
    parser = ArgumentParser()
    parser = StocksDataModule.add_argparse_args(parser)
    parser = GRU.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()

    s_dm = StocksDataModule()
    model = GRU()
    
    """Not available""" 
    # earlystopping = EarlyStopping(monitor='val_loss',
    #                             min_delta=0.01,
    #                             patience=5,
    #                             verbose=False,
    #                             mode="min")
    
    trainer = Trainer.from_argparse_args(args)
    
    trainer.fit(model, s_dm)
    trainer.test(model, s_dm)

    
    
    
    
    
    



