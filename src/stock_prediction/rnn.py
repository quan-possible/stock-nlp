import numpy as np
import pandas as pd
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.functional as F

import pytorch_lightning as pl
import torchmetrics
from sklearn.preprocessing import StandardScaler


class GRU(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--version", type=float,
                            help="specify it in a form X.XX")
        parser.add_argument("--input_size", type=int, default=256)
        parser.add_argument("--output_size", type=bool, default=False)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--hidden_size", type=int, default=5,
                            help="hidden_size of the GRU")
        parser.add_argument("--num_layers", type=int, default=1,
                            help="Number of layers in the GRU.")
        parser.add_argument("--learning_rate", type=float, default=1e-3,
                            help="Learning rate.")
        parser.add_argument("--max_epochs", type=int, default=1000,
                            help="The total number of training epochs.")
                            
        return parser

    def __init__(
        self, hparams, scaler=StandardScaler(), 
        criterion=nn.BCEWithLogitsLoss
    ):
        super().__init__()
        self.hparams = hparams
        self.criterion = criterion
        self.X_scaler, self.y_scaler = scaler,scaler

        self.gru = nn.GRU(
            input_size=self.hparams.input_size, 
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True, 
            dropout=self.hparams.dropout
        )

        self.main = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hparams.input_size * 
                self.hparams.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x, _ = self.gru(x)
        return self.main(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y, y_hat = y.squeeze(), y_hat.squeeze()

        y_inv = torch.tensor(self.y_scaler.inverse_transform(y.cpu()),
                             device=y.device)
        y_hat_inv = torch.tensor(self.y_scaler.inverse_transform(y_hat.cpu()),
                                 device=y.device)

        loss = torch.sqrt(self.criterion(y_hat_inv, y_inv))
        acc = torchmetrics.functional.accuracy(y_hat_inv, y_inv)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y, y_hat = y.squeeze(), y_hat.squeeze()

        y_inv = torch.tensor(self.y_scaler.inverse_transform(y.cpu()),
                             device=y.device)
        y_hat_inv = torch.tensor(self.y_scaler.inverse_transform(y_hat.cpu()),
                                 device=y.device)

        loss = torch.sqrt(self.criterion(y_hat_inv, y_inv))
        acc = torchmetrics.functional.accuracy(y_hat_inv, y_inv)

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    


    



